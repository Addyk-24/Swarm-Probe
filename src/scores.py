import numpy as np
import pandas as pd

from load import ROOT, ACTS, load_table, domain_of
from myprobe import agent_vectors, fit_direction, agent_scores, heldout_auroc

import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

def heldout_scores(meta, acts, layers, seeds,hide_runs=()):
    mode = {r["run_id"]: r["mode"] for r in meta}
    domain = {r["run_id"]: domain_of(r["scenario_id"]) for r in meta}
    role = {(r["run_id"],r["agent_name"]): r["role"] for r in meta}

    rows = []
    train_meta = [dict(r,mode="hidden") if r["run_id"] in hide_runs else r for r in meta]

    

    for layer in layers:
        vectors = agent_vectors(meta,acts,layer)
        for seed in seeds:
            for d in sorted(set(domain.values())):
                scaler,w = fit_direction(train_meta,acts,layer,exclude_domain=d,seed=seed)

                # ruler: control agents from the OTHER domains
                ref = {k:v for k,v in vectors.items() if domain[k[0]] != d and mode[k[0]] == "control"}
                ref_scores = np.array(list(agent_scores(ref,scaler,w).values()))
                mu,sd = ref_scores.mean(), ref_scores.std()
                test = {k:v for k,v in vectors.items() if domain[k[0]] == d}

                for (rid,agent), s in agent_scores(test,scaler,w).items():
                    rows.append({
                        "run_id":rid,
                        "agent": agent, 
                        "role": role[(rid,agent)],         
                        "domain": d, 
                        "layer": layer, 
                        "seed": seed,
                        "score": s, 
                        "score_z": (s - mu) / sd
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    FULL = "--full" in sys.argv
    STRICT = "--strict" in sys.argv
    LAYERS = [26, 27, 28, 29, 30] if FULL else [28]
    SEEDS = list(range(10)) if FULL else [0]

    table, acts = load_table(ACTS / "core_reextract")
    meta = table.to_dict("records")
    hide = set(table[table.cell == "told_didnt"].run_id) if STRICT else set()
    df = heldout_scores(meta, acts, LAYERS, SEEDS, hide_runs=hide)

    cells = table.drop_duplicates("run_id")[["run_id", "cell"]]
    df = df.merge(cells, on="run_id", validate="many_to_one")

    print(len(df), "rows, expect", 400 * len(LAYERS) * len(SEEDS))
    print(df.groupby("role").score_z.mean().round(2).to_dict(), "(colluders >> honest)")

    one = df[(df.layer == 28) & (df.seed == 0)]

    if not STRICT:
        run = one.groupby(["run_id", "domain", "cell"]).score.max().reset_index()   # run score = max agent
        run["y"] = (run.cell != "control").astype(int)
        from_table = {d: roc_auc_score(g.y, g.score) for d, g in run.groupby("domain")}
        from_func = heldout_auroc(meta, acts, 28, seed=0)
        print("per-fold AUROC matches heldout_auroc:",
              all(np.isclose(from_table[d], from_func[d]) for d in from_func))

    run_z = one.groupby(["run_id", "cell"]).score_z.max().reset_index()
    print(run_z.groupby("cell").score_z.agg(["mean", "count"]).round(2))

    if FULL:
        out = ROOT / "results" / ("scores_core_strict.csv" if STRICT else "scores_core.csv")
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print("saved", out)

