import numpy as np
import pandas as pd

from load import ROOT,ACTS, load_acts, load_table
from myprobe import fit_direction, agent_scores

import sys, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

def sysonly_vectors(table,acts,layer):
    """One vector per agent: its system-prompt-only row. No averaging: each agent has exactly 1 row"""
    rows = acts[f"layer_{layer}"]

    return {
        (r.run_id,r.agent_name) : rows[r.i] for r in table.itertuples()
    }

def prompt_only_scores(dmeta,dacts,stable,sacts,layers,seeds):
    """Arrow learned on DISCUSSION (held out by domain), applied to SYSTEM-PROMPT vectors"""
    domain = dict(zip(stable.run_id,stable.domain))
    mode = dict(zip(stable.run_id,stable["mode"]))

    rows = []

    for layer in layers:
        vectors = sysonly_vectors(stable,sacts,layer)
        for seed in seeds:
            for d in sorted(set(domain.values())):
                scalar,w = fit_direction(dmeta,dacts,layer,exclude_domain=d,seed=seed)

                ref = {k:v for k,v in vectors.items() if domain[k[0]] != d and mode[k[0]] == "control"}
                ref_scores = np.array(list(agent_scores(ref,scalar,w).values()))

                mu,sd = ref_scores.mean(), ref_scores.std()
                test = {k:v for k,v in vectors.items() if domain[k[0]] == d}

                

                for (rid,agent), s in agent_scores(test,scalar,w).items():
                    rows.append({
                        "run_id": rid, 
                        "agent": agent, 
                        "layer": layer, 
                        "seed": seed,
                        "score_z": (s - mu) /  sd
                    })
    df  = pd.DataFrame(rows)
    info = stable[["run_id","agent_name","role","cell"]].rename(columns={"agent_name": "agent"})

    return df.merge(info,on=["run_id","agent"],validate="many_to_one")
    



if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    FULL = "--full" in sys.argv
    LAYERS = [26, 27, 28, 29, 30] if FULL else [28]
    SEEDS = list(range(10)) if FULL else [0]

    dmeta, dacts = load_acts(ACTS / "core_reextract")
    stable, sacts = load_table(ACTS / "sysonly")
    stable = stable[stable.set == "core"]

    df = prompt_only_scores(dmeta, dacts, stable, sacts, LAYERS, SEEDS)
    print(len(df), "rows, expect", 400 * len(LAYERS) * len(SEEDS))

    ag = df.groupby(["run_id", "agent", "role", "cell", "layer"]).score_z.mean().reset_index()
    for layer, g in ag.groupby("layer"):
        col = g[g.role == "colluder"].score_z
        hon = g[(g.role == "honest") & (g.cell == "control")].score_z
        auc = roc_auc_score([1] * len(col) + [0] * len(hon), np.concatenate([col, hon]))
        print(f"layer {layer}  colluder vs honest (before speaking) AUROC {auc:.3f}")
    print(ag.groupby(["role", "cell"]).score_z.mean().round(2))

    if FULL:
        df.to_csv(ROOT / "results" / "scores_prompt_only.csv", index=False)

    

