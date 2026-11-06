import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load import ACTS, RELEASED, load_acts, load_table
from myprobe import fit_direction, agent_scores


def auc(pos, neg):
    return roc_auc_score(
        [1]*len(pos) + [0]*len(neg),np.concatenate([pos,neg])
    )

def score_transfer(parts,dmeta,dacts,layers,seeds):
    """parts = list of (table, acts). Arrow from ALL Core discussion; ruler = control agents in `parts`"""

    table = pd.concat([t for t,_ in parts],ignore_index=True)

    mode = dict(zip(table.run_id,table["mode"]))
    rows = []

    for layer in layers:
        vectors = {}
        for t,a in parts:
            vectors.update({(r.run_id,r.agent_name): a[f"layer_{layer}"][r.i] for r in t.itertuples()})

        for seed in seeds:
            scalar,w = fit_direction(dmeta,dacts,layer,exclude_domain=None,seed=seed)
            s = agent_scores(vectors,scalar,w)
            ref = np.array([v for k,v in s.items() if mode[k[0]] == "control"])

            for (rid,agent),v in s.items():
                rows.append({"run_id": rid, "agent": agent, "layer": layer, "seed": seed,
                             "score_z": (v - ref.mean()) / ref.std()})
                
    info = table[["run_id","agent_name","role","cell"]].rename(columns={"agent_name": "agent"})
    return pd.DataFrame(rows).merge(info, on=["run_id", "agent"], validate="many_to_one")


def summary(df,name):
    ag = df.groupby(["run_id", "agent", "role", "cell"]).score_z.mean().reset_index()
    run = ag.groupby(["run_id", "cell"]).score_z.max().reset_index()

    print(f"{name}: {len(df)} rows")
    print(ag.groupby(["cell", "role"]).score_z.mean().round(2).to_string())

    for c in ["told_unjudged", "implicit"]:
        print(f"   run AUROC {c} vs control: {auc(run[run.cell == c].score_z, run[run.cell == 'control'].score_z):.3f}")

    imp = ag[ag.cell == "implicit"]
    print(f"   inside implicit runs, implicit agents vs honest agents: "
          f"{auc(imp[imp.role == 'implicit'].score_z, imp[imp.role == 'honest'].score_z):.3f}")

if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    dmeta, dacts = load_acts(ACTS / "core_reextract")
    LAYERS, SEEDS = [26, 27, 28, 29, 30], list(range(10))

    during = score_transfer([load_table(RELEASED / "transfer"), load_table(ACTS / "transfer_implicit")],
                            dmeta, dacts, LAYERS, SEEDS)
    summary(during, "during the conversation (final message)")

    st, sa = load_table(ACTS / "sysonly")
    st = st[st.set == "transfer"]
    before = score_transfer([(st, sa)], dmeta, dacts, LAYERS, SEEDS)
    summary(before, "before speaking (system prompt only)")
