import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load import ROOT, ACTS, load_table

def auc(pos,neg):
    """AUROC: how often a `pos` value is higher than a `neg` value"""

    return roc_auc_score([1]*len(pos) + [0]* len(neg),np.concatenate([pos,neg]))

def lengths():
    """One length per agent: discussion (mean n_tokens over its 3 messages) and before speaking"""

    t,_ = load_table(ACTS / "core_reextract")
    disc = t.groupby(["run_id","agent_name"]).n_tokens.mean().reset_index()

    s,_ = load_table(ACTS / "sysonly")
    s = s[s.set == "core"]

    pre = s[["run_id","agent_name","n_tokens"]]
    rename = {"agent_name": "agent", "n_tokens":"len"}

    return disc.rename(columns=rename), pre.rename(columns=rename)

def length_check(d,name):
    """d has columns role, cell, score_z, len (one row per agent)"""
    col = d[d.role == "colluder"]
    hon = d[(d.role == "honest") & (d.cell == "control")]
    print(f"{name}: length AUROC {auc(col.len,hon.len):.3f}  probe AUROC {auc(col.score_z,hon.score_z):.3f}")

    # Test A: length effect learned on honest agents only, then removed from everyone
    h = d[d.role == "honest"]

    slope, intercept = np.polyfit(h.len,h.score_z,1)
    resid = d.score_z - (slope*d.len + intercept)

    print(f"   A: slope per 100 tokens {slope * 100:.2f}   AUROC after removing length "
          f"{auc(resid[d.role == 'colluder'], resid[(d.role == 'honest') & (d.cell == 'control')]):.3f}")


    # Test B: only the overlapping length range (short colluders, long honest agents)
    lo,hi = col.len.quantile(0.1), hon.len.quantile(0.9)

    c,h2 = col[col.len <= hi], hon[hon.len >= lo]

    print(f"   B: overlap n={len(c)}+{len(h2)}   length AUROC {auc(c.len, h2.len):.3f}   "
          f"probe AUROC {auc(c.score_z, h2.score_z):.3f}")



if __name__ == "__main__":
    disc_len, pre_len = lengths()
    ag = pd.read_csv(ROOT / "results" / "agents_core.csv").merge(disc_len, on=["run_id", "agent"])
    po = (pd.read_csv(ROOT / "results" / "scores_prompt_only.csv")
            .groupby(["run_id", "agent", "role", "cell"]).score_z.mean().reset_index()
            .merge(pre_len, on=["run_id", "agent"]))
    length_check(ag, "discussion")
    length_check(po, "before speaking")
