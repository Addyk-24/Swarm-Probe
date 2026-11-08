import numpy as np
import pandas as pd

from load import ROOT, ACTS, load_acts, load_table, domain_of
from myprobe import fit_direction, agent_scores
from prompt_only import sysonly_vectors

import sys, warnings
from sklearn.exceptions import ConvergenceWarning

ORDER = {
    "surgery":  ["honest", "honest+filler", "identity", "honest+motive", "honest+tactics",
                 "motive", "tactics", "honest+secret", "honest+channel", "colluder"],
    "surgery2": ["honest", "honest+partner", "honest+groupchannel", "honest+privatenotes", "honest+privatechannel", "honest+channel", "colluder"],
    "surgery3": ["honest", "honest+agenda", "honest+meeting", "honest+member", "colluder"],
}

def surgery_scores(dmeta, dacts, smeta, sacts, stable, ssacts, layers, seeds):
    """Discussion arrow (domain held out) applied to the surgery prompts.
    Ruler = sysonly control agents of the training domains from previous code"""

    sdomain = dict(zip(stable.run_id,stable.domain))
    smode = dict(zip(stable.run_id,stable["mode"]))
    # what basically doing: deception__audit_01__collusion' -> 'audit'
    dom_of_row = [domain_of(m["run_id"].split("__")[1]) for m in smeta]

    rows = []

    for layer in layers:
        ruler_vecs = sysonly_vectors(stable,ssacts,layer)
        for seed in seeds:
            for d in sorted(set(dom_of_row)):
                scalar,w = fit_direction(dmeta,dacts,layer,exclude_domain=d,seed=seed)

                ref = {k:v for k,v in ruler_vecs.items() if sdomain[k[0]] != d and smode[k[0]] == "control"}
                r = np.array(list(agent_scores(ref,scalar,w).values()))

                mu,sd = r.mean(),r.std()
                test = {j:sacts[f"layer_{layer}"][j] for j in range(len(smeta)) if dom_of_row[j] == d}

                for j,s in agent_scores(test,scalar,w).items():
                    rows.append({"j": j, "layer": layer, "seed": seed, "score_z": (s - mu) / sd })

    # j = row number in the surgery arrays
    info = pd.DataFrame(smeta).assign(j=range(len(smeta)))
    return pd.DataFrame(rows).merge(info, on="j", validate="many_to_one")


def summarise(df,order):
    per = df.groupby(["j", "variant", "run_id", "agent_name", "n_tokens"]).score_z.mean().reset_index()
    wide = per.pivot(index=["run_id", "agent_name"], columns="variant", values="score_z")
    out = pd.DataFrame({
        "tokens": per.groupby("variant").n_tokens.mean(),
        "mean_z": per.groupby("variant").score_z.mean(),
        "share_of_gap": wide.sub(wide["honest"], axis=0).div(wide["colluder"] - wide["honest"], axis=0).median(),
        "above_honest": wide.gt(wide["honest"], axis=0).mean(),
    }).loc[order]
    return out.round(2), wide


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    FULL = "--full" in sys.argv

    NAME =  "surgery3" if "--neutral" in sys.argv else "surgery2" if "--controls" in sys.argv else "surgery"
    LAYERS = [26, 27, 28, 29, 30] if FULL else [28]
    SEEDS = list(range(10)) if FULL else [0]

    dmeta, dacts = load_acts(ACTS / "core_reextract")
    smeta, sacts = load_acts(ACTS / NAME)
    stable, ssacts = load_table(ACTS / "sysonly")
    stable = stable[stable.set == "core"]

    df = surgery_scores(dmeta, dacts, smeta, sacts, stable, ssacts, LAYERS, SEEDS)
    print(len(df), "rows, expect", len(smeta) * len(LAYERS) * len(SEEDS))
    table, wide = summarise(df,ORDER[NAME])
    print(table.to_string())
    if FULL:
        df.to_csv(ROOT / "results" / f"scores_{NAME}.csv", index=False)
