import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning

from load import ACTS, load_acts, load_table, domain_of
from myprobe import agent_vectors, fit_direction, agent_scores

SETUP = ["colluder", "honest+motive", "honest+tactics", "honest+secret", "honest+channel"]

def auc(pos,neg):
    return roc_auc_score([1]*len(pos)+[0]*len(neg),np.concatenate([pos,neg]))

def setup_direction(smeta,sacts,layer,exclude_domain,k):
    """Top-k directions along which ADDING set-up text moves the activation. Training domains only"""
    A = sacts[f"layer_{layer}"]
    row = {(m["run_id"],m["agent_name"],m["variant"]): j for j,m in enumerate(smeta)}

    agents = sorted({(m["run_id"], m["agent_name"]) for m in smeta})

    diffs = [A[row[(r,n,v)]] - A[row[r,n,"honest"]]
            for r,n in agents if domain_of(r.split("__")[1]) != exclude_domain
            for v in SETUP]
    
    _,_,vt = np.linalg.svd(np.array(diffs),full_matrices=False)
    return vt[:k].T

def erase(M,U):
    """Remove from every row of M its part inside the span of U's columns"""
    return M - (M@U) @ U.T

def attempt_a(dmeta,dacts,smeta,sacts,stable,ssacts,cell_of,layer,ks):
    domain = {m["run_id"]: domain_of(m["scenario_id"]) for m in dmeta}
    mode = {m["run_id"]: m["mode"] for m in dmeta} 

    for k in ks:
        runs, before = [],[]

        for d in sorted(set(domain.values())):
            U = setup_direction(smeta,sacts,layer,d,k)
            acts2 = {f"layer_{layer}": erase(dacts[f"layer_{layer}"],U)}

            scalar,w = fit_direction(dmeta,acts2,layer,exclude_domain=d,seed=0)
            vec = agent_vectors(dmeta,acts2,layer)
            ref = np.array(list(agent_scores({q:v for q,v in vec.items() if domain[q[0]] != d and mode[q[0]] == "control"},scalar,w).values()))

            for (rid,_),s in agent_scores({q:v for q,v in vec.items() if domain[q[0]] == d},scalar,w).items():
                runs.append({"run_id": rid, "z": (s - ref.mean()) / ref.std()})

            # so bacially before speaking: erasing system-prompt vectors too, score with the SAME arrow

            S = erase(ssacts[f"layer_{layer}"],U)
            sv = {(r.run_id,r.agent_name): S[r.i] for r in stable.itertuples()}

            s_all = agent_scores(sv,scalar,w)
            sref = np.array([v for q,v in s_all.items() if domain[q[0]] != d and mode[q[0]] == "control"])

            for (rid, ag), v in s_all.items():
                if domain[rid] == d:
                    before.append({"run_id": rid, "agent": ag, "z": (v - sref.mean()) / sref.std()})

        run = pd.DataFrame(runs).groupby("run_id").z.max().reset_index()
        run["cell"] = run.run_id.map(cell_of)
        b = pd.DataFrame(before).merge(stable[["run_id", "agent_name", "role", "cell"]]
                                       .rename(columns={"agent_name": "agent"}), on=["run_id", "agent"])
        
        g = lambda c: run[run.cell == c].z

        print(f"k={k:2d}  before speaking {auc(b[b.role == 'colluder'].z, b[(b.role == 'honest') & (b.cell == 'control')].z):.3f}"
              f"   failed-vs-control {auc(g('told_didnt'), g('control')):.3f}"
              f"   did-vs-failed {auc(g('told_did'), g('told_didnt')):.3f}")


def attempt_b(dmeta, dacts, stable, ssacts, cell_of, layers):
    """Colluders only (all were told). Label = did their run succeed? Held out by domain"""
    role = {(m["run_id"], m["agent_name"]): m["role"] for m in dmeta}
    domain = {m["run_id"]: domain_of(m["scenario_id"]) for m in dmeta}
    for layer in layers:
        sources = {"discussion": agent_vectors(dmeta, dacts, layer),
                   "before speaking": {(r.run_id, r.agent_name): ssacts[f"layer_{layer}"][r.i]
                                       for r in stable.itertuples()}}
        for name, vec in sources.items():
            keys = [k for k in vec if role.get(k) == "colluder"]
            X = np.array([vec[k] for k in keys])
            y = np.array([cell_of[k[0]] == "told_did" for k in keys]).astype(int)     # 1 = run succeeded
            dom = np.array([domain[k[0]] for k in keys])
            pred = np.zeros(len(keys))

            for d in sorted(set(dom)):
                tr, te = dom != d, dom == d
                scaler = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=25, max_iter=8, solver="liblinear", random_state=0)
                clf.fit(scaler.transform(X[tr]), y[tr])
                pred[te] = clf.decision_function(scaler.transform(X[te]))

            print(f"layer {layer}  {name:15s}  success vs failure AUROC {roc_auc_score(y, pred):.3f}")

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    dmeta, dacts = load_acts(ACTS / "core_reextract")
    table, _ = load_table(ACTS / "core_reextract")
    cell_of = dict(zip(table.run_id, table.cell))
    smeta, sacts = load_acts(ACTS / "surgery")
    stable, ssacts = load_table(ACTS / "sysonly")
    stable = stable[stable.set == "core"]
    attempt_a(dmeta, dacts, smeta, sacts, stable, ssacts, cell_of, layer=28, ks=[0, 1, 5, 20, 50])
    attempt_b(dmeta, dacts, stable, ssacts, cell_of, layers=[26, 28, 30])