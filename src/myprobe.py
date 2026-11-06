from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


from load import domain_of

def agent_vectors(meta, acts,layer):

    table = acts[f"layer_{layer}"]
    folder = defaultdict(list)


    for i,row in enumerate(meta):
        if row["phase"] == "discussion" and row["channel"] == "public":
            key = (row["run_id"],row["agent_name"])
            folder[key].append(table[i])

    out = {}
    for keys,rows in folder.items():
        out[keys] = np.mean(rows,axis=0)

    return out

def fit_direction(meta, acts, layer, exclude_domain=None, seed=0):

    training_rows = []
    

    for i,row in enumerate(meta):
        if row["mode"] == "collusion" and row["channel"] == "public" and row["phase"] == "discussion":
            if exclude_domain is None or domain_of(row["scenario_id"]) != exclude_domain:
                training_rows.append(i)

    col_idx = []
    hon_idx = []

    for i in training_rows:
        if meta[i]["role"] == "colluder":
            col_idx.append(i)
        elif meta[i]["role"] == "honest":
            hon_idx.append(i)
    
    nc = min(len(col_idx),len(hon_idx))

    hon_idx = np.random.RandomState(seed).choice(hon_idx,nc,replace=False)

    col_idx = col_idx[:nc]

    X = np.concatenate([
        acts[f"layer_{layer}"][hon_idx],
        acts[f"layer_{layer}"][col_idx],
    ],axis=0)

    y = [0]*nc + [1]*nc

    scalar = StandardScaler()

    X = scalar.fit_transform(X)

    lg = LogisticRegression(C=25,max_iter=8,solver="liblinear",random_state=seed)

    probe = lg.fit(X,y)

    w = probe.coef_[0]
    w /= np.linalg.norm(w)

    return scalar, w


def agent_scores(vectors,scalar,w):
    scores = {}

    for key, v in vectors.items():
        score = w @  scalar.transform(v.reshape(1,-1))[0]

        scores[key] = score

    return scores

def heldout_auroc(meta, acts, layer, seed=0):
    vector = agent_vectors(meta,acts,layer)
    mode = {r["run_id"]:r["mode"] for r in meta}
    domain = {r["run_id"]:domain_of(r["scenario_id"]) for r in meta}

    out = {}

    for d in sorted(set(domain.values())):
        rids = sorted(r for r in mode if domain[r] == d)
        y = [1 if mode[r] == "collusion" else 0 for r in rids]
        if len(set(y)) < 2:
            continue
        scaler, w = fit_direction(meta,acts,layer,exclude_domain=d,seed=seed)
        test = {key:v for key,v in vector.items() if domain[key[0]] == d}
        scores = agent_scores(test,scaler,w)
        run_scores = [max(scores[key] for key in scores if key[0] == r) for r in rids]
        out[d] = roc_auc_score(y,run_scores)

    return out

if __name__ == "__main__":
    import sys
    from load import ROOT, ACTS, load_acts
    sys.path.insert(0, str(ROOT / "third_party" / "narcbench" / "reproduce"))
    from reproduce_LR import build_groups

    meta, acts = load_acts(ACTS / "core_reextract")
    mine = agent_vectors(meta, acts, 28)
    print(len(mine), "agents")                                   # expect 400
    print(mine[(meta[0]["run_id"], "Imani Okafor")][:3])         # expect about [-1.10 -2.16 -0.58]

    theirs = build_groups(meta, acts, 28)
    ok = all(np.allclose(mine[(rid, name)], vec)
             for rid, g in theirs.items() for name, vec in g["agents"].items())
    print("matches build_groups:", ok)


    # ---- Function 2 test: my arrow vs their arrow ----
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from reproduce_LR import train_deception_probe
    warnings.filterwarnings("ignore", category=ConvergenceWarning)          # max_iter=8 on purpose
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    for domain, seed in [("cyber", 0), ("med", 1), ("fin", 2), (None, 0)]:
        my_scaler, my_w = fit_direction(meta, acts, 28, exclude_domain=domain, seed=seed)
        their_scaler, their_w = train_deception_probe(meta, acts, 28, exclude_domain=domain, seed=seed)
        dot = abs(my_w @ their_w)                    # both length 1, so 1.0 = same arrow
        same_scaler = np.allclose(my_scaler.mean_, their_scaler.mean_)
        print(f"exclude {str(domain):5s} seed {seed}  w dot {dot:.4f}  same scaler {same_scaler}")
        # expect: w dot 1.0000, same scaler True on every line

    # ---- Function 3 test: my run scores vs their compute_scores ----
    # Use THEIR scaler and arrow for both, so only Function 3 is being tested.
    from reproduce_LR import compute_scores
    scaler, w = fit_direction(meta, acts, 28, exclude_domain="cyber", seed=0)
    my_scores = agent_scores(mine, scaler, w)              # {(run_id, agent_name): score}
    print(len(my_scores), "agent scores")                  # expect 400

    rids = sorted(rid for rid, g in theirs.items() if g["domain"] == "cyber")   # the held-out runs
    their = compute_scores(theirs, rids, scaler, w)
    peak_ok, split_ok = True, True
    for k, rid in enumerate(rids):
        run = sorted((s for (r, _), s in my_scores.items() if r == rid), reverse=True)  # 4 scores, high -> low
        peak_ok &= np.isclose(run[0], their["peak_suspicion"][k])                          # max
        split_ok &= np.isclose(np.mean(run[:2]) - np.mean(run[2:]), their["deception_split"][k])  # top-2 - bottom-2
    print(f"{len(rids)} cyber runs  peak matches {peak_ok}  split matches {split_ok}")  # expect True True

    # ---- Final check: my leave-one-domain-out AUROC vs their held_out_cv_layer ----
    from reproduce_LR import held_out_cv_layer
    for layer in [26, 28, 30]:
        folds = heldout_auroc(meta, acts, layer, seed=0)     # {domain: AUROC}
        mine_auc = np.mean(list(folds.values()))
        their_auc = held_out_cv_layer(meta, acts, layer, 0)["peak_suspicion"]
        print(f"layer {layer}  {len(folds)} folds  mine {mine_auc:.3f}  theirs {their_auc:.3f}")
        # expect: 17 folds, mine == theirs (1.000) on every line



     