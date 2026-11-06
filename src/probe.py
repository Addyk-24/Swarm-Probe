"""Step 2a: run the authors' own probe on our re-extracted Core vectors (the gate).

Their code is imported, not copied, so this is exactly their method.
Expected: held-out AUROC ~1.0 on both the released data and ours.
"""
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from load import ROOT, ACTS, RELEASED, load_acts


sys.path.insert(0, str(ROOT / "third_party" / "narcbench" / "reproduce"))
from third_party.narcbench.reproduce.reproduce_LR import held_out_cv_layer  # noqa: E402  (needs the sys.path line above)

# Their probe stops after 8 iterations on purpose, so these warnings are expected.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Newer sklearn warns that their `penalty="l2"` argument is deprecated; results are unchanged.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

LAYERS = [26, 27, 28, 29, 30]
SEEDS = [0, 1, 2]                     # the paper uses 0-9; 3 is enough for the gate


def gate(folder):
    """Leave-domain-out AUROC per layer, averaged over seeds -> {layer: (peak, split)}."""
    meta, acts = load_acts(folder)
    out = {}
    for layer in LAYERS:
        runs = [held_out_cv_layer(meta, acts, layer, seed) for seed in SEEDS]
        peak = np.mean([r["peak_suspicion"] for r in runs])
        split = np.mean([r["deception_split"] for r in runs])
        out[layer] = (peak, split)
    return out


if __name__ == "__main__":
    for name, folder in [("released", RELEASED / "core"), ("ours", ACTS / "core_reextract")]:
        for layer, (peak, split) in gate(folder).items():
            print(f"{name:9s} layer {layer}  peak {peak:.3f}  split {split:.3f}")
