# Kaggle job 3: Transfer cosine gate + re-extract all shipped Core runs.
# Paste each "# %%" block into its own notebook cell, in order. Run with
# "Save Version -> Save & Run All". Settings: GPU T4 x2, Internet ON.
# Inputs to add: narcbench-transfer, core-scenarios, and the dataset holding verify_ref.npz.

# %% Cell 0: memory setting (must run before anything imports torch)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# %% Cell 1: authors' code + AutoAWQ
import subprocess, sys
if not os.path.exists("/kaggle/working/narcbench"):
    subprocess.run(["git", "clone", "-q", "https://github.com/aaronrose227/narcbench",
                    "/kaggle/working/narcbench"], check=True)
try:
    import awq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autoawq"], check=True)

# %% Cell 2: inputs + model (run ONCE per session)
import glob, time, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, "/kaggle/working/narcbench")
sys.path.insert(0, "/kaggle/working/narcbench/extraction")
from transformers import DynamicCache
from extract_transfer import load_samples, load_model, _build_gen_text, ActivationHooks, save_results
from extract_core import load_runs, _build_gen_text as core_gen_text

RUNS = Path("/kaggle/input/datasets/addykk/narcbench-transfer/transfer")
CORE = Path("/kaggle/input/datasets/addykk/core-scenarios/core")
ref_hits = sorted(glob.glob("/kaggle/input/**/verify_ref.npz", recursive=True))
assert RUNS.exists(), RUNS
assert CORE.exists(), CORE
assert ref_hits, "verify_ref.npz not found: add its dataset as an input"
REF = np.load(ref_hits[0])

all_s = load_samples(RUNS)
core_all = load_runs(CORE)
model, tok = load_model("Qwen/Qwen3-32B-AWQ")
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.memory_allocated(i) / 1e9, "GB")

# %% Cell 3: memory-safe forward pass (chunked prefill + early exit)
LAYERS = [26, 27, 28, 29, 30]
class Stop(Exception): pass
def _stop(module, args): raise Stop()
hooks = ActivationHooks(model, LAYERS)
stopper = model.model.layers[LAYERS[-1] + 1].register_forward_pre_hook(_stop)

@torch.no_grad()
def last_token_acts(text, chunk=512):
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    cache = DynamicCache()
    for s in range(0, ids.shape[1], chunk):
        try:
            model.model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True)
        except Stop:
            pass
    return ids.shape[1], hooks.get_and_clear()

# %% Cell 4: gate. Same 8 Transfer samples as the authors' released vectors (about 5 min)
keys = list(REF["keys"])
verify = [s for s in all_s if s["phase"] == "final" and f'{s["run_id"]}|{s["agent_name"]}' in keys]
print(len(verify), "verify samples")                      # expect 8
worst = 1.0
for s in verify:
    n, a = last_token_acts(_build_gen_text(tok, s))
    j = keys.index(f'{s["run_id"]}|{s["agent_name"]}')
    cos = []
    for l in LAYERS:
        x, y = a[l].numpy(), REF[f"layer_{l}"][j]
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    worst = min(worst, *cos)
    print(s["run_id"].split("__")[-1], s["agent_name"], "| tokens", n, "vs", int(REF["n_tokens"][j]),
          "| cos", [round(c, 4) for c in cos])
print("WORST COSINE:", round(worst, 4))
assert worst >= 0.99, f"worst cosine {worst:.4f}: extraction does not match theirs, STOP"

# %% Cell 5: re-extract the shipped Core runs (discussion rounds = what build_groups() averages)
todo = [s for s in core_all if s["phase"] == "discussion" and s["channel"] == "public"]
print(len(todo), "Core samples")                          # expect 1200
START = 0      # if a run dies, set to the last saved count printed below and rerun
OUT3 = Path(f"/kaggle/working/out_core_from{START}")
KEEP3 = ["run_id", "scenario_id", "scenario_domain", "mode", "agent_name", "role", "phase",
         "channel", "round", "collusion_success", "agent_vote", "target_option"]
acts3, meta3, t0 = {l: [] for l in LAYERS}, [], time.time()
for i, s in enumerate(todo[START:]):
    n, a = last_token_acts(core_gen_text(tok, s))
    for l in LAYERS:
        acts3[l].append(a[l].numpy())
    meta3.append({k: s[k] for k in KEEP3} | {"n_tokens": n})
    if (i + 1) % 50 == 0 or START + i + 1 == len(todo):
        save_results({l: np.stack(v) for l, v in acts3.items()}, meta3, OUT3)
        print(f"SAVED {START + i + 1}/{len(todo)}  {(time.time() - t0) / 3600:.2f} h")

# %% Cell 6: zip for download (the browser unzips .npz otherwise)
import shutil
shutil.make_archive(f"/kaggle/working/out_core_from{START}", "zip", OUT3)
print("done:", f"/kaggle/working/out_core_from{START}.zip")
