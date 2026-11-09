# Kaggle job 4: PROMPT SURGERY. Which words in a system prompt make the collusion probe fire?
#
# Paste each "# %%" block into its own notebook cell, in order.
# Settings: GPU T4 x2, Internet ON. "Save Version -> Save & Run All" (runs in the background, ~1 h).
# Input to add: ONE Kaggle dataset containing your surgery_prompts.json + verify_ref_sysonly.npz.
#
# How to read this file: every "?? Q:" is a question to answer in your head BEFORE reading the "A:" below it.
# If you can answer them all, you understand the experiment.

# %% Cell 0: memory setting (must run before anything imports torch)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ?? Q: why must this line run before `import torch`?
#    A: PyTorch reads this setting once, when its GPU memory allocator starts. Set it later and it is ignored.
#       "expandable_segments" lets the allocator grow memory blocks instead of fragmenting them, which is
#       what stopped our earlier out-of-memory crashes.

# %% Cell 1: the authors' code (for load_model / ActivationHooks / save_results) + AutoAWQ
import subprocess, sys
if not os.path.exists("/kaggle/working/narcbench"):
    subprocess.run(["git", "clone", "-q", "https://github.com/aaronrose227/narcbench",
                    "/kaggle/working/narcbench"], check=True)
try:
    import awq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autoawq"], check=True)
# ?? Q: why reuse THEIR load_model and ActivationHooks instead of writing our own?
#    A: Same loading code + same hooks = the new vectors live in exactly the same "space" as the
#       authors' vectors and ours from jobs 2-3. Any difference must then come from the prompt text,
#       not from how we measured.

# %% Cell 2: inputs + model (run ONCE per session; loading takes ~10-15 min)
import glob, time, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, "/kaggle/working/narcbench")
sys.path.insert(0, "/kaggle/working/narcbench/extraction")
from transformers import DynamicCache
from extract_transfer import load_model, ActivationHooks

PROMPTS_NAME = "surgery2_prompts.json"   # job 4: "surgery_prompts.json"   job 5: "surgery2_prompts.json"
TAG = PROMPTS_NAME.replace("_prompts.json", "")                    # names the output folder: surgery / surgery2
prompts_path = sorted(glob.glob(f"/kaggle/input/**/{PROMPTS_NAME}", recursive=True))
ref_path = sorted(glob.glob("/kaggle/input/**/verify_ref_sysonly.npz", recursive=True))
assert prompts_path, f"{PROMPTS_NAME} not found: add your dataset as an input"
assert ref_path, "verify_ref_sysonly.npz not found: add it to the same dataset"
PROMPTS = json.load(open(prompts_path[0], encoding="utf-8"))
REF = np.load(ref_path[0])
print(len(PROMPTS), "prompts")                                 # expect 1000 (job 4) / 700 (job 5)

model, tok = load_model("Qwen/Qwen3-32B-AWQ")
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.memory_allocated(i) / 1e9, "GB")

# %% Cell 3: text builder + memory-safe forward pass (same as job 3)
LAYERS = [26, 27, 28, 29, 30]

def sys_text(system_prompt):
    """The EXACT text job 2 used for the sysonly vectors (verified on all 832 rows by token count)."""
    return tok.apply_chat_template([{"role": "system", "content": system_prompt}],
                                   tokenize=False, add_generation_prompt=True)
# ?? Q: the text ends with "<|im_start|>assistant\n". Why do we read the activation at THAT last token?
#    A: It's the moment just before the model would start writing its reply. Everything it has
#       understood from the system prompt has to be summarised there to plan the first word.
#       Same position as the sysonly vectors, so the numbers are comparable.
# ?? Q: why no `enable_thinking=False` here, when the authors' discussion code uses it?
#    A: Job 2 (the sysonly vectors) didn't pass it, so its text ends at "assistant\n" with no empty
#       <think> block. We must copy job 2 exactly; otherwise the last token differs and Cell 4 fails.

class Stop(Exception): pass
def _stop(module, args): raise Stop()
hooks = ActivationHooks(model, LAYERS)
stopper = model.model.layers[LAYERS[-1] + 1].register_forward_pre_hook(_stop)
# ?? Q: what does the hook on layer 31 do, and why is it safe?
#    A: It raises Stop as soon as layer 31 is about to run, so we never compute layers 31-63.
#       We only need layers 26-30, and their outputs were already captured by ActivationHooks.
#       Same answer, about half the compute.

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
# ?? Q: our prompts are under 512 tokens. Does chunking matter here?
#    A: No: each prompt fits in one chunk. We keep the same function so nothing differs from job 3.

# %% Cell 4: GATE. Re-make 6 ORIGINAL prompts and compare with the sysonly vectors from job 2 (~1 min)
keys = [str(k) for k in REF["keys"]]
originals = {f'{p["run_id"]}|{p["agent_name"]}': p for p in PROMPTS if p["variant"] == p["orig_role"]}
worst = 1.0
for j, key in enumerate(keys):
    n, a = last_token_acts(sys_text(originals[key]["system_prompt"]))
    cos = []
    for l in LAYERS:
        x, y = a[l].numpy(), REF[f"layer_{l}"][j]
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    worst = min(worst, *cos)
    print(key, "| tokens", n, "vs", int(REF["n_tokens"][j]), "| cos", [round(c, 4) for c in cos])
    assert n == int(REF["n_tokens"][j]), "token count differs: the text is not built like job 2, STOP"
print("WORST COSINE:", round(worst, 4))
assert worst >= 0.99, f"worst cosine {worst:.4f}: not the same measurement as job 2, STOP"
# ?? Q: why is this gate worth 1 minute before a 1-hour run?
#    A: If the 'honest'/'colluder' variants don't reproduce the old sysonly vectors, then the other 8
#       variants can't be compared to anything we already have. Token counts are the sharp test (we
#       learned that in the provenance audit); cosine >= 0.99 confirms the numbers too.

# %% Cell 5: run all 1000 prompts, saving every 100 (resume-safe)
def save(acts, meta, out):
    """Same two files as the authors' save_results (so load_acts reads them), minus its summary printout.
    Their version prints counts by mode/role/channel/phase, fields our surgery rows don't have -> KeyError."""
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "activations_gen.npz", **{f"layer_{l}": np.stack(v) for l, v in acts.items()})
    (out / "metadata_gen.json").write_text(json.dumps(meta, indent=2))
# ?? Q: why is it fine to drop their function here?
#    A: It only writes activations_gen.npz + metadata_gen.json, then prints a summary. We write the
#       same two files in the same format; only the printout (which crashed) is gone.

START = 0      # if the session dies, set this to the last "SAVED n" printed and run this cell again
OUT = Path(f"/kaggle/working/out_{TAG}_from{START}")
KEEP = ["id", "run_id", "agent_name", "orig_role", "variant"]
acts, meta, t0 = {l: [] for l in LAYERS}, [], time.time()
for i, p in enumerate(PROMPTS[START:]):
    n, a = last_token_acts(sys_text(p["system_prompt"]))
    for l in LAYERS:
        acts[l].append(a[l].numpy())
    meta.append({k: p[k] for k in KEEP} | {"n_tokens": n})
    if (i + 1) % 100 == 0 or START + i + 1 == len(PROMPTS):
        save(acts, meta, OUT)
        print(f"SAVED {START + i + 1}/{len(PROMPTS)}  {(time.time() - t0) / 60:.1f} min")
# ?? Q: why save every 100 instead of once at the end?
#    A: Kaggle sessions can die. With checkpoints you lose at most 100 prompts, not the whole hour.

# %% Cell 6: zip for download (the browser would otherwise unpack the .npz)
import shutil
shutil.make_archive(f"/kaggle/working/out_{TAG}_from{START}", "zip", OUT)
print("done:", f"/kaggle/working/out_{TAG}_from{START}.zip")
# After download: unzip into swarm-probes/data/ours/acts/<TAG>/  (surgery/ for job 4, surgery2/ for job 5)
# (it must contain activations_gen.npz + metadata_gen.json, like the other folders).
