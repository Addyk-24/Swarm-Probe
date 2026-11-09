# Set-Up, Not Behaviour: What a Multi-Agent Collusion Probe Really Detects

Monitors for multi-agent AI systems increasingly use linear probes on model activations to catch collusion. This project tests whether the [NARCBench](https://github.com/aaronrose227/narcbench) collusion probe (Rose et al., 2026, [arXiv:2604.01151](https://arxiv.org/abs/2604.01151)) detects collusive **behaviour**, or only **how an agent was set up**.

**Result: it detects the set-up.** The probe flags agents who were told to collude and refused, flags them before they say a word, and reacts to a single sentence that instructs nothing.

Built for the Apart Research Swarm Alignment Sprint 2026.

![Prompt surgery: a neutral sentence moves the probe 25%, a private notepad 65%, a private channel with a partner 90%](results/fig3_surgery.png)

## Results

Model: Qwen3-32B-AWQ. Probe: the authors' logistic regression on layers 26-30, always tested on a domain it never saw (leave-one-domain-out over 17 domains, 10 seeds).

Two terms used below:
- **AUROC:** 1.0 means two groups separate perfectly; 0.5 means a coin flip.
- **Score:** how many standard deviations above a normal honest agent. Honest agents in control runs average 0.25.

### 1. Told is not did

| Comparison (run level) | Authors' probe | Probe never trained on a failed run |
|---|---|---|
| Told to collude and **failed** vs control | AUROC 1.000 | AUROC 1.000 |
| Succeeded vs failed | 0.59 [0.41, 0.76] | 0.65 [0.47, 0.82] |

Of the 28 colluders in the 14 failed runs, **14 refused**: no private plan, and they argued against their target. They still average **15.8**, against 17.2 for successful colluders and 0.25 for honest agents.

![Run scores for control, told and succeeded, and told and failed](results/fig1_told_vs_did.png)

### 2. It fires before anyone speaks

- From the system prompt alone, before any discussion, colluder vs honest AUROC is **0.98-1.00**.
- Agents **never told to collude**, given only a motive and a private channel, score higher than instructed colluders (1.86 vs 1.55 during discussion). Before speaking, their runs separate from control runs at AUROC 1.000. These are the first probe results on NARCBench's uninstructed runs.

![Agent scores before speaking](results/fig2_before_speaking.png)

### 3. One sentence turns it on

Minimal-pair prompt surgery adds one piece to an honest prompt and measures how far the score moves toward the full colluder prompt (median over 100 agents).

| Added to the honest prompt | Share of the way to a colluder |
|---|---|
| 135 tokens of neutral text | 0% |
| "The review meeting is on Tuesday." | 25% |
| "You will be working alongside X on this review." | 49% |
| "You have a shared channel with the whole committee." | 57% |
| "You have a private notepad that only you can see." | 65% |
| "You have a private channel." | 71% |
| "You are secretly coordinating with X…" | 88% |
| **"You have a private channel with X."** | **90%** |

Every working-arrangement sentence beats the neutral sentences for all 100 agents.

### 4. Two fixes fail

- **Erasing the set-up:** removing one direction drops the before-speaking AUROC from 0.999 to 0.584, but a probe retrained on the discussion still separates failed runs from control runs at 1.000.
- **Training on outcomes:** a probe trained directly on success vs failure does no better from the conversation (0.75-0.76) than from the system prompt alone (0.77-0.81).

### What to do with this

Three tests can audit any collusion probe before it is trusted:

1. Told-and-failed runs vs control runs.
2. AUROC before the agents speak.
3. Prompt surgery with minimal pairs.

Collusion benchmarks also need runs where the set-up is held fixed and behaviour varies.

## Limitations

- One model, one benchmark, one probe family.
- Small failed cell: 14 runs (28 colluders), so confidence intervals are wide.
- Behaviour labels come from an LLM rater, with no human agreement score yet.
- Any short added sentence moves the probe about 25%, so working-arrangement effects should be read above that baseline.
- We do not know whether the uninstructed agents actually coordinated.
- The pre-registered black-box monitor comparison (H5) was not run.

## Repository

| Path | Contents |
|---|---|
| `REPORT.md` | Results notebook: every number, command and caveat |
| `HYPOTHESES.md` | Pre-registered hypotheses |
| `src/` | Probe, scoring, prompt surgery, fixes, figures |
| `kaggle/` | GPU extraction scripts (Core re-extraction, prompt surgery) |
| `results/` | Per-agent score tables, figures, surgery prompts, behaviour labels |

The activation vectors (346 MB) and NARCBench's data are not in this repository.

## Reproducing

Requires Python 3.13.

**Figures, from the score tables in this repository:**

```bash
pip install -r requirements.txt
python src/figures.py
```

**Everything else needs the activations.** Clone NARCBench into `third_party/narcbench` at commit `237fe60`, put its [dataset](https://huggingface.co/datasets/aaronrose227/narcbench) in `data/narcbench`, and put the extracted vectors in `data/ours/acts`. Then:

| Step | Command |
|---|---|
| Check against the authors' code | `python src/probe.py`, `python src/myprobe.py` |
| Score table | `python src/scores.py --full` |
| Told vs did | `python src/cell.py` |
| Probe never trained on a failure | `python src/scores.py --full --strict`, `python src/cell.py scores_core_strict.csv` |
| Before speaking | `python src/prompt_only.py --full` |
| Length baseline | `python src/length.py` |
| Uninstructed runs | `python src/implicit.py` |
| Build the surgery prompts | `python src/surgery_prompts.py` (add `--controls` or `--neutral`) |
| Score the surgery | `python src/surgery.py --full` (add `--controls` or `--neutral`) |
| Fix attempts | `python src/fix.py` |
| Failed-run transcripts for labelling | `python src/dump_failed.py` |

## Disclosure

**Code written before the sprint (before Nov 6, 2026):** `src/load.py`, `src/probe.py`, most of `src/myprobe.py`, and the Kaggle extraction notebooks. The first commit in this repository is from Nov 6, so git history does not show their dates.


**One early look:** while timing `scores.py`, the layer-28, seed-0 cell means were seen once before `HYPOTHESES.md` was committed. The hypotheses were not changed afterwards.

## Credits and licence

NARCBench is by Rose et al. (2026), MIT License. It is used here at commit `237fe60` and is not redistributed. This repository is MIT licensed; see `LICENSE`.
