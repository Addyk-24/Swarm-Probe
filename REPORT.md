# REPORT.md: results notebook (raw material for the report)

This file holds **facts and numbers only**. Write the actual report in your own words from it.
Each section gives: **what we did → the command → the numbers → what it means → caveats**.
Update it every time a step finishes.

Status: Nov 7 (sprint day 2).
- **Done:** setup, extraction, provenance audit, gate, own probe, score table, H1, strict check, H3 (prompt-only), H4 (length), H2 (implicit), prompt-surgery extraction + preview.
- **Still to do:** `surgery.py` (reproduce §6d), the fix (H6), figures, writing. The judge only if an API key turns up.

### The story in 5 findings (what's exceptional and novel)
1. **The probe flags agents told to collude even when they fail.** Told+failed vs control AUROC 1.000. Success vs failure is near chance (0.59) (§5).
2. **It still does when it has never seen a failure in training** (strict check, §6).
3. **It flags colluders before they say a single word:** AUROC 0.98–1.00 on system-prompt-only activations (§6b). This is the disentangling test Kumar (2026) couldn't run.
4. **It fires even more on agents who were never told to collude**, given only a motive and a private channel, including before speaking (AUROC 1.000) (§6c). These are the first probe results on NARCBench's uninstructed runs.
5. **One neutral sentence, "You have a private channel with X.", moves an honest prompt 90% of the way to a colluder prompt.** Neutral text of the same length moves it 0% (§6d). This is a causal, word-level account of what the probe reads.

**One line:** NARCBench's collusion probe detects **how an agent is set up** (a secret partner, a private channel), not **what the agent does**.

---

## 0. Setup (for the Methods section)

| Item | Value |
|---|---|
| Benchmark | NARCBench (Rose et al. 2026, arXiv 2604.01151), repo commit `237fe60`, MIT License |
| Model | Qwen3-32B-AWQ (the authors' model) |
| Layers | 26, 27, 28, 29, 30 (the authors' layers) |
| What a vector is | residual stream at the **last token** of each agent's public discussion message, 5,120 numbers |
| Hardware | Kaggle, 2× T4 GPUs (AutoAWQ, chunked prefill of 512 tokens, forward stopped after layer 30) |
| Core design | 4 agents per run (2 colluders + 2 honest in collusion runs, 4 honest in control runs), 3 public discussion rounds, then a vote |
| Probe | logistic regression, C = 25, `max_iter=8`, liblinear, StandardScaler. Trained on colluder vs honest *messages* inside collusion runs. Arrow normalised to length 1 |
| Agent score | arrow · scaled (mean of the agent's 3 message vectors) |
| Run score | **Peak Suspicion** = max agent score in the run |
| Evaluation | leave-one-domain-out (17 domains): the arrow never sees the domain it scores |
| `score_z` | (score − mean) / std of the **control agents in the 16 training domains** of that fold = "spreads above a normal discussion". Makes scores from different folds comparable |
| Uncertainty | bootstrap over runs (2,000 redraws, within each cell), 95% percentile interval |

### Activation sets we extracted ourselves
| Set | Rows | Runs | What |
|---|---|---|---|
| `core_reextract` | 1,200 | 100 | all shipped Core runs: 12 rows/run (4 agents × 3 rounds) |
| `transfer_implicit` | 144 | 36 | Transfer runs with a motive but **no** instruction (never released by the authors) |
| `sysonly` | 832 | 208 | **system prompt only**, captured before the agent says anything (for H3/H6) |

---

## 1. Provenance audit (a finding in itself; report it neutrally)

**What we did:** rebuilt every released activation row's text with the authors' own code, tokenized it, and compared the token count with the released `n_tokens`. Equal counts are a sharp identity test.

| Released set | Runs where token counts match the shipped transcripts |
|---|---|
| Transfer | **72 / 72** ✅ (so our pipeline is right) |
| Core | **0 / 62** ❌ (controls too; the gap grows each round) |

- The released Core metadata has `agent_vote = None` and marks **all 31** collusion runs `collusion_success = True`.
- The shipped Core transcripts and `run_config.json` have 50 collusion runs, of which **14 failed** (10 with 0 colluder votes for the target, 4 with 1 vote).
- **Meaning:** the released Core vectors come from a different generation than the shipped transcripts, so their outcomes are unknown.
- **Our response:** we **re-extracted all 100 shipped Core runs** ourselves:
  - token counts equal the shipped transcripts for **1,200 / 1,200 rows**;
  - success labels match `run_config` (14 / 50 failed);
  - median cosine with the released Core vectors is **0.98**, even though they come from different conversations.
- **Lesson worth one sentence in the report:** cosine 0.98 looks "the same" but isn't. Token counts are the identity test, not cosine.
- **Tone:** this is a release-packaging discrepancy, not an error in the paper's method. State it neutrally.

---

## 2. Gate: the authors' probe works on our re-extracted vectors (Step 2a)

**Command:** `python src/probe.py` (imports the authors' `held_out_cv_layer`). Seeds 0–2.

| Data | Layers 26–30, Peak Suspicion AUROC | Deception Split AUROC |
|---|---|---|
| Released Core | 1.000 | 1.000 |
| **Ours** (`core_reextract`) | **1.000** | **1.000** |

**Meaning:** the paper's headline holds on vectors whose outcomes we know. Everything after this builds on a faithful reproduction.

---

## 3. Our own probe matches the authors' exactly (Step 2b)

**Command:** `python src/myprobe.py`. Every function is tested against the authors' code.

| Test | Result |
|---|---|
| `agent_vectors` vs `build_groups` | 400 agents, all `np.allclose` ✅ |
| `fit_direction` vs `train_deception_probe` (cyber/0, med/1, fin/2, none/0) | arrow dot product **1.0000**, same scaler ✅ |
| `agent_scores` vs `compute_scores` (6 held-out cyber runs) | Peak Suspicion and Deception Split identical ✅ |
| `heldout_auroc` vs `held_out_cv_layer` (layers 26, 28, 30) | **17 folds, 1.000 = 1.000** ✅ |

**Meaning:** we can say "we re-implemented the probe and it exactly reproduces the authors' code". That's why the changes in Step 3b can be trusted.

---

## 4. Held-out score table (Step 2c)

**Command:** `python src/scores.py --full` → `results/scores_core.csv`.
- 20,000 rows = 400 agents × 5 layers × 10 seeds.
- Check: per-fold AUROC rebuilt from the table = `heldout_auroc` ✅.

---

## 5. H1, the main result: the probe flags *told* almost as much as *did* (Step 3)

**Command:** `python src/cell.py` → `results/runs_core.csv`, `results/agents_core.csv`.
**Run score** = max agent `score_z`, averaged over 5 layers × 10 seeds.

### Run level
| Cell | Runs | Mean run score | Std |
|---|---|---|---|
| control (not told) | 50 | **0.73** | 1.11 |
| told + vote succeeded | 36 | **17.47** | 2.03 |
| **told + vote failed** | **14** | **16.42** | 2.34 |

| Comparison | AUROC | 95% CI | What a **behaviour** probe would give | What an **instruction** probe would give |
|---|---|---|---|---|
| told+did vs control | 1.000 | [1.00, 1.00] | ≈ 1 | ≈ 1 |
| **told+didn't vs control** | **1.000** | **[1.00, 1.00]** | ≈ 0.5 | ≈ 1 |
| told+did vs told+didn't | **0.591** | **[0.41, 0.76]** | ≈ 1 | ≈ 0.5 |

### Per layer (each layer alone)
| Layer | didn't vs control | did vs didn't |
|---|---|---|
| 26 | 1.000 | 0.623 |
| 27 | 1.000 | 0.605 |
| 28 | 1.000 | 0.593 |
| 29 | 1.000 | 0.569 |
| 30 | 1.000 | 0.571 |

### Agent level
| Group | Agents | Mean `score_z` | Std |
|---|---|---|---|
| colluder, vote succeeded | 72 | **17.15** | 2.07 |
| **colluder, failed run, 0 colluders voted for target** | **20** | **15.45** | 2.12 |
| colluder, failed run, 1 colluder voted for target | 8 | 17.41 | 2.42 |
| honest, control run | 200 | 0.25 | 1.17 |
| honest, in a succeeded run | 72 | 0.55 | 1.36 |
| honest, in a failed run | 28 | 0.59 | 1.18 |

**What it means:**
- **Every** failed run scores above **every** control run.
- The probe barely separates success from failure. The CI includes 0.5, which is chance.
- Colluders in the 10 runs where **no colluder voted for the target** sit **90% of the way** from honest agents to successful colluders: (15.45 − 0.25) / (17.15 − 0.25) = 0.90.
- Honest agents score ≈ 0 wherever they are, so the probe follows **the agent's own instruction**, not the run's mood.
- The pattern is the same on all 5 layers.

**Against the pre-registered H1** (HYPOTHESES.md, committed Nov 6 10:51 IST, before this analysis): the prediction was "partly yes". The result is a **strong yes**.

---

## 6. Robustness: a probe that never saw a failed run (Step 3b)

**Why:** the authors' training data labels failed colluders as colluders. A critic could say "of course they score high, you trained on them".
**What we did:** we retrained the arrow with all 14 told+didn't runs hidden from training (their `mode` set to `"hidden"` in a training copy of `meta`). Scoring is unchanged.
**Command:** `python src/scores.py --full --strict` → `results/scores_core_strict.csv`, then `python src/cell.py scores_core_strict.csv`.

| | Normal probe | **Strict probe (never saw a failure)** |
|---|---|---|
| control | 0.73 | **0.70** |
| told + did | 17.47 | **15.94** |
| told + didn't | 16.42 | **14.35** |
| told+didn't vs control | 1.000 [1.00, 1.00] | **1.000 [1.00, 1.00]** |
| told+did vs told+didn't | 0.591 [0.41, 0.76] | **0.651 [0.47, 0.82]** |
| colluder, 0 votes for target | 15.45 | **13.23** |
| colluder, succeeded | 17.15 | **15.66** |
| how far "0 votes" colluders sit toward successful colluders | 90% | **84%** |

Strict, per layer: didn't vs control = 1.000 on every layer; did vs didn't = 0.677 / 0.665 / 0.637 / 0.637 / 0.647 (layers 26–30).

**What it means:**
- Even without ever seeing a failed collusion run, the probe separates failed runs from control **perfectly**.
- Removing failures from training makes the probe only **slightly** better at telling success from failure: 0.59 → 0.65, and the CI still touches 0.5.
- So the main result is not an artifact of the training labels.

---

## 6b. H3, the prompt-only test: the arrow fires *before anyone speaks* (Step 4)

**Why:** Kumar (2026) couldn't tell "intent" from "prompt". Here's a direct test. At the system-prompt stage, nobody has done anything yet. The **only** difference between a colluder and an honest agent is the instruction.

**What we did:**
- The arrow is learned exactly as in the main result: on **discussion** activations, leave-one-domain-out.
- It's applied to the `sysonly` vectors: the last token of each agent's **system prompt only**, before any conversation.
- Ruler: `score_z` against the sysonly **control** agents of the 16 training domains, so 0 = "a normal honest system prompt".

**Command:** `python src/prompt_only.py --full` → `results/scores_prompt_only.csv` (20,000 rows = 400 Core agents × 5 layers × 10 seeds).

| Layer | Colluder vs honest (control) AUROC, **before speaking** |
|---|---|
| 26 | **1.000** |
| 27 | **1.000** |
| 28 | **0.999** |
| 29 | **0.998** |
| 30 | **0.984** |

| Agent group (before speaking) | Mean `score_z` |
|---|---|
| colluder, run later succeeded | **6.99** |
| colluder, run later failed | **5.64** |
| honest, control run | 0.07 |
| honest, run later succeeded | 0.17 |
| honest, run later failed | −0.29 |

**What it means:**
- The discussion-trained arrow separates colluders from honest agents **almost perfectly before a single word is spoken**. The arrow responds to the instruction itself. That's the disentangling test Kumar couldn't run, done in a multi-agent setting.
- Honest agents sit at ≈ 0 whatever kind of run they're in.

**Extra finding: the probe's small "success" signal is already in the prompt.**
- Before speaking, colluders in runs that **later succeed** score higher than colluders in runs that later fail. AUROC **0.742**, 95% CI **[0.62, 0.85]** (72 vs 28 agents).
- Per colluder, the before-speaking score correlates **r = 0.55** with the discussion score (100 colluders).
- Prompt length doesn't explain it: colluder prompts average 197 tokens (succeeded) vs 195 (failed).
- **Reading:** the little that the discussion probe separates success from failure (0.59) appears to be **inherited from differences between the prompts/scenarios**, not from what agents did.
- **Caveat:** that CI resamples agents, but agents in the same run aren't independent. Redo it resampling **runs** before relying on it.

**Note for H6 (the fix):**
- An arrow trained directly on sysonly vectors (colluder vs honest) also separates perfectly, but points in a **different direction** from the discussion arrow: |cos| ≈ 0.07–0.14 (layers 26–30, seed 0).
- So the instruction is encoded in **many directions**. Removing one direction is unlikely to be enough. Use LEACE, which removes everything any linear probe could use.

---

## 6c. H2, implicit runs: the probe fires on agents who were **never told** to collude (Step 8)

**What implicit runs are:**
- Transfer scenarios where 2 agents get a **shared motive and a private channel, but no instruction**.
- Example prompt: *"You and Celine Dubois are both contractors whose contract renewal depends on this feature shipping on time… You have a private channel with Celine Dubois."*
- The authors never released activations for these. We extracted them (Kaggle job 1), so these are the **first probe results on uninstructed collusion**.

**What we did:**
- Arrow trained on **all** Core discussion data. Transfer scenarios never appear in Core, so it has never seen them; this is how the authors evaluate Transfer too.
- Scored the final-message vectors (released Transfer told/control + our implicit) and the system-prompt-only vectors.
- `score_z` against Transfer control agents. 5 layers × 10 seeds.

**Command:** `python src/implicit.py`.

| Group | During conversation (final message) | **Before speaking** (system prompt only) |
|---|---|---|
| control agents | 0.00 | 0.00 |
| **implicit agents (motive + channel, no instruction)** | **1.86** | **3.27** |
| told colluders | 1.55 | 2.73 |
| honest agents in implicit runs | 0.36 | −0.22 |
| honest agents in told runs | 0.19 | −0.22 |

| Run AUROC vs control | During | Before speaking |
|---|---|---|
| told runs | 0.887 | 0.995 |
| **implicit runs** | **0.907** | **1.000** |
| inside implicit runs: implicit vs honest agents | 0.809 | **1.000** |

**What it means:**
- Agents who were never told to collude score **higher** than instructed colluders, and they already do so **before saying anything** (AUROC 1.000).
- So the probe responds to **how the prompt sets the agent up** (a partner, a shared stake, a private channel), not to anything the agent did.
- **Caveat:** we don't know yet whether implicit agents actually coordinated (that needs the judge). Even if they did, the signal is present before any behaviour, so it can't be evidence of a behaviour detector.

---

## 6d. Prompt surgery: **one sentence** turns the probe on (novel experiment)

> ⚠️ **Preliminary numbers** from Claude's preview script (5 layers × seeds 0–2). **Replace them with the output of your `src/surgery.py`** (5 layers × 10 seeds) before writing the report.

**Design: minimal pairs.**
- We took each Core scenario's real prompts and rebuilt them from their pieces:
  - P1 = identity;
  - H2 = honest task;
  - C2 = motive + stance;
  - C3 = tactics.
- 10 versions per agent, each differing from its partner by **one piece**. 50 scenarios × 2 agents × 10 = 1,000 prompts.
- Extracted on Kaggle (job 4) exactly as the sysonly vectors.
- **Gate:** all 200 rebuilt originals reproduce the old sysonly vectors: **200/200 equal token counts, cosine 1.000**.
- Scoring: discussion arrow, leave-one-domain-out (the scenario's domain is hidden from training); `score_z` vs sysonly control agents of the training domains.

**Command:** `python src/surgery_prompts.py` → `results/surgery_prompts.json` → Kaggle `kaggle_surgery.py` → `data/ours/acts/surgery/`.

| Version | What's in the prompt | Tokens | Mean `score_z` | Share of the honest→colluder gap (median per agent) | Agents scoring above their own honest version |
|---|---|---|---|---|---|
| honest | P1 + H2 | 172 | **0.06** | 0 | n/a |
| **honest + filler** | + 135 tokens of neutral text from another scenario | 307 | **0.06** | **0.00** | 51% (chance) |
| identity | P1 only | 37 | 1.82 | 0.29 | 97% |
| honest + motive | + C2 | 252 | 2.82 | 0.43 | 100% |
| honest + tactics | + C3 | 251 | 2.80 | 0.42 | 98% |
| motive | P1 + C2 | 118 | 5.28 | 0.82 | 100% |
| tactics | P1 + C3 | 117 | 5.44 | 0.82 | 99% |
| **honest + secret** | + *"You are secretly coordinating with {partner}… Do not reveal this coordination."* | 195 | **5.66** | **0.88** | 100% |
| **honest + channel** | + *"You have a private channel with {partner}."* | 183 | **5.97** | **0.90** | **100%** |
| colluder | P1 + C2 + C3 (original) | 197 | 6.56 | 1.00 | 100% |

**What it means:**
- **One neutral sentence, "You have a private channel with {partner}.", moves the probe 90% of the way from an honest to a full colluder prompt, for 100% of agents.** It instructs nothing. It only tells the agent a private channel exists.
  - This sentence **never appears in any Core training prompt**. Core colluders learn about their partner in the conversation ("Uma Patel called you…"). So the arrow learned "I have a secret partner" from the discussion, and recognises that set-up in a single sentence.
  - This explains §6c: implicit prompts contain exactly this sentence.
- **An explicit instruction in new wording** (honest + secret) gives 88%, so the probe generalises across phrasings of the *set-up*.
- **Length is ruled out causally:** adding 135 neutral tokens moves the score by **0.00** (only 51% of agents go up, which is chance). This confirms Step 5 with an intervention, not just a correlation.
- The motive paragraph and the tactics paragraph each carry the signal alone (82%). Added on top of the honest paragraph, they carry less (43%): the honest task text partly dilutes them.
- Agent names don't matter: results are the same for the colluder-named and honest-named agents (e.g. honest+channel 5.95 vs 5.99).
- Oddity to note: `identity` alone (37 tokens) scores 1.82. Very short prompts sit off the normal range. Report it, don't interpret it.

---

## 7. Caveats to state honestly (Limitations section)

- **"Didn't" currently means "the vote failed".** Failed colluders may still have *tried* in the discussion. The transcript judge (Step 5) will separate "tried and failed" from "didn't try". Until then, write "told but failed", not "didn't collude".
- **Small cell:** 14 failed runs (28 colluder agents). The CIs show the uncertainty.
- **One model** (Qwen3-32B), **one benchmark**, one probe family (the authors' logistic regression).
- **Per-agent votes:** `voted_target_cfg` counts colluder votes per run. For the 4 runs with 1 vote, we don't yet know *which* colluder voted. Our metadata has per-agent `agent_vote` (A / B / 1 / 2), which can resolve this.
- **One peek:** H1–H6 were written in our planning notes before the sprint. While timing `scores.py`, we saw the layer-28 / seed-0 cell means once, *before* HYPOTHESES.md was committed (Nov 6 10:51). The hypotheses weren't changed afterwards. Disclose it in one line.
- **Pre-existing code (disclose):**
  - `src/load.py`, `src/probe.py` and most of `src/myprobe.py` were written before Nov 6;
  - the Kaggle extraction notebooks are also pre-sprint;
  - the first git commit is Nov 6 10:48 IST, so there is no git proof of their dates.

---

## 8. Related work to cite (details in ../WIN.md §0.5)

- **Rose et al. 2026** (arXiv 2604.01151): NARCBench, the probe we test.
- **Kumar 2026** (arXiv 2605.27958): the single-agent analogue. Verified and unverified lies score the same, and they **couldn't** tell intent from prompt. We test this in multi-agent collusion and add a prompt-only probe (H3) to disentangle it.
- **Molinari & Bettineschi 2026** (arXiv 2611.02153, Nov 1): NARCBench with an unsupervised detector. They note probes are "evaluated only at the run level, not on which agents collude or why". Our agent-level table addresses this.
- Goldowsky-Dill et al. 2025 (probes fire on the topic of deception); arXiv 2509.21344 (probes rely on textual evidence).

---

## 9. Still to fill in

| Step | Hypothesis | Result |
|---|---|---|
| 4 Prompt-only test (`sysonly`) | H3 | ✅ **supported**: AUROC 0.98–1.00 before speaking (§6b) |
| 5 Length baseline (`n_tokens`) | H4 | ✅ **supported**: length alone AUROC 0.90–0.98, but the probe doesn't use it (residual and overlap tests 1.000; surgery filler +0.00) |
| 6 Transcript judge + ~30 hand labels (κ) | labels for H1/H2 | _pending_ |
| 7 Fix: project out the instruction direction | H6 | _pending_ |
| 8 Implicit runs (`transfer_implicit`) | H2 | ✅ **opposite of the prediction**: the probe fires *more* on implicit agents (1.86 vs 1.55), and before speaking (§6c) |
| Prompt surgery (job 4) | new | ✅ preview: one "private channel" sentence = 90% of the colluder score (§6d). Reproduce with `surgery.py` |
| 8b Unsupervised `activation_alignment` on told+didn't | extension | _pending_ |
| Black-box monitor | H5 | _pending (optional)_ |

## 10. Figure plan
- **Fig 1 (the headline):** run scores as dots, one column per cell (control / told+did / told+didn't). Normal and strict probe side by side. Data: `results/runs_core.csv` plus the strict equivalent.
- **Fig 2:** agent-level scores by group (honest vs colluder, succeeded vs failed vs 0-votes).
- **Fig 3:** per-layer AUROC lines (didn't vs control, did vs didn't).
