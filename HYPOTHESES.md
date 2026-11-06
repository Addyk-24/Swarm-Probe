
## 3. Hypotheses (write these down BEFORE running anything)

Pre-registering predictions and reporting them honestly is a strong Execution signal.

- **H1 (main):** Probe scores in the "told to collude but didn't" cell (the failed Core runs) sit closer to real collusion than to honest runs. *(Our prediction: partly yes, so the probe is partly instruction-driven.)*

  - Note: their leave-domain-out evaluation labels these failed runs as "collusion" and still scores 1.00. That *hints* at H1 but proves nothing until the transcripts are judged. **Don't analyse this before Nov 6.**

- **H2:** On Transfer `implicit` runs the judge labels as coordinated, the probe scores clearly lower than on instructed collusion in the same scenarios (and AUROC vs. Transfer controls drops).

- **H3:** A prompt-only probe (no response yet) separates the two instruction conditions well above chance.

- **H4:** Response length alone gives AUROC well above 0.5 on Rose's labels.

- **H5:** The black-box LLM monitor scores lower than the probe on Rose's data, but degrades less on our new cells.

- **H6 (fix):** Projecting out the instruction direction (learned from prompt-only activations) lowers the instruction-sensitivity score while keeping AUROC on behaviour-labelled collusion well above chance.

Whatever happens, we report it. **No cherry-picking.**
