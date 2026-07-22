# Advanced training methods — additional references

Companion to `DNN_PLAN.md` and `LEARNING_GUIDE.md`. Those already cover the
essentials (standardisation, dropout, OneCycle, AMP, `torch.compile`, diverse
ensembling) and the modern tabular libraries (TabM, RealMLP, TabPFN). This file
lists methods that are **not** in those guides, with a note on why each one fits
*this* model (deep BatchNorm+GELU MLP, 21 low-level features, 10M rows, large
batches, AdamW + OneCycle) and roughly what to expect.

## Honest expectations first

The measured practical ceiling for this dataset is ~**0.891 single / ~0.893
ensemble** on the low-level features (see `DNN_PLAN.md` §11). The current
in-progress run's best single model is already **0.8907** (trial 27). So we are
*already at the single-model ceiling*. Read the rest of this file with that in
mind:

- Most single-knob tweaks below buy **≤ +0.002 AUC**, often less, and some buy
  nothing on a dataset this large (10M rows is itself a strong regulariser).
- The methods with genuine upside here are the ones that either (a) let you
  train **reliably deeper** nets (residual connections), (b) change the
  **input representation** (numerical embeddings — the only technique shown to
  sometimes give real jumps on GBDT-friendly tabular data), or (c) add
  **near-free generalisation / diversity** for the ensemble (SWA, EMA, snapshot
  ensembling).
- Verify any gain on the held-out **test** set, not just val — at this AUC the
  val/test noise (±0.0005 on 500k rows) is comparable to the effect sizes.

---

## A. Architecture — enabling depth at medium width

This is the most relevant section given the plan to push more layers at medium
width (384–768).

### A1. Residual / skip-connection MLPs (the "ResNet" tabular baseline)
A plain BatchNorm MLP gets progressively harder to optimise past ~8–10 layers;
signal and gradients attenuate. Wrapping each block as `x + f(x)` (pre-norm:
`Linear→BN→act→Dropout→Linear`, added back to the input, with a projection when
widths differ) removes that ceiling and is *the* prerequisite for the deeper
nets you want to search. Gorishniy et al. found a ResNet-style MLP is a strong,
frequently-omitted tabular baseline.

- **Fit here:** directly enables the "more layers at medium width" plan. Make
  residual the default for `depth > 8`, or add it as a boolean search knob.
- **Expected:** lets depth 10–14 train without collapse; on its own usually
  +0.000–0.002, but it *unlocks* the depth region rather than helping at d7.
- Gorishniy, Rubachev, Khrulkov, Babenko (2021), *Revisiting Deep Learning
  Models for Tabular Data* — arXiv:2106.11959 (NeurIPS 2021). Reference impl:
  `rtdl` / `rtdl_revisiting_models` on PyPI.

### A2. Embeddings for numerical features (PLR / periodic)
Instead of feeding each scalar feature straight into the first `Linear`, map it
through a small learned embedding first — piecewise-linear (PLE) or periodic
(sin/cos of learned frequencies) + linear + ReLU ("PLR"). This is the single
technique in the literature shown to let plain MLPs *match attention models and
sometimes close the gap to GBDTs* on tabular data. Physics kinematics (masses,
pT, angles) are continuous and often multi-modal, exactly where these embeddings
help.

- **Fit here:** the biggest *architectural* lever with real (not fractional)
  upside; also the largest code change. Worth a dedicated experiment rather than
  a search knob at first.
- **Expected:** the one method that could push a single model past ~0.891 if
  anything can; no guarantee on this dataset, but highest ceiling of the list.
- Gorishniy, Rubachev, Babenko (2022), *On Embeddings for Numerical Features in
  Tabular Deep Learning* — arXiv:2203.05556 (NeurIPS 2022). Impl in the same
  `rtdl` ecosystem (`rtdl_num_embeddings`).

### A3. Self-normalising networks (SELU + AlphaDropout) — a BatchNorm-free path
An alternative deep-FC recipe: SELU activation with LeCun-normal init and
AlphaDropout keeps activations self-normalised without BatchNorm, which removes
BatchNorm's large-batch statistics issues entirely. Mainly interesting as a
*diverse* ensemble member (decorrelated from your BN+GELU nets), not as a
replacement.

- **Fit here:** cheap source of ensemble diversity; unlikely to beat BN+GELU
  head-to-head.
- Klambauer, Unterthiner, Mayr, Hochreiter (2017), *Self-Normalizing Neural
  Networks* — arXiv:1706.02515 (NeurIPS 2017).

---

## B. Normalisation & regularisation

### B1. Ghost Batch Normalization (directly relevant to your batch sizes)
Your best trials use batch **8192**, and 16k/32k did worse (see trends). Large
batches give BatchNorm very "easy" statistics and are associated with a
generalisation gap. Ghost BN computes BN statistics over fixed **virtual
sub-batches** (e.g. 256–2048) inside the big batch, restoring the regularising
noise of small-batch BN while keeping large-batch throughput. Used by TabNet.

- **Fit here:** high relevance — it targets exactly the large-batch regime you
  train in, and may explain why 8192 > 32768. A cheap knob to add.
- **Expected:** can recover some of the AUC lost at larger batches; lets you use
  16k/32k for speed without the penalty. +0.000–0.002.
- Hoffer, Hubara, Soudry (2017), *Train longer, generalize better…* —
  arXiv:1705.08741. TabNet (uses ghost BN): Arik & Pfister — arXiv:1908.07442.

### B2. Label smoothing
Already wired as a knob (`label_smoothing`, currently 0.0). Softening BCE targets
to e.g. 0.02–0.05 can regularise and improve probability calibration; at 10M rows
the effect is usually small but occasionally helps a slightly-overfit wide net.

- **Fit here:** trivial to add to the search (0.0–0.05). Low cost, low-to-modest
  payoff.
- Müller, Kornblith, Hinton (2019), *When Does Label Smoothing Help?* —
  arXiv:1906.02629 (NeurIPS 2019).

### B3. mixup / manifold mixup
Train on convex combinations of input pairs (and their labels). A strong,
architecture-agnostic regulariser; manifold mixup interpolates hidden
activations instead. Cheap to try, effect on large tabular sets is hit-or-miss.

- **Fit here:** optional; more diversity for the ensemble than raw single-model
  gain.
- Zhang, Cissé, Dauphin, Lopez-Paz (2018), *mixup* — arXiv:1710.09412 (ICLR).

---

## C. Weight averaging & cheap ensembling

These give most of the "free lunch" left on the table and compose with your
existing probability-averaging ensemble.

### C1. Stochastic Weight Averaging (SWA)
Average the weights over the tail of training (constant/cyclic LR near the end).
Finds flatter minima that generalise better, at nearly zero extra cost. First-
class in PyTorch: `torch.optim.swa_utils.AveragedModel` + `SWALR`, and remember
`update_bn()` to recompute BatchNorm stats on the averaged weights.

- **Fit here:** high relevance, minimal code, composes with OneCycle (run SWA
  over the last ~25% of epochs). One of the best effort-to-payoff items.
- **Expected:** +0.001–0.003 fairly reliably for a single model.
- Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (2018), *Averaging Weights
  Leads to Wider Optima and Better Generalization* — arXiv:1803.05407 (UAI).

### C2. EMA of weights (exponential moving average)
Maintain a shadow copy of the weights updated as `ema = decay*ema +
(1-decay)*w` (decay ~0.999) and evaluate with it. Same spirit as SWA, even
cheaper, applied continuously. Often a small, reliable bump; re-run BN stats or
keep BN in eval-tracking.

- **Fit here:** trivial to add; try alongside or instead of SWA.
- Widely used; see Polyak–Ruppert averaging and the SWA paper above for the
  theory link.

### C3. Snapshot ensembling / cyclic LR
Instead of training N independent nets for the ensemble, use a cyclic LR
(cosine warm restarts) within a single long run and checkpoint at each cycle's
minimum — you get M diverse members for ~1 training budget.

- **Fit here:** cuts the cost of your diverse ensemble; complements (doesn't
  replace) architecturally-diverse members.
- Huang et al. (2017), *Snapshot Ensembles: Train 1, Get M for Free* —
  arXiv:1704.00109 (ICLR). Cyclic schedule: Loshchilov & Hutter (2017), *SGDR* —
  arXiv:1608.03983 (ICLR).

---

## D. Optimisation (large-batch specific)

### D1. LAMB / LARS (layer-wise adaptive LR for large batches)
Designed for exactly the large-batch regime. Layer-wise trust-ratio scaling lets
you push batch size without the usual generalisation loss. Only worth it if you
*want* to go beyond 8192 for wall-clock reasons — otherwise 8192 + AdamW is
already your best region, so this is speed-motivated, not accuracy-motivated.

- LAMB: You et al. (2019), arXiv:1904.00962. LARS: You, Gitman, Ginsburg
  (2017), arXiv:1708.03888.

### D2. Lookahead
Wraps AdamW: take k fast steps, then interpolate back toward a slow weight copy.
Cheap stability/generalisation bump, composes with AdamW.

- Zhang, Lucas, Hinton, Ba (2019), *Lookahead Optimizer* — arXiv:1907.08610
  (NeurIPS).

---

## E. HPO procedure improvements (Optuna)

Observations on the current search itself, not the model:

- **Early stopping is commented out**, so every non-pruned trial runs the full
  100 epochs even after val AUC plateaus. With OneCycle the best checkpoint is
  usually well before epoch 100. Either re-enable patience-based early stopping
  or shorten search epochs (e.g. 40–50) and retrain the top-k configs longer.
  This roughly doubles trials/hour at negligible AUC cost.
- **Pruner:** `MedianPruner` works, but `HyperbandPruner` (multi-fidelity,
  successive halving) allocates budget better across many trials and pairs well
  with the shortened-epoch idea above. Akiba et al. (2019), *Optuna* —
  arXiv:1907.10902; Hyperband: Li et al. (2018), arXiv:1603.06560.
- **Standardisation:** the notebook feeds raw inputs — collected with the other
  `DNN_PLAN.md` gaps in §F below.
- **Checkpoint-reload fragility (minor):** conditional `nn.Dropout` insertion in
  `nn.Sequential` makes state-dict keys index-dependent, which is what breaks the
  ensemble-reload cell when loading checkpoints saved under a different dropout
  code path. Use named sub-modules or an always-present `Dropout(p)` (p may be 0)
  so indices stay stable across code changes.

---

## F. Not-yet-adopted techniques from `DNN_PLAN.md`

These are already recommended in `DNN_PLAN.md` but are **not** in the current
`OptunaOptimiseDnn.ipynb`. Collected here so the whole to-do lives in one place;
`§` numbers point back to the plan. Ordered by likely impact.

### Affects model quality
- **Standardise inputs (z-score on train stats only)** — `DNN_PLAN.md` §7, §8.2,
  §12 call this mandatory and the "#1 killer." The notebook feeds raw features;
  the first `BatchNorm1d` rescales the first layer's output, which is why you're
  at ~0.89 and not stalled — but standardising the raw inputs is free and improves
  first-layer conditioning. Fit `StandardScaler` (or a quantile/log transform for
  the skewed positive kinematics) on **train only**; never touch val/test stats.
- **Diverse ensembling** — the plan's headline lever (`DNN_PLAN.md` §11): single
  0.891 → ensemble 0.893. Your `objective` trains singles, and the ensemble cell
  currently crashes on reload (see the checkpoint-fragility note in §E above).
  Fixing that and averaging the top ~6 architecturally-distinct configs is the
  most reliable way to convert your existing singles into the dataset ceiling.
  Composes with the SWA / EMA / snapshot methods in §C above.
- **TabM / RealMLP** — `DNN_PLAN.md` §3, §11 flag TabM as "the most likely thing
  to edge past a plain MLP" (`pip install pytabkit`). Bigger lift than any single
  knob; worth a separate experiment rather than a search dimension.

### Cheap refinements
- **No-weight-decay group for BatchNorm gains & biases** — `DNN_PLAN.md` §8.5.
  You pass `model.parameters()` as one group, so AdamW decays BN scale/shift and
  biases too. Put them in a `weight_decay=0` param group; standard, marginally
  helps.
- **Gradient clipping** `clip_grad_norm_(1–5)` — `DNN_PLAN.md` §8.9. Not in the
  training loop; cheap stability insurance, especially as you push depth.
- **Early stopping** — `DNN_PLAN.md` §8.10. You keep the best-val checkpoint, but
  the patience block is commented out, so every un-pruned trial runs the full 100
  epochs (also noted in §E above).

### Methodology / comparability
- **Canonical split (last 500k = test, 500k = val)** — `DNN_PLAN.md` §7, §8.1.
  You use a random `train_test_split(random_state=42)`; the plan says random is
  fine for the hackathon, but the last-500k split is what makes AUC directly
  comparable to the paper.
- **Feature-set comparison (low vs all vs high)** — `DNN_PLAN.md` §7, §11. You're
  fixed to `"low"`; the paper's headline result is showing the 21-feature net
  matches/beats the 28-feature one.
- **`.npy` cache (`mmap_mode='r'`)** — `DNN_PLAN.md` §7. You re-read the 8 GB CSV
  each session; caching to `.npy` cuts reload to seconds. Engineering only, no AUC
  effect.

### Evaluation / reporting
- **Background rejection at fixed signal efficiency** — `DNN_PLAN.md` §10 (the
  paper's Fig. 7 metric). Not computed; it's the physics-native way to report the
  result alongside ROC-AUC.

---

## References (arXiv IDs)

The four most load-bearing citations were verified against arXiv during writing
(2026-07-22); the rest are canonical papers cited from knowledge (cutoff Jan
2026) — IDs are stable, but confirm before quoting in a write-up.

**Verified:**
1. Gorishniy, Rubachev, Khrulkov, Babenko (2021), *Revisiting Deep Learning
   Models for Tabular Data* — [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
   ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)).
2. Gorishniy, Rubachev, Babenko (2022), *On Embeddings for Numerical Features in
   Tabular Deep Learning* —
   [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)
   ([NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf)).
3. Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (2018), *Averaging Weights
   Leads to Wider Optima and Better Generalization* —
   [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
   ([SWA code](https://github.com/timgaripov/swa); now in `torch.optim.swa_utils`).
4. Hoffer, Hubara, Soudry (2017), *Train longer, generalize better…* (Ghost
   Batch Normalization) — [arXiv:1705.08741](https://arxiv.org/abs/1705.08741).

**Canonical (from knowledge):**
5. Klambauer et al. (2017), *Self-Normalizing Neural Networks* — arXiv:1706.02515.
6. Arik & Pfister (2019/2021), *TabNet* — arXiv:1908.07442.
7. Müller, Kornblith, Hinton (2019), *When Does Label Smoothing Help?* — arXiv:1906.02629.
8. Zhang, Cissé, Dauphin, Lopez-Paz (2018), *mixup* — arXiv:1710.09412.
9. Huang et al. (2017), *Snapshot Ensembles* — arXiv:1704.00109.
10. Loshchilov & Hutter (2017), *SGDR: Warm Restarts* — arXiv:1608.03983.
11. You et al. (2019), *LAMB / Large-Batch Optimization* — arXiv:1904.00962.
12. You, Gitman, Ginsburg (2017), *LARS* — arXiv:1708.03888.
13. Zhang, Lucas, Hinton, Ba (2019), *Lookahead Optimizer* — arXiv:1907.08610.
14. Akiba et al. (2019), *Optuna* — arXiv:1907.10902; Li et al. (2018),
    *Hyperband* — arXiv:1603.06560.
