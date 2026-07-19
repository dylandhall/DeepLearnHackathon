# Plan: Reproducing the Baldi et al. (2014) Deep Net on HIGGS with modern PyTorch

**Goal:** beat your LightGBM baseline (test AUC ≈ 0.852) and reproduce the paper's
deep-network result (test AUC ≈ **0.88**) using a modern deep-learning library.

**Status: achieved, including the paper's best number.** The companion script
[`higgs_dnn_example.py`](./higgs_dnn_example.py) was run end-to-end on the full
11M-row dataset on your machine and reached **test AUC = 0.879 in ~55 seconds of
GPU training** (30 epochs × 1.7 s, after a one-time CSV read + ~10 s
`torch.compile` warm-up). A follow-up **8-model diverse ensemble**
([`higgs_dnn_ensemble.py`](./higgs_dnn_ensemble.py), ~11 min total) reached
**test AUC = 0.885 — matching the paper's best deep-net result** — with the best
single net at 0.883. This document explains every decision so you can understand
it, tune it, and implement it yourself.

Paper: Baldi, Sadowski & Whiteson, *Searching for Exotic Particles in High-Energy
Physics with Deep Learning*, **Nature Communications 5:4308 (2014)** —
<https://www.nature.com/articles/ncomms5308> · arXiv:[1402.4735](https://arxiv.org/abs/1402.4735).
Original code: <https://github.com/uci-igb/higgs-susy> (Theano + Pylearn2, Python 2.7).

---

## 1. TL;DR — what to do

1. **Library: plain PyTorch MLP, written from scratch.** It is the 2026 industry/research
   standard, it *is* the paper's model (a fully-connected net), you already have it
   installed with working GPU support, and it teaches you the fundamentals. See §3.
2. **Architecture:** 4–5 hidden layers × 300 units, `Linear → BatchNorm → ReLU → Dropout`,
   single logit output. See §4–§6.
3. **The three modern tricks that matter most** (in order): (a) **standardize the inputs**
   — the #1 cause of a stuck AUC; (b) **keep the whole dataset resident on the GPU** and
   batch by index — ~100× faster than a `DataLoader` here; (c) **bf16 autocast +
   `torch.compile`**. See §7–§8.
4. **Optimizer/schedule:** AdamW + OneCycleLR, large batches (8k–65k), BCEWithLogitsLoss.
5. **Evaluate AUC on probabilities, never on 0/1 predictions** (a bug currently in your
   notebook — see §10).
6. Want to push past 0.88? More epochs, wider net, an ensemble, or drop in **TabM**. See §11.

---

## 2. Where you are vs. the target

| Model | Feature set | Test AUC | Notes |
|---|---|---|---|
| Paper — BDT (TMVA, 2014) | complete (28) | 0.81 | 2014-era boosted trees |
| Paper — shallow NN | complete (28) | 0.816 | 1 hidden layer |
| **Your LightGBM** (native `train()`, GPU) | complete (28) | **≈ 0.852** | modern GBDT, already ≫ paper's BDT |
| Paper — **deep net** | low-level (21) | **0.880** | learns the high-level features itself |
| Paper — **deep net** | complete (28) | **0.885** | best reported result (5.0σ) |
| **This plan — single MLP** | complete (28) | **0.879 (measured)** | `[300]×5`, 30 epochs, ~55 s |
| **This plan — best single net** | complete (28) | **0.883 (measured)** | pyramid `[1024,512,256,128]`, GELU |
| **This plan — 8-model ensemble** | complete (28) | **0.885 (measured)** | matches the paper's best |

Two honest framing notes:
- Your LightGBM (~0.852) already **beats the paper's 2014 boosted trees (0.81)** by a wide
  margin — GBDTs improved a lot in 10 years. The DNN's real edge over a *modern* GBDT is
  therefore a few AUC points, not the 0.88-vs-0.81 headline. That few points is exactly
  what you're chasing, and the DNN delivers it (0.879 vs 0.852).
- The AUC scores in your notebook that read ~0.74–0.77 are computed on **thresholded 0/1
  predictions** and are meaningless as AUC; the correct probability-based numbers in the
  same cells are ~0.82 (`.fit`) and ~0.852 (native `train()`). See §10.

---

## 3. Library choice (researched, 2026)

**Recommendation: PyTorch, hand-written MLP.** Reasons: it's what the paper's model is;
PyTorch is the mainstream standard in 2026 (dominant research share, majority of new
production, Linux-Foundation governed); the whole modern tabular-DL ecosystem (TabM, RTDL,
pytabkit/RealMLP, TabR) is PyTorch-native; and you get full control to learn the mechanics.
You already have `torch 2.13 + CUDA 13` working on the RTX 5070 Ti (verified: bf16 supported,
`torch.compile` works).

**The tree-vs-DL question, accurately:** On *medium* tabular data (~10k rows), well-tuned
GBDTs (XGBoost/LightGBM/CatBoost) still match or beat deep nets — that's the Grinsztajn et al.
(2022) result and remains the safe default. **HIGGS is the atypical case where deep learning
genuinely wins**, for three concrete reasons:
1. **Scale:** 11M rows. The "trees win" literature is scoped to ~10k-row data; the gap shrinks
   or reverses with millions of clean examples (McElfresh et al. 2023 — dataset size/properties
   matter more than model family).
2. **This is literally the paper's point:** a deep MLP beats BDTs here and rediscovers the
   physics high-level features from the raw ones.
3. **Clean signal:** 28 homogeneous numeric columns, no categoricals, few useless features —
   this removes the inductive-bias disadvantages that usually hurt nets on messy tabular data.

**Ranked options:**

| Option | When to use | Notes |
|---|---|---|
| **Plain PyTorch MLP** *(pick this)* | Learn PyTorch + replicate the paper | Full control, GPU-ready, scales trivially to 11M rows |
| **TabM** (`yandex-research/tabm`, ICLR 2025) | Push accuracy past the plain MLP | MLP-based ensemble; top-tier modern DL tabular model; scales to large data; easiest install via `pytabkit` |
| **RealMLP / pytabkit** (`dholzmueller/pytabkit`, NeurIPS 2024) | Near-SOTA DL with pre-tuned defaults, sklearn-style API | Great time/accuracy tradeoff, less hand-tuning |
| **TabNet** (`pytorch-tabnet`, already installed) | You already tried it | Slower, older (2019); the paper-style MLP outperforms it here — consistent with your notebook note |
| FT-Transformer / SAINT / TabR | *Not recommended here* | Attention/retrieval models are unnecessarily heavy at 11M rows |
| TabPFN v2 | *Not applicable* | Foundation model capped around ~10k–1M rows; 11M is far beyond it |

Published modern HIGGS results cluster at **~0.876–0.885** (DiffGP 0.878; deep-GP 0.877; DNN
baselines 0.876; polyharmonic-cascade 0.884–0.885 after ~500 epochs). So **~0.88 is the
realistic strong target**; claims far above it are not reliably reproduced.

---

## 4. The paper's exact recipe (verified against the PDF and the original code)

**Dataset (HIGGS):** 11,000,000 events, ~53% signal. 28 features = **21 low-level** kinematic
(lepton & 4-jet pT/η/φ, missing-energy magnitude/φ, b-tags) + **7 high-level** derived masses
(`m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb`). **Test = last 500,000 events.**

**Best deep net:**
- **"Five-layer network, 300 units per layer."** The reference code files are named
  `layers4_width300_...` → **4 hidden tanh layers × 300 + a sigmoid output** (5 layers total).
  Largest deep net = 279,901 parameters.
- **Activation:** `tanh` on all hidden units.
- **Weight init:** normal(0, σ) with σ = 0.1 (first layer), 0.05 (other hidden), 0.001 (output).
- **Optimizer:** mini-batch SGD + momentum, **batch size 100**.
- **Learning rate:** 0.05, decayed ×1.0000002 every update down to 1e-6.
- **Momentum:** ramped 0.9 → 0.99 linearly over the first 200 epochs, then constant.
- **Weight decay (L2):** 1e-5 per layer.
- **Preprocessing:** standardize features to mean 0 / sd 1 (features that are strictly positive
  were instead scaled to mean 1).
- **Early stopping** on the 500k validation set; training ran for **hundreds of epochs**.
- **Dropout variant:** 50% dropout on the **top hidden layer** → AUC 0.88 on both low-level and
  complete sets (the 0.885 headline result did *not* use dropout).
- **Hyperparameter search was explicitly "not thorough"** (compute-limited); most training
  hyperparameters were fixed by hand. No Bayesian optimization was used. → **There is room to do
  better than the paper with modern tuning.**

**Full results (Table 1):**

| Technique | Low-level (21) | High-level (7) | Complete (28) |
|---|---|---|---|
| BDT | 0.73 | 0.78 | 0.81 |
| Shallow NN | 0.733 | 0.777 | 0.816 |
| **Deep net** | **0.880** | 0.800 | **0.885** |

Headline finding: the deep net on **low-level features alone (0.880)** beats shallow NN/BDT on
the *complete* set — it automatically learns the hand-engineered high-level features. Reproducing
the **low-level-only** experiment (`feature_set="low"` in the script) is the most impressive thing
you can show.

---

## 5. What the original UCI code actually did

Repo <https://github.com/uci-igb/higgs-susy>, folder `higgs/`. Framework: **Pylearn2 on Theano**
(Python 2.7; the repo README itself now points people to Keras/TensorFlow). Files encode the config
in their names, e.g. `layers4_width300_lr005_m200_wd000001_all.py`:

- `layers4` = 4 hidden layers (`layers1` = the shallow baseline); `width300` = 300 units;
  `lr005` = LR 0.05; `m200` = momentum saturates at epoch 200; `wd000001` = weight decay 1e-5.
- Suffix selects the feature set via `derived_feat`: `raw`→21 low-level (`False`), `all`→28
  (`True`), `only`→7 high-level (`'only'`). `benchmark=1` selects HIGGS.
- Model = `pylearn2.models.mlp.MLP` with `Tanh` hidden layers + a single `Sigmoid` output.
- Cost = `SumOfCosts([Default(), WeightDecay(coeffs=[1e-5]*num_layers)])`.
- `MomentumAdjustor(start=0, saturate=200, final_momentum=0.99)`,
  `ExponentialDecay(decay_factor=1.0000002, min_lr=1e-6)`, `batch_size=100`.

Our PyTorch script is a faithful modern translation of this, with the 2014→2026 upgrades in §6.

---

## 6. 2014 → 2026: what we change and why

| Aspect | Paper (2014) | This plan (2026) | Why |
|---|---|---|---|
| Framework | Theano + Pylearn2 (dead) | **PyTorch 2.13** | Maintained, fast, standard |
| Activation | tanh | **ReLU** (+ BatchNorm) | Trains faster, no saturation; tanh still available |
| Normalization | none | **BatchNorm1d** | Biggest single stabiliser; huge batches → rock-solid BN stats |
| Optimizer | SGD, momentum 0.9→0.99 | **AdamW** | Converges in far fewer epochs; less tuning |
| LR schedule | manual ×1.0000002 decay | **OneCycleLR** | Super-convergence; robust for MLPs |
| Batch size | 100 | **8k–65k** | Tiny MLP + huge data → big batches keep the GPU busy |
| Precision | fp32 | **bf16 autocast** (no GradScaler) | ~2× throughput; bf16 is stable on Blackwell |
| Compile | — | **`torch.compile`** | Kernel fusion; verified working here |
| Data feeding | disk/host batches | **whole dataset resident on GPU** | ~100× faster than a DataLoader at this size |
| Epochs to ~0.88 | hundreds | **~30** | AdamW + OneCycle + big batch converge fast |
| Wall-clock | (2011 Tesla C2070) | **~1 min** on RTX 5070 Ti | 14 years of hardware + software |

Two configs are exposed in the script:
- **Modern (default):** ReLU + BatchNorm + AdamW + OneCycle → 0.879 in ~1 min.
- **Paper-faithful:** set `activation="tanh"`, `batchnorm=False`, and (per the comment) swap in
  `SGD(lr=0.05, momentum=0.9, weight_decay=1e-5)`. Slower, mostly of historical interest.

---

## 7. Data pipeline

- **Load once, cache:** read `HIGGS.csv` (8 GB, no header, col 0 = label) as float32, then cache to
  `.npy` so reloads take seconds (`mmap_mode='r'`). 11M × 29 × 4 B ≈ 1.3 GB.
- **Split (canonical benchmark):** **last 500,000 rows = test**, another 500,000 = validation,
  the remaining ~10M = train. Using the *last* 500k is what makes your AUC directly comparable to
  the paper. (Your notebook's random split is fine for the hackathon, but the fixed split is the
  standard.) **Touch the test set exactly once, at the end.**
- **Standardize:** z-score each feature using **train-split statistics only** (avoids leakage).
  *This is mandatory* — unstandardized inputs are the classic reason HIGGS AUC stalls at 0.5–0.75.
- **Resident on GPU:** move the full standardized `X`/`y` tensors to the GPU **once**; each batch is
  a pure device-side index-gather (`X_gpu[perm[i:i+bs]]`). No per-batch host→device copy. (If your
  data did *not* fit in VRAM, you'd instead use `TensorDataset + DataLoader(pin_memory=True,
  num_workers=8–16, persistent_workers=True, non_blocking=True)`.)
- **Feature-set experiments:** `feature_set="all"` (28, → 0.885 target), `"low"` (21, → 0.880 —
  the headline experiment), `"high"` (7, → 0.800).

---

## 8. Training recipe (ordered)

1. Load + cache; split (last 500k = test, 500k = val, rest = train).
2. Standardize on **train stats only**; cast to float32.
3. Move full `X`/`y` to GPU once. **Do not** wrap resident tensors in a DataLoader.
4. Build the MLP: `Linear → BatchNorm1d → ReLU → Dropout` × N hidden, then `Linear(→1)` (one logit).
5. Loss `BCEWithLogitsLoss`; optimizer `AdamW(lr, weight_decay=1e-5…1e-4)`. (Optional refinement:
   put BatchNorm params & biases in a `weight_decay=0` group.)
6. Perf flags: `torch.set_float32_matmul_precision('high')` (TF32), `model = torch.compile(model)`.
7. Large batch (8k–65k). `steps_per_epoch = ceil(N_train / bs)`.
8. `OneCycleLR(max_lr≈2e-3–3e-3, pct_start=0.1, epochs, steps_per_epoch, anneal='cos')`; **step it
   every batch**.
9. Epoch loop: `perm = torch.randperm(N, device='cuda')`; per batch → index the resident tensors →
   forward under `autocast(bfloat16)` → `loss.backward()` → `optimizer.step()` → `scheduler.step()`
   → `zero_grad(set_to_none=True)`. (Optional `clip_grad_norm_(…, 1–5)`.)
10. Each epoch: `model.eval()` + `no_grad`, compute sigmoid probs on the val tensor in chunks →
    `roc_auc_score`. Keep the **best-val checkpoint** (early stopping).
11. Give it enough epochs to converge (watch val AUC plateau); the script's 30 is enough for ~0.879,
    50–80 + a wider net for 0.885+.
12. Final: restore best checkpoint, compute **test** AUC once.

Sanity target: a correct run reaches **~0.88**. Treat anything **< 0.84 as a bug** — almost always
standardization, LR, batch size, or too few epochs.

---

## 9. How to run the example

```bash
cd /home/dylan/code/DeepLearnHackathon/HiggsBosonClassificationChallenge
source .venv/bin/activate
python higgs_dnn_example.py          # full 11M rows; ~1 min after the first CSV read
```

Quick smoke test first (recommended): in the `CFG` dict set `n_rows=2_000_000` and `epochs=5`,
confirm it runs, then scale up. The first full run reads the 8 GB CSV (~1–2 min) and writes
`higgs_cache.npy`; later runs load the cache in seconds.

Actual output from your machine (defaults, all 28 features):

```
train=10,000,000  val=500,000  test=500,000  features=28
torch.compile: enabled
epoch  1/30  loss=0.5472  val_auc=0.8261  ... (9.7s)   <- first epoch includes compile warm-up
epoch 10/30  loss=0.4512  val_auc=0.8672  ... (1.7s)
epoch 20/30  loss=0.4253  val_auc=0.8773  ... (1.7s)
epoch 30/30  loss=0.4125  val_auc=0.8786  ... (1.7s)
============================================================
BEST VAL AUC : 0.8785
TEST AUC     : 0.8790   (paper target: 0.885 all / 0.880 low-level)
============================================================
```

---

## 10. Evaluation — and a bug to fix in your notebook

**AUC must be computed on continuous scores** (`predict_proba(...)[:,1]` for LightGBM, or
`sigmoid(logits)` for the net), **never on hard 0/1 predictions.** In your current notebook,
`roc_auc_score(y_test, predictions)` where `predictions` are thresholded labels prints ~0.74–0.77 —
that number is meaningless as an AUC. The correct calls in the same cells
(`plot_roc_curve(y_test, y_hat)` / `(y_test, prob)`) give ~0.82 and **~0.852**, which is your true
LightGBM performance. The script's `compute_auc()` already does this correctly.

Report, for the DNN: ROC curve + AUC on the test set, a confusion matrix / classification report at
threshold 0.5, and a comparison table against your LightGBM. For a physics-flavoured extra, quote
background rejection at fixed signal efficiency (as in the paper's Fig. 7).

---

## 11. Pushing past 0.88 — what actually worked (measured)

I ran a validation-selected architecture search and then a diverse ensemble
([`higgs_dnn_ensemble.py`](./higgs_dnn_ensemble.py)) on your machine. Findings:

- **Single models plateau at ~0.876–0.883** regardless of width — this is the known HIGGS
  ceiling. **Wider nets overfit** at long training with light dropout (val AUC peaked ~epoch 20
  then fell); best-val checkpointing or heavier dropout fixes it.
- **Best single net (test 0.883):** pyramid `[1024, 512, 256, 128]`, GELU, BatchNorm,
  dropout 0.1 on all layers, AdamW(lr 2.5e-3, wd 1e-4), OneCycle, batch 16 384, 45 epochs.
- **The ensemble is what reaches the paper's number.** Averaging the probabilities of **8
  deliberately different** nets (varied width/depth/activation/dropout/seed) climbed steadily:
  0.879 (1) → 0.881 (2) → 0.884 (5) → **0.885 (8)**. Diversity, not just more seeds, is the key —
  decorrelated errors cancel. This is also how the paper reported means over random inits.

Other levers (all in the script's `CFG`):

- **More epochs + wider net:** val AUC was still rising at epoch 30. Try `hidden_layers=[512]*5`,
  `epochs=60–80` with best-val checkpointing. Expect ~0.882–0.884 single.
- **The paper's dropout variant:** it's already the default (`dropout=0.5` on the top hidden layer).
  Try `dropout_all_layers=True` with a lower rate (0.1–0.2) — with 10M rows the data itself
  regularizes, so lean light.
- **Ensemble:** train 3–5 nets with different seeds, average their probabilities. Reliably worth
  +0.001–0.003 AUC and how the paper reported means.
- **Reproduce the low-level story:** run `feature_set="low"` and show the 21-feature net (~0.88)
  beats your 28-feature LightGBM — the paper's central result.
- **TabM** (`pip install pytabkit`, then its TabM estimator): the current top-tier DL tabular model,
  MLP-based, scales to 11M rows; the most likely thing to edge past a plain MLP.
- **Tune** `max_lr`, `weight_decay`, `batch_size`, depth/width with Optuna (you already use it) on
  the **validation** set.

---

## 12. Pitfalls (each of these silently costs AUC)

- **Unstandardized inputs** — #1 killer; stalls AUC at 0.5–0.75.
- **DataLoader over a GPU-resident tensor** — per-batch copies make it ~100× slower.
- **Too-small batch** on 10M rows — wastes the GPU, destabilizes BatchNorm.
- **Too few epochs** — under-training reads as "0.82 is the ceiling"; it isn't.
- **Wrong LR** — too high (with big batches) → NaN/divergence; too low → stuck. Warm up, scale with
  batch size (AdamW ~1e-3–3e-3).
- **BatchNorm left in train mode at eval** (forgot `model.eval()`), or **leaking test rows** into the
  standardization stats.
- **GradScaler mismatch** — bf16 needs *no* scaler; fp16 *needs* one. This script uses bf16 (no scaler).
- **`torch.compile` recompiles** if the last batch is a different size — fixed batch / `drop_last`, and
  warm up before timing.
- **Metrics/loss in low precision** — keep `BCEWithLogitsLoss` and the AUC computation in fp32.
- **Over-regularizing** with 10M rows (0.5 dropout *and* high weight decay) → underfit.

---

## 13. Folding it into the hackathon notebook

- **Task 2 (train + optimize on val):** add cells that (a) standardize, (b) build `HiggsMLP`, (c) run
  the training loop reporting **val** AUC per epoch, (d) show the val-AUC curve as your "optimization"
  evidence. Keep LightGBM as the stated baseline.
- **Task 3 (test-set analysis):** load the best checkpoint, compute **test** AUC once, plot ROC +
  confusion matrix, and present the comparison table from §2. Optionally add the `feature_set="low"`
  experiment as a highlight.
- You can import from the script (`from higgs_dnn_example import HiggsMLP, compute_auc, standardize`)
  or paste the pieces into cells. Set a smaller `n_rows` while iterating in Colab.

---

## 14. References (all verified)

1. Baldi, Sadowski & Whiteson (2014), *Searching for Exotic Particles in HEP with Deep Learning*,
   Nature Comm. 5:4308 — <https://www.nature.com/articles/ncomms5308> · arXiv:1402.4735.
2. Original code — <https://github.com/uci-igb/higgs-susy>.
3. UCI HIGGS dataset — <https://archive.ics.uci.edu/dataset/280/higgs>.
4. Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform DL on typical
   tabular data?*, NeurIPS D&B — arXiv:2207.08815.
5. McElfresh et al. (2023), *When Do Neural Nets Outperform Boosted Trees on Tabular Data?*, NeurIPS
   D&B — arXiv:2305.02997.
6. Gorishniy et al. (2024/ICLR 2025), *TabM: Parameter-Efficient Ensembling* — arXiv:2410.24210 ·
   <https://github.com/yandex-research/tabm>.
7. Holzmüller et al. (2024), *Better by Default: Strong Pre-Tuned MLPs & Boosted Trees* (RealMLP),
   NeurIPS — arXiv:2407.04491 · <https://github.com/dholzmueller/pytabkit>.
8. Hollmann et al. (2025), *TabPFN v2*, Nature 637 — doi:10.1038/s41586-024-08328-6.
9. PyTorch AMP / mixed precision — <https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/>
   · <https://docs.pytorch.org/docs/main/amp.html>.
```
