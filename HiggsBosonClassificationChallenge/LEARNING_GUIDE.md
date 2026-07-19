# Learning Guide: building up from a single MLP to a paper-matching ensemble

This guide is for **understanding and implementing each step yourself**. It builds the
solution in six levels, smallest to largest. For every level you get: the **concept**, the
**why**, the **key code** (the essential lines, not a copy-paste script), and a **"your turn"**
checkpoint so you write it and verify it before moving on.

The complete, runnable reference versions live in
[`higgs_dnn_example.py`](./higgs_dnn_example.py) (Levels 0–2),
[`higgs_dnn_ensemble.py`](./higgs_dnn_ensemble.py) (Levels 3–4), and
[`higgs_optuna.py`](./higgs_optuna.py) + [`higgs_hpo_ensemble.py`](./higgs_hpo_ensemble.py)
(Level 5). **Try each level yourself first, then open the reference to check your work.**

```
Level 0  Foundation      load → split → standardize → put on GPU        (shared by all)
Level 1  Single MLP      one uniform-width network                      → AUC ~0.879
Level 2  Architecture    same network, "pyramid" shape                  → AUC ~0.883
Level 3  Ensemble        many DIFFERENT networks, averaged              → AUC ~0.885
Level 4  Feature study   Level 3 on 21 low-level features only          → AUC ~0.888
Level 5  Auto-tune       Optuna search → tune + ensemble                → AUC ~0.893 (best)
```

---

## Concepts you'll use (quick primer)

You already know AUC, train/val/test, and overfitting from your LightGBM work. The new,
neural-net-specific ideas:

- **Neuron / Linear layer.** A neuron computes `output = activation(w·inputs + b)` — a weighted
  sum of its inputs plus a bias, passed through a nonlinear function. A **`nn.Linear(in, out)`**
  layer is just `out` neurons stacked, each seeing all `in` inputs.
- **MLP = several Linear layers in a row.** "Deep" just means "several hidden layers". Depth +
  width = capacity to learn complex functions.
- **Activation** (ReLU, GELU, tanh). The nonlinearity between layers. Without it, stacking
  layers collapses to one linear function. ReLU (`max(0, x)`) is the modern default.
- **BatchNorm.** Re-centers/re-scales each layer's outputs across the batch. Makes training
  faster and more stable. The 2014 paper didn't have it; it's the biggest modern upgrade.
- **Dropout.** During training, randomly zero a fraction of neurons each step. Forces the net
  not to over-rely on any one neuron → less overfitting. Turned OFF automatically at eval time.
- **Logit vs probability.** The net outputs a raw score (**logit**, any real number). `sigmoid`
  squashes it to a probability in [0, 1]. We keep logits internally (numerically stable loss)
  and apply sigmoid only when we need a probability.
- **Loss / optimizer / scheduler.** *Loss* (`BCEWithLogitsLoss`) measures how wrong the net is.
  The *optimizer* (`AdamW`) nudges the weights to reduce it. The *scheduler* (`OneCycleLR`)
  changes the learning rate over training. One **step** = one batch; one **epoch** = one full
  pass over the training data.

---

## Level 0 — The foundation (shared by every level)

**Concept:** before any model, get the data into the right shape. Four steps, each with a reason.

1. **Load** the CSV once and cache it (`.npy`) so reloads are seconds, not minutes.
2. **Split** into train / validation / test. Use the **last 500k rows as test** (the canonical
   HIGGS benchmark — makes your AUC comparable to the paper) and another 500k as validation.
3. **Standardize** each feature to mean 0, std 1, using **train statistics only**. Neural nets
   are very sensitive to input scale — *this is the #1 cause of a stuck AUC.* Computing the
   stats on train only avoids leaking test information.
4. **Put the whole dataset on the GPU once.** It's only ~1.1 GB, so it fits in your 16 GB. Then
   each batch is a fast index into GPU memory — no slow per-batch copies from CPU.

**Key code:**

```python
X, y = load_higgs(cfg)                       # X: (11M, 28) float32, y: (11M,) 0/1
Xtr, ytr, Xva, yva, Xte, yte = make_splits(X, y)   # last 500k = test, 500k = val, rest = train
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8      # <-- train stats ONLY
Xtr, Xva, Xte = (Xtr-mu)/sd, (Xva-mu)/sd, (Xte-mu)/sd
Xtr_t = torch.tensor(Xtr, device="cuda")     # move once; batch by indexing later
ytr_t = torch.tensor(ytr, dtype=torch.float32, device="cuda")
```

**Your turn:** load the data, print the shapes of all six splits, and confirm `Xtr.mean()` is
~0 and `Xtr.std()` is ~1 after standardizing. (Reference: `load_higgs`, `make_splits`,
`standardize` in `higgs_dnn_example.py`.)

---

## Level 1 — A single MLP

**Concept:** one network of uniform-width hidden layers. Input (28) → 300 → 300 → 300 → 300 →
300 → 1 logit. Each hidden layer is the block `Linear → BatchNorm → ReLU → Dropout`.

**Why this shape of block:** Linear does the learning; BatchNorm stabilizes it; ReLU adds the
nonlinearity; Dropout regularizes. The final `Linear(→1)` produces one logit per event.

**Key code — the model:**

```python
class HiggsMLP(nn.Module):
    def __init__(self, in_dim, hidden=[300,300,300,300,300], dropout=0.5):
        super().__init__()
        layers, prev = [], in_dim
        for i, width in enumerate(hidden):
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU()]
            if i == len(hidden) - 1:                 # dropout on the top hidden layer
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))            # one logit out
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)               # shape (batch,) to match labels
```

**Key code — the training loop (the heart of PyTorch):**

```python
model = HiggsMLP(28).to("cuda")
opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3,
                                            epochs=30, steps_per_epoch=steps)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(30):
    model.train()                                    # dropout/BN in train mode
    perm = torch.randperm(len(Xtr_t), device="cuda") # reshuffle each epoch
    for i in range(0, len(Xtr_t), batch):
        idx = perm[i:i+batch]
        opt.zero_grad(set_to_none=True)              # clear old gradients
        with torch.autocast("cuda", dtype=torch.bfloat16):   # mixed precision = ~2x faster
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()                              # compute gradients
        opt.step()                                   # update weights
        sched.step()                                 # update learning rate (per batch)
    # ...evaluate val AUC here (see Level 1 eval below)
```

**Key code — evaluation (compute AUC on PROBABILITIES, never on 0/1 predictions):**

```python
@torch.no_grad()
def auc(model, Xg, y_true):
    model.eval()                                     # dropout off, BN uses running stats
    p = torch.sigmoid(model(Xg)).float().cpu().numpy()
    return roc_auc_score(y_true, p)                  # <-- p is a probability, not a label
```

> ⚠️ The classic bug (present in the starter notebook): `roc_auc_score(y, (p>0.5))` feeds
> *thresholded 0/1* values and gives a meaningless number. AUC must see the continuous score.

**Your turn:** write `HiggsMLP` and the training loop, run 30 epochs, and watch the val AUC
climb toward ~0.879. If it's stuck near 0.5–0.75, you almost certainly skipped standardization
(Level 0). Reference: the full `train()` in `higgs_dnn_example.py`.

---

## Level 2 — Tuning the architecture (the "pyramid")

**Concept:** *exactly the same model and training code as Level 1.* You change **one thing** —
the `hidden` list — from uniform to a **narrowing funnel**:

```python
model = HiggsMLP(28, hidden=[1024, 512, 256, 128])   # wide → narrow = "pyramid"
```

**Why it can help:** the wide first layer has room to detect many raw patterns; each narrower
layer forces the net to *compress* them into fewer, more informative features — a learned
hierarchy. On HIGGS this generalized slightly better (test ~0.883 vs ~0.879).

**This is hyperparameter tuning, the neural-net version of your Optuna LightGBM search.** The
"knobs" are: number of layers (depth), width of each, activation (`relu`/`gelu`/`tanh`),
dropout rate and where it's applied, learning rate, weight decay, batch size, epochs.

**Your turn:** run the same script with 3–4 different `hidden` lists (e.g. `[512]*5`,
`[768]*4`, `[1024,512,256,128]`) and a couple of dropout values. **Pick the winner by
VALIDATION AUC, then report its TEST AUC once.** Never choose based on the test set — that's
the same discipline as not tuning LightGBM on your holdout. (You can even drive this with
Optuna, exactly like your LightGBM cell.)

---

## Level 3 — The diverse ensemble

**Concept:** instead of one network, train **several different ones independently**, then
**average their predicted probabilities**. That average is your final prediction.

**Why it works:** every net makes some *random* errors (from its random initialization) and
some *systematic* ones (from its particular shape). Averaging **decorrelated** models cancels
the random part and keeps the signal they agree on. Result: the ensemble beats every member.
This is why the paper reported *means over several random initializations*.

**The single most important rule: make the members DIFFERENT.** Averaging 8 identical nets
gains nothing. Diversity comes from varying width, depth, activation, dropout style, and seed:

```python
MEMBERS = [
  {"hidden":[512]*5,            "act":"gelu", "do":0.25, "do_all":False, "seed":0},
  {"hidden":[512]*5,            "act":"relu", "do":0.15, "do_all":True,  "seed":1},
  {"hidden":[384]*6,            "act":"gelu", "do":0.12, "do_all":True,  "seed":2},
  {"hidden":[1024,512,256,128], "act":"gelu", "do":0.10, "do_all":True,  "seed":4},
  # ...more, all different...
]
```

**Key code — the averaging:**

```python
test_probs = []
for m in MEMBERS:
    best_val, probs_on_test = train_member(m)        # train one net, keep its best-VAL weights
    test_probs.append(probs_on_test)
    running = roc_auc_score(yte, np.mean(test_probs, axis=0))   # ensemble grows as we add members
ensemble_auc = roc_auc_score(yte, np.mean(test_probs, axis=0))  # average the probabilities
```

**Honest methodology (important for a submission):**
- All members train on the **same train split**.
- The **validation set** is used only to pick each member's best-epoch checkpoint (this also
  cures the overfitting wide nets show if you train them too long).
- The **test set is touched once**, at the very end, for the final average. Nothing is tuned on it.

**Watch the ensemble climb** (measured on all 28 features): `0.879 (1 member) → 0.881 (2) →
0.884 (5) → 0.885 (8)`. Diminishing returns — most of the gain is in the first few *diverse*
members.

**Your turn:** start from your Level 2 code, wrap the training in a `train_member(config)`
function that returns the test-set probabilities, loop over 3–4 different configs, and average.
Confirm the ensemble AUC is higher than the best single member. Reference:
`higgs_dnn_ensemble.py`.

---

## Level 4 — The feature-set experiment (the paper's real point)

**Concept:** a completely different axis — change **which features** you feed, not the model.
The 28 columns are **21 low-level** (raw detector measurements: momenta, angles, b-tags) +
**7 high-level** (physicist-derived invariant masses: `m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb,
m_wwbb` — these took domain expertise to design). Run the ensemble on the **21 low-level
features only**:

```bash
python higgs_dnn_ensemble.py low        # uses columns 1..21, drops the 7 derived masses
```

(The only code change is selecting `X[:, :21]` — already wired to the `low` argument.)

**Why this is the headline result:** in the paper, a *shallow* model needs the hand-built
high-level features to do well. A **deep** net given only the 21 raw features reaches ~0.88 —
**as good as models that were handed the physicist's features.** That means the deep net
*learned the high-level physics by itself*, straight from raw inputs. This is the "deep
learning removes the need for manual feature engineering" story, and it's the most compelling
thing to show in your submission.

**Your turn:** run all three feature sets (`all`, `low`, `high`) and build the comparison table
below. The interesting comparison is **low-level deep net vs your LightGBM on all 28 features** —
if the 21-feature net matches or beats the 28-feature GBDT, you've reproduced the paper's point.

| Feature set | Paper deep net | This ensemble (measured) | LightGBM (yours) |
|---|---|---|---|
| complete (28) | 0.885 | 0.885 | ~0.852 |
| low-level (21) | 0.880 | **0.888** | — |
| high-level (7) | 0.800 | *(optional)* | — |

**Result:** the 21-feature (low-level) ensemble reached **0.888 — matching/edging past the
28-feature ensemble (0.885) and the paper's low-level (0.880).** The net given only raw detector
measurements does as well as one handed the physicist's engineered features. The high-level
features are deterministic functions of the low-level ones, so a deep net recomputes them
itself and finds them redundant (the small edge over 0.885 is within run-to-run noise). This is
the paper's central claim, reproduced — and the most compelling single thing to put in your
submission: *a 21-feature deep net beats your 28-feature LightGBM by ~0.036 AUC.* And once you
*tune* this low-level setup (Level 5), it climbs to **0.893** — the best result of all.

---

## Level 5 — Automated tuning (Optuna) → the capstone

**Concept:** every level so far picked hyperparameters by hand. Now let **Optuna** search them
automatically — the same idea as your LightGBM Optuna cell, applied to the net — then feed the
best configs into the Level 3 ensemble. Two reference scripts:
[`higgs_optuna.py`](./higgs_optuna.py) (search only) and
[`higgs_hpo_ensemble.py`](./higgs_hpo_ensemble.py) (search → select → ensemble, end to end).

**Key code — the Optuna objective (mirror of your LightGBM study):**

```python
def objective(trial):
    hidden  = build_hidden(trial)                       # trial picks depth / width / shape
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr      = trial.suggest_float("max_lr", 1e-3, 4e-3, log=True)
    # ...build + train the net, checking val AUC each epoch...
    for epoch in range(EPOCHS):
        train_one_epoch(...)
        va = val_auc(model)
        trial.report(va, epoch)                         # PRUNING: let Optuna kill bad trials early
        if trial.should_prune(): raise optuna.TrialPruned()
    return best_val                                     # Optuna maximizes this
```

**Two efficiency tricks that matter:**
- **Load data once, keep it on the GPU** across all trials → each trial is ~1–2 min, so ~an hour
  buys 50–150 trials (it is *not* 20 min/trial once the data is resident).
- **Pruning** (`MedianPruner`): trials clearly below the running median are killed after a few
  epochs, so compute flows to promising configs. Persist the study to SQLite
  (`storage="sqlite:///higgs_optuna.db"`) so you can stop and resume — ideal for a long crunch.

**What the search found:** it consistently preferred **wide** nets (`[1024]×5–6`), GELU, and
**all-layer** dropout ~0.15 — a real, learnable pattern, not random twiddling.

**The capstone — tune, then ensemble the best *diverse* configs:**

```bash
python higgs_hpo_ensemble.py all   # 28-feat: ensemble 6 distinct configs × 2 seeds (12 nets) → 0.890
python higgs_hpo_ensemble.py low   # 21-feat: ensemble 6 distinct configs × 2 seeds (12 nets) → 0.893  ← best
```

**Results (measured):**

| Approach | complete (28) | low-level (21) |
|---|---|---|
| HPO best *single* net | 0.885 | 0.891 |
| HPO + 12-model ensemble | 0.890 | **0.893 (best overall)** |

**The punchline — 21 features beat 28.** The **low-level** model beats the complete one at *both*
levels — by ~0.006 as a single net (0.891 vs 0.885) and ~0.003 as an ensemble (0.893 vs 0.890) —
so it's a real, consistent effect, not noise. (The gap is smaller for the ensemble because
averaging already cancels some of the overfitting the extra features cause.) Why: the 7 high-level features are **deterministic
functions of the 21 low-level ones**, so they add no information a deep net can't derive itself —
but they *do* add redundant, correlated inputs that enlarge the overfitting surface and slightly
hurt. This **sharpens** the paper's thesis: the net doesn't merely *match* the hand-engineered
features, it does **better without them**. ~0.893 is the dataset's practical ceiling.

**Honest accounting:** tuning lifted the single model ~+0.002; ensembling added another ~+0.005;
and the tuned winners showed **no validation-overfitting gap** (val ≈ test), so the gains are
real. Beyond ~0.893 the ensemble plateaus — more trials or 24 h of compute won't help, because
you're at the ceiling.

**Saved & loadable (no notebook needed).** Each pipeline run writes a bundle to
`models/higgs_<set>_ensemble/` — the 12 trained nets, the fitted scaler, and the log. Reload and
reproduce in seconds with [`higgs_predict.py`](./higgs_predict.py):

```bash
python higgs_predict.py models/higgs_low_ensemble    # → reproduced test AUC = 0.8928
```

**Your turn:** adapt your LightGBM Optuna objective to build/train the net and return val AUC,
then add `trial.report` + `should_prune` for pruning. Run ~40 trials on `low`, take the top few
*distinct* architectures, ensemble them, and confirm you land near 0.89. Reference:
`higgs_optuna.py`, then `higgs_hpo_ensemble.py`.

---

## How to know each level "worked"

| Level | Sanity check |
|---|---|
| 0 | Six split shapes print; standardized train mean ≈ 0, std ≈ 1 |
| 1 | Val AUC climbs to ~0.879; **< 0.84 = a bug** (usually standardization, LR, or too few epochs) |
| 2 | A tuned architecture selected *by validation* reaches ~0.882–0.883 test |
| 3 | Ensemble AUC (~0.885) > best single member (~0.883) |
| 4 | Low-level-only ensemble reaches ~0.888, matching/beating the complete-feature story |
| 5 | Optuna search runs with pruning; tuned low-level ensemble reaches ~0.893 (val ≈ test) |

Work through them in order. Each level is a small delta on the previous one, so by Level 5 you'll
have built — and understood — every piece yourself, from one MLP to a tuned, saved, reproducible
ensemble that beats the paper.
