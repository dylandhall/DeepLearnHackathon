"""
higgs_optuna.py
====================================================================
Optuna hyper-parameter search for the PyTorch DNN -- the neural-net equivalent
of the LightGBM/Optuna cell you already wrote. Same idea: define an objective
that builds + trains a model and returns a score; let Optuna propose smarter
hyper-parameters over many trials.

TWO THINGS THAT MAKE THIS EFFICIENT (and different from a naive loop):
  1. Load the data ONCE and keep it on the GPU; every trial reuses it. A single
     trial is then ~1-2 min, NOT 20 -- so a few hours already buys 50-150 trials.
  2. PRUNING: report the validation AUC after each epoch; Optuna kills trials
     that are clearly losing early, so compute goes to promising configs.

HONEST EXPECTATIONS (read DNN_PLAN / LEARNING_GUIDE too):
  * A single model on HIGGS plateaus ~0.876-0.883; the whole literature clusters
    at 0.876-0.885. We already hit 0.883 single / 0.885 ensemble. So HPO will
    likely gain only ~+0.001-0.002 on a single model -- real but small; you are
    near the dataset's ceiling.
  * With a 500k validation set, AUC is measured to ~+/-0.0007. If you pick the
    best of hundreds of trials by val AUC, some of that "win" is just noise
    (validation overfitting). ALWAYS confirm the winner on the untouched TEST
    set at the end, and don't trust a val gain smaller than ~0.001.
  * Best use of a long run: find a few STRONG, DIVERSE configs here, then
    ENSEMBLE them (see higgs_dnn_ensemble.py). That beats chasing one config.

Run:
    python higgs_optuna.py                 # 40 trials on all 28 features
    python higgs_optuna.py 200             # 200 trials
    python higgs_optuna.py 1000 low        # 1000 trials on the 21 low-level features
The study is saved to higgs_optuna.db (SQLite), so you can Ctrl-C and re-run to
RESUME where you left off -- ideal for a 24-hour crunch.
--------------------------------------------------------------------
"""
import sys
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
import higgs_dnn_example as H

torch.set_float32_matmul_precision("high")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
FEATURE_SET = sys.argv[2] if len(sys.argv) > 2 else "all"
EPOCHS = 30            # fixed per trial so trials are comparable & fast; raise to tune longer
BATCHES = [8192, 16384, 32768]

# ---- load ONCE, keep on GPU, reuse for every trial ----------------------
cfg = dict(H.CFG); cfg["feature_set"] = FEATURE_SET; cfg["n_rows"] = None
X, y = H.load_higgs(cfg)
Xtr, ytr, Xva, yva, Xte, yte = H.make_splits(X, y)
Xtr, Xva, Xte = H.standardize(Xtr, Xva, Xte)
Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
Xva_t = torch.tensor(Xva, device=DEV); Xte_t = torch.tensor(Xte, device=DEV)
IN = Xtr.shape[1]
print(f"feature_set={FEATURE_SET}  in_dim={IN}  train={len(Xtr):,}  trials={N_TRIALS}", flush=True)


@torch.no_grad()
def val_auc(model):
    model.eval()
    p = []
    for i in range(0, len(Xva_t), 131072):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p.append(torch.sigmoid(model(Xva_t[i:i + 131072])).float().cpu().numpy())
    return roc_auc_score(yva, np.concatenate(p))


def build_hidden(trial):
    """Sample an architecture: a base width, a depth, and a shape (uniform/pyramid)."""
    depth = trial.suggest_int("depth", 3, 6)
    base = trial.suggest_categorical("base_width", [256, 384, 512, 640, 768, 1024])
    shape = trial.suggest_categorical("shape", ["uniform", "pyramid"])
    if shape == "uniform":
        return [base] * depth
    # pyramid: halve each layer, floor at 64
    return [max(64, base // (2 ** i)) for i in range(depth)]


def objective(trial):
    # ---- sample hyper-parameters (this is the search space you can edit) ----
    hidden = build_hidden(trial)
    act = trial.suggest_categorical("act", ["relu", "gelu"])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    do_all = trial.suggest_categorical("dropout_all_layers", [True, False])
    lr = trial.suggest_float("max_lr", 1e-3, 4e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch = trial.suggest_categorical("batch", BATCHES)

    torch.manual_seed(0); np.random.seed(0)
    model = H.HiggsMLP(IN, hidden, activation=act, batchnorm=True,
                       dropout=dropout, dropout_all_layers=do_all).to(DEV)
    # NOTE: no torch.compile here -- with a new architecture every trial, compiling
    # would recompile constantly. Eager mode keeps trials fast and stable.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr_t); spe = (n + batch - 1) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=EPOCHS,
                                                steps_per_epoch=spe, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step(); sched.step()
        va = val_auc(model)
        best = max(best, va)
        # ---- PRUNING: tell Optuna how we're doing; let it kill hopeless trials ----
        trial.report(va, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best        # Optuna maximizes the best validation AUC of this config


if __name__ == "__main__":
    # TPE sampler (smart Bayesian search, like your LightGBM study) + median pruner.
    study = optuna.create_study(
        direction="maximize",
        study_name=f"higgs_{FEATURE_SET}",
        storage=f"sqlite:///higgs_optuna.db",   # persistent -> resumable across restarts
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print("\n" + "=" * 60)
    print(f"best validation AUC : {study.best_value:.4f}")
    print("best params         :")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ---- confirm the winner ONCE on the untouched TEST set -------------------
    bp = study.best_params
    hidden = ([bp["base_width"]] * bp["depth"] if bp["shape"] == "uniform"
              else [max(64, bp["base_width"] // (2 ** i)) for i in range(bp["depth"])])
    torch.manual_seed(0); np.random.seed(0)
    model = H.HiggsMLP(IN, hidden, activation=bp["act"], batchnorm=True,
                       dropout=bp["dropout"], dropout_all_layers=bp["dropout_all_layers"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=bp["max_lr"], weight_decay=bp["weight_decay"])
    n = len(Xtr_t); spe = (n + bp["batch"] - 1) // bp["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=bp["max_lr"], epochs=EPOCHS,
                                                steps_per_epoch=spe, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_va = None, 0.0
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bp["batch"]):
            idx = perm[i:i + bp["batch"]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step(); sched.step()
        va = val_auc(model)
        if va > best_va:
            best_va = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    tp = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), 131072):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tp.append(torch.sigmoid(model(Xte_t[i:i + 131072])).float().cpu().numpy())
    test = roc_auc_score(yte, np.concatenate(tp))
    print(f"\nbest config TEST AUC: {test:.4f}   (val was {best_va:.4f}; "
          f"gap>~0.002 hints at validation overfitting)")
    print("=" * 60)
