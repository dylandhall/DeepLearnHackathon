"""
higgs_hpo_ensemble.py  --  the capstone pipeline
====================================================================
Ties the whole thing together and answers one question:
    does tuning (Optuna) + ensembling genuinely clear the paper's 0.885?

Pipeline:
    Phase 1  HPO         reuse/extend the Optuna study in higgs_optuna.db
    Phase 2  Select      pick the top-K DIVERSE configs (distinct architectures,
                         best validation AUC first)
    Phase 3  Retrain     train each picked config with several SEEDS, keeping the
                         best-VALIDATION checkpoint of each
    Phase 4  Ensemble    average all members' probabilities; score ONCE on the
                         untouched TEST set

Why "diverse configs x seeds": ensembles gain from DECORRELATED errors. Different
architectures decorrelate more than different seeds of one architecture, so we mix
both. Selection uses validation only; the test set is touched once at the very end.

Run:
    python higgs_hpo_ensemble.py                    # all 28 features
    python higgs_hpo_ensemble.py low                # 21 low-level features
    python higgs_hpo_ensemble.py all 80 6 2         # feature_set, target_trials, top_k, seeds
--------------------------------------------------------------------
"""
import os
import sys
import json
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
import higgs_dnn_example as H

optuna.logging.set_verbosity(optuna.logging.WARNING)   # quieter per-trial logs
torch.set_float32_matmul_precision("high")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_SET   = sys.argv[1] if len(sys.argv) > 1 else "all"
TARGET_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 60   # total HPO trials (reuses existing)
TOP_K         = int(sys.argv[3]) if len(sys.argv) > 3 else 6    # distinct configs to ensemble
SEEDS         = int(sys.argv[4]) if len(sys.argv) > 4 else 2    # seeds per config
EPOCHS_HPO, EPOCHS_FINAL, BATCHES = 30, 35, [8192, 16384, 32768]
SAVE_DIR = os.path.join("models", f"higgs_{FEATURE_SET}_ensemble")   # models + scaler + manifest go here

# ---- data: load once, resident on GPU, shared by every trial & member ----
cfg = dict(H.CFG); cfg["feature_set"] = FEATURE_SET; cfg["n_rows"] = None
X, y = H.load_higgs(cfg)
Xtr, ytr, Xva, yva, Xte, yte = H.make_splits(X, y)
# standardize on TRAIN stats; keep mu/sd so we can save them with the models (a loaded
# model is useless without the exact scaler that was fit at training time).
mu = Xtr.mean(axis=0, keepdims=True); sd = Xtr.std(axis=0, keepdims=True) + 1e-8
Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
Xva_t = torch.tensor(Xva, device=DEV); Xte_t = torch.tensor(Xte, device=DEV)
IN = Xtr.shape[1]
print(f"feature_set={FEATURE_SET} in_dim={IN} target_trials={TARGET_TRIALS} "
      f"top_k={TOP_K} seeds={SEEDS}\n", flush=True)


def build_hidden(p):
    """Reconstruct the layer widths from a params dict (uniform or pyramid)."""
    if p["shape"] == "uniform":
        return [p["base_width"]] * p["depth"]
    return [max(64, p["base_width"] // (2 ** i)) for i in range(p["depth"])]


@torch.no_grad()
def predict(model, Xg):
    model.eval()
    out = []
    for i in range(0, len(Xg), 131072):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(Xg[i:i + 131072])).float().cpu().numpy())
    return np.concatenate(out)


def run_config(p, seed, epochs, trial=None, want_test=False):
    """Train one config; keep best-VAL checkpoint. Returns (best_val, test_probs_or_None)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = H.HiggsMLP(IN, build_hidden(p), activation=p["act"], batchnorm=True,
                       dropout=p["dropout"], dropout_all_layers=p["dropout_all_layers"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=p["max_lr"], weight_decay=p["weight_decay"])
    n = len(Xtr_t); spe = (n + p["batch"] - 1) // p["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["max_lr"], epochs=epochs,
                                                steps_per_epoch=spe, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, p["batch"]):
            idx = perm[i:i + p["batch"]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step(); sched.step()
        va = roc_auc_score(yva, predict(model, Xva_t))
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:                      # pruning during HPO
            trial.report(va, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    test_probs = None
    if want_test:
        model.load_state_dict(best_state)
        test_probs = predict(model, Xte_t)
    return best_val, test_probs, best_state       # best_state = CPU weights of the best-val epoch


def objective(trial):
    p = {
        "depth": trial.suggest_int("depth", 3, 6),
        "base_width": trial.suggest_categorical("base_width", [256, 384, 512, 640, 768, 1024]),
        "shape": trial.suggest_categorical("shape", ["uniform", "pyramid"]),
        "act": trial.suggest_categorical("act", ["relu", "gelu"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "dropout_all_layers": trial.suggest_categorical("dropout_all_layers", [True, False]),
        "max_lr": trial.suggest_float("max_lr", 1e-3, 4e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", BATCHES),
    }
    return run_config(p, seed=0, epochs=EPOCHS_HPO, trial=trial)[0]


# ==== Phase 1: HPO (reuse the persistent study, top up to TARGET_TRIALS) ====
study = optuna.create_study(
    direction="maximize", study_name=f"higgs_{FEATURE_SET}",
    storage="sqlite:///higgs_optuna.db", load_if_exists=True,
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
)
done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"[Phase 1] study has {done} completed trials", flush=True)
if done < TARGET_TRIALS:
    print(f"[Phase 1] running {TARGET_TRIALS - done} more trials...", flush=True)
    study.optimize(objective, n_trials=TARGET_TRIALS - done)
print(f"[Phase 1] best val AUC so far: {study.best_value:.4f}", flush=True)

# ==== Phase 2: pick TOP_K DIVERSE configs (distinct architecture signatures) ====
completed = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                   key=lambda t: t.value, reverse=True)
picked, seen_sig = [], set()
for t in completed:
    p = t.params
    sig = (p["depth"], p["base_width"], p["shape"], p["act"], p["dropout_all_layers"])
    if sig in seen_sig:                     # keep only distinct architectures -> more decorrelated
        continue
    seen_sig.add(sig); picked.append(t)
    if len(picked) == TOP_K:
        break
print(f"\n[Phase 2] selected {len(picked)} diverse configs (by val AUC):", flush=True)
for t in picked:
    p = t.params
    print(f"    val={t.value:.4f}  {build_hidden(p)} {p['act']} "
          f"do={p['dropout']:.2f}/all={p['dropout_all_layers']} lr={p['max_lr']:.1e}", flush=True)

# ==== Phase 3: retrain picked configs x SEEDS, collect TEST probabilities ====
print(f"\n[Phase 3] retraining {len(picked)} configs x {SEEDS} seeds = "
      f"{len(picked)*SEEDS} members  ->  saving to {SAVE_DIR}/", flush=True)
os.makedirs(SAVE_DIR, exist_ok=True)
member_probs, member_meta, best_single = [], [], 0.0
for t in picked:
    hidden = build_hidden(t.params)
    for s in range(SEEDS):
        _, tp, state = run_config(t.params, seed=s, epochs=EPOCHS_FINAL, want_test=True)
        fname = f"member_{len(member_meta):02d}.pt"
        torch.save(state, os.path.join(SAVE_DIR, fname))         # <-- persist this member's weights
        member_probs.append(tp)
        single = float(roc_auc_score(yte, tp)); best_single = max(best_single, single)
        member_meta.append({                                     # everything needed to rebuild it
            "file": fname, "hidden": hidden, "activation": t.params["act"],
            "batchnorm": True, "dropout": float(t.params["dropout"]),
            "dropout_all_layers": bool(t.params["dropout_all_layers"]),
            "seed": s, "single_test_auc": round(single, 4),
        })
        ens = float(roc_auc_score(yte, np.mean(member_probs, axis=0)))
        print(f"    member {len(member_meta):2d}: single_test={single:.4f} | "
              f"ENSEMBLE_test={ens:.4f}", flush=True)

# ==== Phase 4: save the bundle + report ====
final = float(roc_auc_score(yte, np.mean(member_probs, axis=0)))
np.save(os.path.join(SAVE_DIR, "test_probs.npy"), np.mean(member_probs, axis=0))   # exact-repro cache
manifest = {                                    # the scaler + arch metadata a loader needs
    "feature_set": FEATURE_SET, "in_dim": int(IN),
    "scaler_mean": mu.ravel().astype(float).tolist(),
    "scaler_std": sd.ravel().astype(float).tolist(),
    "epochs_final": EPOCHS_FINAL,
    "best_single_test_auc": round(float(best_single), 4),
    "ensemble_test_auc": round(final, 4),
    "members": member_meta,
}
with open(os.path.join(SAVE_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 64, flush=True)
print(f"HPO+ENSEMBLE RESULT  (feature_set={FEATURE_SET})", flush=True)
print(f"  best single tuned model   test={best_single:.4f}", flush=True)
print(f"  full ensemble ({len(member_probs)} members) test={final:.4f}", flush=True)
print(f"  paper deep-net best        test=0.885", flush=True)
print(f"  -> {'CLEARS' if final > 0.885 else 'does NOT clear'} 0.885", flush=True)
print(f"\n  saved {len(member_meta)} models + manifest.json + test_probs.npy to {SAVE_DIR}/", flush=True)
print(f"  reload & reproduce (no retraining):  python higgs_predict.py {SAVE_DIR}", flush=True)
print("=" * 64, flush=True)
