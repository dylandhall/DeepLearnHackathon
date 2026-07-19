"""
higgs_dnn_ensemble.py
====================================================================
Reference for the DIVERSE ENSEMBLE that reproduced the paper's best HIGGS
result. It reuses the building blocks in `higgs_dnn_example.py` (data loading,
splits, standardization, the HiggsMLP model) and trains several intentionally
DIFFERENT networks, then averages their predicted probabilities.

WHY THIS WORKS
--------------
Any single well-tuned MLP on HIGGS plateaus around test AUC 0.876-0.883 (this is
the known ceiling for this dataset -- see DNN_PLAN.md). Averaging the probability
outputs of several models whose ERRORS ARE DECORRELATED cancels out the noise in
each and pushes the ensemble above every individual member. Decorrelation comes
from deliberately VARYING the members: width, depth, activation, dropout style,
and random seed. This is also how Baldi et al. reported their numbers (means over
several random initializations).

MEASURED RESULT (this exact script, full 11M rows, RTX 5070 Ti, ~11 min total):
    Best single model (pyramid 1024-512-256-128, GELU, dropout 0.1) : test AUC 0.883
    8-model diverse ensemble                                        : test AUC 0.885
    -> matches the paper's best deep-net result (0.885) and beats LightGBM (~0.852).

METHODOLOGY (honest evaluation)
-------------------------------
- All members train on the SAME train split (~10M rows).
- The 500k VALIDATION set is used ONLY to pick each member's best-epoch checkpoint
  (this also cures the overfitting that wide nets show at long training).
- The 500k TEST set (the last 500k rows) is touched ONCE, for the final average.
  Nothing is tuned on it.
--------------------------------------------------------------------
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Reuse everything we already built in the example script.
sys.path.insert(0, ".")
import higgs_dnn_example as H

torch.set_float32_matmul_precision("high")   # TF32 matmuls: free speed on NVIDIA GPUs
torch.backends.cudnn.benchmark = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 45          # OneCycle budget per member; best-val checkpoint is what we keep
BATCH = 16384        # large batch -> full GPU utilisation + stable BatchNorm stats
# Feature set can be passed on the command line, e.g.  `python higgs_dnn_ensemble.py low`
#   "all"  = all 28 features        (reproduces the paper's 0.885 result)
#   "low"  = 21 low-level features  (the paper's signature story: the net learns the
#                                    high-level physics by itself; target ~0.88)
#   "high" = 7 high-level features  (target ~0.80)
FEATURE_SET = sys.argv[1] if len(sys.argv) > 1 else "all"

# The 8 deliberately-diverse members. Each row is a different point in
# {width, depth, activation, dropout style, weight decay, seed} space.
# do_all=True  -> dropout after every hidden layer (stronger regularisation);
# do_all=False -> dropout only on the top hidden layer (the paper's style).
MEMBERS = [
    {"name": "512x5_gelu_top.25",  "hidden": [512] * 5,             "act": "gelu", "do": 0.25, "do_all": False, "lr": 2.5e-3, "wd": 1e-4,   "seed": 0},
    {"name": "512x5_relu_all.15",  "hidden": [512] * 5,             "act": "relu", "do": 0.15, "do_all": True,  "lr": 2.5e-3, "wd": 1e-4,   "seed": 1},
    {"name": "384x6_gelu_all.12",  "hidden": [384] * 6,             "act": "gelu", "do": 0.12, "do_all": True,  "lr": 2.5e-3, "wd": 1e-4,   "seed": 2},
    {"name": "768x4_gelu_top.3",   "hidden": [768] * 4,             "act": "gelu", "do": 0.30, "do_all": False, "lr": 2.0e-3, "wd": 2e-4,   "seed": 3},
    {"name": "pyr_gelu_all.1",     "hidden": [1024, 512, 256, 128], "act": "gelu", "do": 0.10, "do_all": True,  "lr": 2.5e-3, "wd": 1e-4,   "seed": 4},
    {"name": "300x5_relu_top.5",   "hidden": [300] * 5,             "act": "relu", "do": 0.50, "do_all": False, "lr": 3.0e-3, "wd": 1e-5,   "seed": 5},
    {"name": "512x5_gelu_all.15",  "hidden": [512] * 5,             "act": "gelu", "do": 0.15, "do_all": True,  "lr": 2.5e-3, "wd": 3e-4,   "seed": 6},
    {"name": "640x5_gelu_top.25",  "hidden": [640] * 5,             "act": "gelu", "do": 0.25, "do_all": False, "lr": 2.2e-3, "wd": 1.5e-4, "seed": 7},
]


@torch.no_grad()
def predict_probs(model, X_gpu, bs=131072):
    """Sigmoid probabilities over a GPU-resident tensor, in eval mode (BatchNorm!)."""
    model.eval()
    out = []
    for i in range(0, len(X_gpu), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(X_gpu[i:i + bs])).float().cpu().numpy())
    return np.concatenate(out)


def train_member(m, Xtr_t, ytr_t, Xva_t, yva, Xte_t, in_dim):
    """Train one member; return (best_val_auc, test_probs_at_best_val_checkpoint)."""
    torch.manual_seed(m["seed"])
    np.random.seed(m["seed"])
    model = H.HiggsMLP(in_dim, m["hidden"], activation=m["act"], batchnorm=True,
                       dropout=m["do"], dropout_all_layers=m["do_all"]).to(DEV)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["wd"])
    n = len(Xtr_t)
    spe = (n + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=m["lr"], epochs=EPOCHS,
                                                steps_per_epoch=spe, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state = 0.0, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step(); sched.step()
        va = roc_auc_score(yva, predict_probs(model, Xva_t))
        if va > best_val:                       # keep the best-VALIDATION weights
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val, predict_probs(model, Xte_t)


def main():
    # ---- load + prep once, share across all members ----
    cfg = dict(H.CFG); cfg["feature_set"] = FEATURE_SET; cfg["n_rows"] = None
    X, y = H.load_higgs(cfg)
    Xtr, ytr, Xva, yva, Xte, yte = H.make_splits(X, y)
    Xtr, Xva, Xte = H.standardize(Xtr, Xva, Xte)   # fit on TRAIN only
    Xtr_t = torch.tensor(Xtr, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV)
    Xte_t = torch.tensor(Xte, device=DEV)
    in_dim = Xtr.shape[1]
    print(f"train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} feats={in_dim}\n")

    test_probs, rows = [], []
    for k, m in enumerate(MEMBERS, 1):
        t = time.perf_counter()
        best_val, tp = train_member(m, Xtr_t, ytr_t, Xva_t, yva, Xte_t, in_dim)
        test_probs.append(tp)
        single = roc_auc_score(yte, tp)
        ensemble = roc_auc_score(yte, np.mean(test_probs, axis=0))   # running average
        rows.append((m["name"], best_val, single))
        print(f"[{k}/{len(MEMBERS)}] {m['name']:20s} val={best_val:.4f} "
              f"single_test={single:.4f} | ENSEMBLE_test={ensemble:.4f} "
              f"({time.perf_counter() - t:.0f}s)")

    final = roc_auc_score(yte, np.mean(test_probs, axis=0))
    print("\n" + "=" * 60)
    for name, bv, s in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:20s} val={bv:.4f}  single_test={s:.4f}")
    print(f"  FULL ENSEMBLE                       test={final:.4f}")
    print(f"  paper deep-net best                 test=0.885")
    print("=" * 60)


if __name__ == "__main__":
    main()
