"""
higgs_predict.py  --  load a SAVED ensemble and use it (no retraining, no notebook)
====================================================================
`higgs_hpo_ensemble.py` writes a bundle to models/higgs_<set>_ensemble/:
    manifest.json   feature set, input dim, the fitted scaler (mean/std), and each
                    member's architecture + filename + single-model test AUC
    member_NN.pt    one saved network per ensemble member (state_dict)
    test_probs.npy  the ensemble's probabilities on the test split (for exact checks)
    run_log.txt     the full training log

This script rebuilds those networks from the manifest, loads their weights, and
averages them -- so you can reproduce the score or score NEW events in seconds,
from a plain Python process. Nothing here trains.

Usage:
    python higgs_predict.py                             # models/higgs_all_ensemble (default)
    python higgs_predict.py models/higgs_low_ensemble   # a specific bundle

Programmatic (e.g. to score your own array of events):
    from higgs_predict import load_ensemble, ensemble_proba
    models, man = load_ensemble("models/higgs_all_ensemble")
    probs = ensemble_proba(models, man, X_raw)   # X_raw: (N, in_dim) RAW (unstandardised) features
--------------------------------------------------------------------
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
import higgs_dnn_example as H          # only for the HiggsMLP class + data loader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ensemble(model_dir):
    """Rebuild every member network from the manifest and load its saved weights."""
    with open(os.path.join(model_dir, "manifest.json")) as f:
        man = json.load(f)
    models = []
    for m in man["members"]:
        net = H.HiggsMLP(man["in_dim"], m["hidden"], activation=m["activation"],
                         batchnorm=m["batchnorm"], dropout=m["dropout"],
                         dropout_all_layers=m["dropout_all_layers"])
        state = torch.load(os.path.join(model_dir, m["file"]),
                           map_location=DEV, weights_only=True)
        net.load_state_dict(state)
        net.to(DEV).eval()             # eval() = dropout off, BatchNorm uses saved running stats
        models.append(net)
    return models, man


@torch.no_grad()
def ensemble_proba(models, man, X_raw):
    """Average signal probability across the ensemble.

    X_raw is RAW (unstandardised) features with `in_dim` columns; we apply the exact
    scaler that was fit at training time (stored in the manifest) before predicting.
    """
    mu = np.asarray(man["scaler_mean"], dtype=np.float32)
    sd = np.asarray(man["scaler_std"], dtype=np.float32)
    Xt = torch.tensor((np.asarray(X_raw, dtype=np.float32) - mu) / sd, device=DEV)
    acc = np.zeros(len(Xt), dtype=np.float64)
    for net in models:
        chunks = []
        for i in range(0, len(Xt), 131072):
            if DEV.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    chunks.append(torch.sigmoid(net(Xt[i:i + 131072])).float().cpu().numpy())
            else:
                chunks.append(torch.sigmoid(net(Xt[i:i + 131072])).numpy())
        acc += np.concatenate(chunks)
    return acc / len(models)           # mean probability = the ensemble prediction


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/higgs_all_ensemble"
    models, man = load_ensemble(model_dir)
    print(f"loaded {len(models)} members from {model_dir}  "
          f"(feature_set={man['feature_set']}, saved ensemble AUC={man['ensemble_test_auc']})")

    # Reproduce the score on the canonical test split (last 500k rows).
    cfg = dict(H.CFG); cfg["feature_set"] = man["feature_set"]; cfg["n_rows"] = None
    X, y = H.load_higgs(cfg)
    Xtr, ytr, Xva, yva, Xte, yte = H.make_splits(X, y)   # Xte is RAW; ensemble_proba standardises it
    probs = ensemble_proba(models, man, Xte)
    print(f"reproduced test AUC = {roc_auc_score(yte, probs):.4f}   "
          f"(manifest said {man['ensemble_test_auc']})")
