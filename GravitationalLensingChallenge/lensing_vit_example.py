"""
Minimal Vision Transformer for the Strong Lensing substructure classification task.

Reference implementation for TRANSFORMER_LEARNING_GUIDE.md. It is intentionally
small and self-contained: read it top to bottom and every piece of a Vision
Transformer (ViT) is visible in plain PyTorch, no library magic.

Task: classify a (1, 64, 64) simulated strong-lensing image into one of 3 dark
matter substructure classes:  axion  |  cdm  |  no (smooth / no substructure).

Three models are included so you can see the jump the transformer buys you:
  --model mlp      a "basic DNN": flatten the image -> two Linear layers    (the baseline)
  --model deepmlp  the same DNN, but with --layers configurable hidden layers (depth alone)
  --model vit      the same task with a minimal Vision Transformer          (the point)

Usage:
    python lensing_vit_example.py --model vit          # train the ViT (default)
    python lensing_vit_example.py --model mlp          # train the MLP baseline
    python lensing_vit_example.py --model deepmlp --layers 4   # a deeper MLP baseline
    python lensing_vit_example.py --model vit --epochs 30 --no-augment

Everything (data, both models, train loop, metrics) lives in this one file.
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# axion/cdm/no in alphabetical order -> class indices 0/1/2 (same order torchvision
# ImageFolder would use, so a model you train here lines up with the notebook's).
CLASSES = ["axion", "cdm", "no"]
DATA_ROOT = "dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Level 0 — data: load every .npy once, cache a single stacked array, put on GPU
# ---------------------------------------------------------------------------
def load_split(split):
    """Return X (N,1,64,64) float32 and y (N,) int64 for split in {'train','val'}.

    The 30k/7.5k tiny .npy files are slow to open one-by-one, so the first call
    stacks them into a single cache file; later calls load that in one read.
    """
    cache_x = os.path.join(DATA_ROOT, f"cache_{split}_x.npy")
    cache_y = os.path.join(DATA_ROOT, f"cache_{split}_y.npy")
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        return np.load(cache_x), np.load(cache_y)

    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(DATA_ROOT, split, name, "*.npy")))
        print(f"  loading {len(files):>5} {split}/{name} images...")
        for f in files:
            xs.append(np.load(f).astype(np.float32))  # each is (1, 64, 64), already in [0,1]
            ys.append(label)
    X = np.stack(xs)                       # (N, 1, 64, 64)
    y = np.asarray(ys, dtype=np.int64)     # (N,)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y


def get_data():
    """Load both splits and move them onto the GPU once (they fit in <1 GB)."""
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    print(f"train {Xtr.shape}  val {Xva.shape}  range [{Xtr.min():.2f}, {Xtr.max():.2f}]")
    to = lambda a: torch.as_tensor(a, device=DEVICE)
    return to(Xtr), to(ytr), to(Xva), to(yva)


def augment_d4(x):
    """Cheap physics-motivated augmentation: lensing is rotationally symmetric, so a
    random flip + 0/90/180/270 rotation is a label-preserving view. Runs on the GPU
    batch in place of a DataLoader transform. x: (B,1,64,64)."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])              # horizontal flip
    k = int(torch.randint(0, 4, (1,)).item())    # 0..3 quarter turns
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


# ---------------------------------------------------------------------------
# Level 1 — the "basic DNN": flatten the image and run it through an MLP
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    """Baseline. Throws away all 2D structure: 64*64=4096 pixels -> hidden -> 3 logits.
    It works, but it has to relearn 'which pixels are neighbours' from scratch."""

    def __init__(self, in_pixels=64 * 64, hidden=512, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_pixels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Same structure as TinyMLP, but the number of hidden layers is a knob instead of a
    hardcoded two. We stack `n_layers` identical Linear->BatchNorm->ReLU->Dropout blocks
    (n_layers=2 is exactly TinyMLP), then a final Linear to the 3 logits. Turn --layers up
    to see that depth alone, with no 2D structure and no attention, buys the baseline little."""

    def __init__(self, in_pixels=64 * 64, hidden=512, n_classes=3, dropout=0.3, n_layers=2):
        super().__init__()
        layers = [nn.Flatten()]
        d_in = in_pixels
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden                              # every layer after the first sees `hidden` inputs
        layers += [nn.Linear(d_in, n_classes)]         # d_in == in_pixels only if n_layers == 0
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Level 2 — the transformer, built from three small pieces
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Cut the image into non-overlapping patch_size x patch_size squares and linearly
    project each into a `dim`-vector token. A strided Conv2d does exactly this: one
    conv window per patch, no overlap. (1,64,64) --patch 8--> 8x8 = 64 tokens of size dim."""

    def __init__(self, img_size=64, patch_size=8, in_ch=1, dim=64):
        super().__init__()
        self.n_tokens = (img_size // patch_size) ** 2      # 64 patches
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, dim, 8, 8)
        return x.flatten(2).transpose(1, 2)  # (B, 64, dim)  -> a sequence of tokens


class Attention(nn.Module):
    """Multi-head self-attention: every token looks at every other token and mixes in a
    weighted sum of their values. This is the ONE thing a transformer does that the MLP
    and the CNN cannot do from layer 1: relate a pixel-patch in one corner directly to a
    patch in the opposite corner (a lensing arc can span the whole image)."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)   # produce query, key, value in one matmul
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, heads, N, N) similarity scores
        attn = attn.softmax(dim=-1)                      # each token's weights sum to 1
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    """One transformer encoder block, pre-norm style:
        x = x + Attention(LayerNorm(x))     <- tokens exchange information
        x = x + MLP(LayerNorm(x))           <- each token is processed on its own
    The residual (+ x) is what lets you stack many blocks without the signal vanishing."""

    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MinimalViT(nn.Module):
    """The whole Vision Transformer, ~5 conceptual lines:
        patchify -> prepend a learnable CLS token -> add position embeddings
                 -> N transformer blocks -> read out the CLS token -> classify.
    """

    def __init__(self, img_size=64, patch_size=8, in_ch=1, dim=64,
                 depth=4, heads=4, n_classes=3, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_ch, dim)
        n = self.patch.n_tokens
        # A CLS token is a learnable "summary" slot; after attention it has gathered a
        # global view of the image, and we classify from it. (+1 token for it.)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        # Learnable position embeddings: attention is order-blind, so we ADD a vector
        # that tells each token where it sits in the 8x8 grid.
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x)                                  # (B, 64, dim)
        cls = self.cls.expand(B, -1, -1)                   # (B, 1, dim)
        x = torch.cat([cls, x], dim=1) + self.pos          # (B, 65, dim)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                          # classify from the CLS token


# ---------------------------------------------------------------------------
# Train + evaluate (shared by both models)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, Xva, yva, batch=512):
    """Return (accuracy, macro-AUC, macro-F1) on the validation set.
    AUC is computed on softmax PROBABILITIES one-vs-rest, the challenge's metric."""
    model.eval()
    probs = []
    for i in range(0, len(Xva), batch):
        logits = model(Xva[i:i + batch])
        probs.append(logits.softmax(dim=1).float().cpu())
    p = torch.cat(probs).numpy()               # (N, 3) probabilities
    y = yva.cpu().numpy()
    preds = p.argmax(1)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, p, multi_class="ovr", average="macro")   # <-- probs, not labels
    f1 = f1_score(y, preds, average="macro")
    return acc, auc, f1


def warmup_cosine(step, total_steps, warmup_frac=0.1):
    """LR multiplier: linearly warm up, then cosine-decay to 0. This is the standard
    transformer schedule. Warmup matters a lot for from-scratch ViTs: the attention
    weights are random at the start, so a few gentle steps stop the model diverging."""
    warmup = int(total_steps * warmup_frac)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train(model, data, epochs=25, batch=256, lr=1e-3, augment=True):
    Xtr, ytr, Xva, yva = data
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = (len(Xtr) + batch - 1) // batch
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total))
    loss_fn = nn.CrossEntropyLoss()
    best_auc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xtr), device=DEVICE)   # reshuffle each epoch
        running = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            if augment:
                xb = augment_d4(xb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()                                 # step the LR every batch
            running += loss.item() * len(idx)
        acc, auc, f1 = evaluate(model, Xva, yva)
        if auc > best_auc:                               # keep the best-val checkpoint
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  loss {running/len(Xtr):.4f}  "
              f"val_acc {acc:.4f}  val_auc {auc:.4f}  val_f1 {f1:.4f}  ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)                # restore best weights before returning
    print(f"best val AUC: {best_auc:.4f}")
    return best_auc


def build_model(name, layers=2):
    if name == "mlp":
        return TinyMLP()
    if name == "deepmlp":
        return DeepMLP(n_layers=layers)
    if name == "vit":
        return MinimalViT()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "deepmlp", "vit"], default="vit")
    ap.add_argument("--layers", type=int, default=2, help="hidden layers for --model deepmlp")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"device: {DEVICE}   model: {args.model}")
    data = get_data()
    model = build_model(args.model, layers=args.layers)
    train(model, data, epochs=args.epochs, batch=args.batch, lr=args.lr,
          augment=not args.no_augment)


if __name__ == "__main__":
    main()
