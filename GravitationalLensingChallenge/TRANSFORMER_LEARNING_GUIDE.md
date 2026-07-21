# Learning Guide: a basic image DNN, then a minimal Vision Transformer

This guide is for **understanding and implementing each piece yourself**. Goal: classify a
`(1, 64, 64)` strong-lensing image into one of three dark-matter substructure classes —
`axion`, `cdm`, `no` (smooth) — first with a plain "flatten → MLP" network, then by swapping in
a **minimal Vision Transformer (ViT)** so you see exactly what the transformer adds and where it
sits in the pipeline.

The complete, runnable reference lives in [`lensing_vit_example.py`](./lensing_vit_example.py).
**Try each level yourself first, then open the reference to check your work.** This is the
*minimum* transformer — the full ViT/Swin/MAE roadmap is [`PLAN.md`](./PLAN.md), which this guide
is the on-ramp to. Don't jump ahead; get this working and understood first.

```
Level 0  Data        load 30k .npy → cache → GPU tensors                 (shared by both models)
Level 1  Basic DNN   flatten 64×64 → MLP → 3 logits                       → val AUC ~0.989  (baseline)
Level 2  Patchify    image → sequence of patch tokens                     (the ViT's first layer)
Level 3  Attention   let every token look at every other token           (the ONE new idea)
Level 4  Assemble    CLS token + positions + N blocks + head              → the Vision Transformer
Level 5  Train it    warmup + cosine LR (ViTs are fragile from scratch)   → val AUC ~0.966
```

> **Honest headline up front:** on *this* dataset the 2.4M-param MLP (**0.989**) actually beats the
> 142k-param from-scratch ViT (**0.966**). That's expected and it's the real lesson — see
> "Honest expectations" at the end. You're implementing the transformer to *understand* it and to
> have the foundation the PLAN.md upgrades (more capacity, pretraining) build on, not because a tiny
> ViT wins on 30k easy images.

---

## Concepts you'll use (transformer-specific primer)

You already know MLPs, AUC, train/val, dropout, BatchNorm, the AdamW training loop, and mixed
precision from the Higgs work. A ViT reuses **all** of that. The genuinely new ideas are only
these five:

- **Token / patch.** A transformer doesn't eat a grid of pixels; it eats a **sequence of vectors**
  ("tokens"). For images we make tokens by cutting the picture into small squares (**patches**),
  e.g. 8×8, and linearly projecting each patch into a `dim`-length vector. A 64×64 image at patch
  size 8 → an 8×8 grid → **64 tokens**.
- **Self-attention.** The core operation. Each token computes a **query**, and every token a
  **key** and **value**. A token's new representation is a weighted average of *all* tokens' values,
  weighted by how well its query matches each key. Plain English: **every patch can look directly at
  every other patch, from the very first layer.** A CNN only sees a small neighbourhood at a time; a
  lensing arc can span the whole image, so "look everywhere at once" is the point.
- **CLS token.** A single extra, learnable "summary" vector we prepend to the sequence. After the
  attention layers it has gathered a global read of the image, and we attach the classifier to it.
  (Averaging all tokens — "global average pooling" — is a common alternative.)
- **Positional embedding.** Attention is **order-blind** — permute the tokens and it can't tell.
  So we *add* a learnable vector to each token encoding *where* it sits in the 8×8 grid. Without
  this the model literally cannot use spatial layout.
- **Pre-norm residual block + warmup.** Each transformer block is `x = x + Attn(LayerNorm(x))` then
  `x = x + MLP(LayerNorm(x))`. The `x +` (residual) lets you stack many blocks; LayerNorm (not
  BatchNorm) keeps each token stable. From-scratch ViTs are **touchy at the start**, so the learning
  rate is **warmed up** from ~0 over the first few % of steps — skip this and it often diverges.

That's the whole vocabulary. Everything else is the training loop you already know.

---

## Level 0 — The data foundation (shared by both models)

**Concept:** get the images into `(N, 1, 64, 64)` float32 tensors on the GPU before any model.

1. The dataset is 30,000 train + 7,500 val tiny `.npy` files under `dataset/{train,val}/{axion,cdm,no}/`.
   Each is one `(1, 64, 64)` float64 image, **already min-max normalized to `[0, 1]`** — so unlike
   Higgs, you do **not** need to standardize. (Confirm the range prints as `[0.00, 1.00]`.)
2. Opening 37,500 files one-by-one is slow, so **stack them into a single cache `.npy` on first run**;
   later runs load in one read. (Same "cache once" trick as Higgs' `.npy` cache.)
3. **Label mapping:** classes sorted alphabetically → `axion=0, cdm=1, no=2` (the order torchvision's
   `ImageFolder` uses, so a model here lines up with the starter notebook).
4. **Move both splits onto the GPU once** — they're <1 GB total, so every batch is a fast index, no
   per-batch CPU→GPU copy (identical reasoning to Higgs).

**Key code:**

```python
def load_split(split):                       # split in {"train","val"}
    xs, ys = [], []
    for label, name in enumerate(["axion","cdm","no"]):
        for f in sorted(glob.glob(f"dataset/{split}/{name}/*.npy")):
            xs.append(np.load(f).astype(np.float32))   # (1,64,64), already in [0,1]
            ys.append(label)
    return np.stack(xs), np.asarray(ys, np.int64)      # (N,1,64,64), (N,)
# ...then cache to disk, and: Xtr = torch.as_tensor(Xtr, device="cuda")  (move once)
```

**One augmentation worth having (physics-motivated).** Gravitational lensing is rotationally
symmetric, so a random flip + 90°/180°/270° rotation is a *label-preserving* new view (the "D4"
dihedral group from PLAN.md). It's ~4 lines and does it on the GPU batch:

```python
def augment_d4(x):                            # x: (B,1,64,64)
    if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])   # horizontal flip
    k = int(torch.randint(0, 4, (1,)).item())                    # 0..3 quarter turns
    return torch.rot90(x, k, dims=[2, 3]) if k else x
```

**Your turn:** write `load_split`, print the six shapes, and confirm the pixel range is `[0, 1]` and
each class has 10,000 train / 2,500 val images. Reference: `load_split`, `get_data`, `augment_d4`.

---

## Level 1 — The basic DNN (your baseline)

**Concept:** the simplest thing that works — **flatten** the 64×64 image to a 4096-vector and run it
through a 2-hidden-layer MLP, exactly the block you used on Higgs (`Linear → BatchNorm → ReLU →
Dropout`) ending in **3 logits** (one per class).

**Why start here:** it's a strong, honest baseline in ~10 lines, it wires up the whole train/eval
harness you'll reuse for the ViT, and it makes the transformer's value measurable. Its weakness is
conceptual: flattening **throws away all 2D structure** — the net has to relearn which pixels are
neighbours from scratch.

**Key code — the model:**

```python
class TinyMLP(nn.Module):
    def __init__(self, in_pixels=64*64, hidden=512, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_pixels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),    nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),                 # 3 logits, NO softmax here
        )
    def forward(self, x): return self.net(x)
```

**Key code — loss and metric (3-class, not binary):**

```python
loss_fn = nn.CrossEntropyLoss()               # expects raw logits + integer labels (0/1/2)
# eval: AUC on PROBABILITIES, one-vs-rest, the challenge's metric:
p = model(Xva).softmax(dim=1).cpu().numpy()   # (N,3) probabilities
auc = roc_auc_score(yva.cpu(), p, multi_class="ovr", average="macro")   # <-- probs, not argmax
```

> ⚠️ Same classic bug as Higgs, multi-class version: feed `roc_auc_score` the **softmax
> probabilities**, never the `argmax` class labels. Labels give a meaningless AUC.

**Your turn:** write `TinyMLP`, reuse a training loop shaped like your Higgs one (AdamW,
`CrossEntropyLoss`, per-epoch val), run 25 epochs. You should reach **val accuracy ~0.93, val AUC
~0.989**. If you're stuck near 0.33 (chance for 3 balanced classes) something is unwired — check the
labels are `0/1/2` integers and the loss is `CrossEntropyLoss`. Reference: `TinyMLP`, `train`,
`evaluate`.

---

## Level 2 — Patchify: turn the image into a sequence

**Concept:** the transformer's *only* image-specific layer. Cut the 64×64 image into non-overlapping
**8×8 patches** (an 8×8 grid = **64 patches**) and linearly project each into a `dim`-vector token.
Output: `(B, 64, dim)` — a **sequence of 64 tokens**, which is what attention consumes.

**The elegant trick:** a strided convolution *is* "one linear projection per non-overlapping patch."
A `Conv2d(1, dim, kernel_size=8, stride=8)` places exactly one window per patch with no overlap, so
it patchifies **and** projects in a single layer:

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_ch=1, dim=64):
        super().__init__()
        self.n_tokens = (img_size // patch_size) ** 2          # 64
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):                    # x: (B,1,64,64)
        x = self.proj(x)                     # (B, dim, 8, 8)
        return x.flatten(2).transpose(1, 2)  # (B, 64, dim)  → sequence of tokens
```

**Why patch size 8?** It's the knob that trades detail vs cost: smaller patches = more, finer tokens
(better for tiny subhalo features, more compute); larger = fewer, coarser tokens. 8 gives a
comfortable 64 tokens on a 64×64 image. (PLAN.md's Pipeline A recommends 8×8 for exactly this
"fine-grained substructure" reason.)

**Your turn:** build `PatchEmbed`, push one batch through it, and confirm the output shape is
`(B, 64, dim)`. Sanity: `n_tokens` should equal `(64/8)**2 = 64`. Reference: `PatchEmbed`.

---

## Level 3 — Self-attention: the one genuinely new operation

**Concept:** this is what makes a transformer a transformer. From the token sequence, produce three
projections — **Q**uery, **K**ey, **V**alue — then each token's output is a **softmax-weighted sum of
every token's value**, weighted by query·key similarity. "Multi-head" just runs several of these in
parallel on slices of the vector so different heads can attend to different things.

**Why it matters here:** unlike a CNN (local windows) or the MLP (no spatial notion at all), attention
relates **any** patch to **any** other patch in a single layer — a lensing arc spanning opposite
corners is one hop away, not many.

**Key code (write this one by hand — it's the heart of the model):**

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)   # Q, K, V in one matmul
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):                                   # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, heads, N, N) similarity
        attn = attn.softmax(dim=-1)                         # each row sums to 1
        out  = (attn @ v).transpose(1, 2).reshape(B, N, D)  # weighted sum of values
        return self.proj(out)
```

The `* self.scale` (= 1/√head_dim) keeps the dot-products from blowing up as `dim` grows — without it
the softmax saturates and gradients vanish. (In production you'd call
`F.scaled_dot_product_attention`, which fuses this and is faster; the explicit form above is so you
can *see* it.)

**Your turn:** implement `Attention`, feed it a `(2, 64, 64)` tensor, and confirm it returns the same
shape and that each attention row sums to 1 (`attn.sum(-1)` ≈ all ones). Reference: `Attention`.

---

## Level 4 — Assemble the Vision Transformer

**Concept:** stack the pieces. A **Block** is the pre-norm residual unit; the **ViT** is patchify →
prepend CLS → add positions → N blocks → read out the CLS token → classify.

**Key code — the block:**

```python
class Block(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # tokens exchange info
        x = x + self.mlp(self.norm2(x))      # each token processed on its own
        return x
```

**Key code — the full model:**

```python
class MinimalViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_ch=1, dim=64, depth=4, heads=4, n_classes=3):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_ch, dim)
        n = self.patch.n_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))          # learnable summary token
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))      # learnable positions (+1 for CLS)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch(x)                                        # (B, 64, dim)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos   # (B, 65, dim)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x)[:, 0])                     # classify from CLS token (index 0)
```

Two things people forget: **`+ self.pos`** (without positions the model is spatially blind) and
initializing `cls`/`pos` small (`trunc_normal_(std=0.02)`) so early training is stable.

**Config for "minimal":** `dim=64, depth=4, heads=4, patch=8` → ~142k parameters. Deliberately tiny;
the PLAN.md endgame is `dim=768, depth=12` (ViT-Base) with pretraining.

**Your turn:** assemble `Block` and `MinimalViT`, run one batch, confirm output shape `(B, 3)` and
print the param count (~142k). Reference: `Block`, `MinimalViT`.

---

## Level 5 — Train it right (the part that actually bites)

**Concept:** the model is the easy half; a from-scratch ViT is **fragile to train** and this is where
you'll spend your debugging time. The reference `train()` is the Higgs loop with **two ViT-specific
changes** that matter enormously:

**1. Warmup + cosine learning-rate schedule (not OneCycle).**

```python
def warmup_cosine(step, total, warmup_frac=0.1):
    warmup = int(total * warmup_frac)
    if step < warmup: return step / max(1, warmup)                 # ramp 0 → 1
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))                       # cosine 1 → 0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, epochs * steps))
```

**2. Keep the best-validation checkpoint** and restore it at the end (val AUC wobbles epoch-to-epoch
early on, so the last epoch isn't always the best):

```python
if auc > best_auc: best_auc, best_state = auc, {k: v.cpu().clone() for k,v in model.state_dict().items()}
# ...after the loop: model.load_state_dict(best_state)
```

**A real lesson from building this — the learning rate is not optional trivia.** With
`OneCycleLR` at `max_lr=3e-4` (a fine default for the MLP) the ViT **got stuck**: train loss fell but
val accuracy thrashed between 0.36 and 0.55 and AUC capped at ~0.90. The *same model* at **`lr=1e-3`
with warmup+cosine** trains smoothly to **0.966**. Same architecture, only the optimization changed.
Two takeaways: (a) tiny ViTs often want a *higher* peak LR than you'd guess, and (b) the warmup is
what buys the stability. If your ViT looks broken, suspect the LR schedule before the architecture.

**Rest of the loop is identical to Higgs:** AdamW (`weight_decay=0.05`), `CrossEntropyLoss`,
`torch.autocast(bfloat16)`, reshuffle each epoch, `augment_d4` on each train batch, evaluate on val
each epoch.

**Your turn:** wire up `warmup_cosine` + best-checkpoint into your loop and run:

```bash
python lensing_vit_example.py --model vit          # 25 epochs, lr 1e-3, warmup+cosine
```

You should watch val AUC climb smoothly to **~0.966** (acc ~0.875) and stay stable — no thrashing. If
it thrashes or caps near 0.90, your LR/warmup is off (that's the exact bug above). Reference:
`warmup_cosine`, `train`.

---

## Results & honest expectations

Measured on the 7,500-image val set, seed 0, identical settings (25 epochs, lr 1e-3, warmup+cosine,
D4 augmentation):

| Model | Params | val accuracy | val macro-AUC | epoch time |
|---|---|---|---|---|
| `TinyMLP` (basic DNN) | 2.36 M | **~0.93** | **~0.989** | ~0.1 s |
| `MinimalViT` | 0.14 M | ~0.875 | ~0.966 | ~0.6 s |

**Yes — the plain MLP beats the from-scratch minimal ViT here, and that's the honest, expected
result.** Why, and why it's still worth doing:

- **Transformers are data- and scale-hungry.** They have almost no built-in image bias (a CNN knows
  "nearby pixels relate"; a ViT must *learn* that). On 30k relatively easy 64×64 images a well-tuned
  MLP or CNN is a very strong baseline, and a **142k-param** ViT — **16× smaller** than the MLP — is
  competitive but doesn't win. That's not the transformer failing; it's this regime not yet playing
  to its strengths.
- **Where the ViT actually pulls ahead** is exactly the PLAN.md roadmap: more capacity
  (`dim=192+`, `depth=8-12`), **self-supervised pretraining** (MAE — learn the physics from unlabeled
  images first, then fine-tune), and harder/larger data where "look everywhere at once" beats local
  filters. This minimal ViT is the *foundation* those upgrades bolt onto, not the finished model.
- **Don't over-read one number.** These are single-seed val results and the metrics wobble a little
  run-to-run; treat them as "~0.99 vs ~0.97", not a precise ranking.

So the deliverable of this guide is **understanding** — you can now point at every tensor in a ViT and
say what it's for — plus a clean, runnable baseline pair to measure future upgrades against.

---

## How to know each level "worked"

| Level | Sanity check |
|---|---|
| 0 | Six split shapes print; pixel range `[0,1]`; 10k/2.5k per class |
| 1 | MLP val AUC ~0.989; **stuck at ~0.33 = a wiring bug** (labels or loss) |
| 2 | `PatchEmbed` output is `(B, 64, dim)`; `n_tokens == 64` |
| 3 | `Attention` returns input shape; each attention row sums to 1 |
| 4 | `MinimalViT` outputs `(B, 3)`; ~142k params |
| 5 | ViT val AUC climbs **smoothly** to ~0.966; thrashing near 0.90 = LR/warmup bug |

---

## Where to go next (into PLAN.md)

Once this is understood and running, the highest-value next steps — smallest change first — are:

1. **Scale the ViT you already have:** bump `dim` to 128–192 and `depth` to 6–8. One-line change,
   see how far it closes the gap to the MLP.
2. **Try global-average-pooling instead of the CLS token** (average all tokens before the head).
   PLAN.md notes GAP can regularize better on noisy symmetric data — a small experiment.
3. **CNN stem tokenizer:** replace the single patch conv with a shallow 2–3 layer conv stem before
   patchifying — adds a little locality bias and often helps small-data ViTs a lot.
4. **MAE self-supervised pretraining** (PLAN.md Pipeline A): the big one. Pretrain the encoder to
   reconstruct masked patches on all images (labels not needed), then fine-tune the classifier. This
   is where transformers typically overtake the baseline.

Do them one at a time, each measured against the two baselines in the table above. That's the whole
method: small, honest deltas — same discipline as the Higgs levels.
