"""Train the ISL sign classifier on the landmark cache.

  python ml/prepare_data.py    # once, builds data/cache/landmarks.npz
  python ml/train.py           # trains, evaluates, exports static/models/isl/

The model is a small temporal convolution network: fast enough to run in the browser
on every new frame. Weights are exported as raw float32 for static/js/sign-model.js.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.labels import display  # noqa: E402
from ml.features import FEAT_DIM, SEQ_LEN, frames_features, resample, trim  # noqa: E402

CACHE = ROOT / "data" / "cache" / "landmarks.npz"
EXPORT = ROOT / "static" / "models" / "isl"

# Swap the signer's left and right when mirroring (pose pairs and the two hands).
MIRROR = np.arange(49)
MIRROR[[1, 2, 3, 4, 5, 6]] = [2, 1, 4, 3, 6, 5]
MIRROR[7:28], MIRROR[28:49] = np.arange(28, 49), np.arange(7, 28)


class SignNet(nn.Module):
    def __init__(self, n_classes, width=160, drop=0.35):
        super().__init__()
        self.inp = nn.Linear(FEAT_DIM, width)
        self.blocks = nn.ModuleList()
        for k, d in ((5, 1), (5, 2), (3, 4)):
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(width, width, k, padding=d * (k // 2), dilation=d),
                "bn": nn.BatchNorm1d(width),
            }))
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(width * 2, n_classes)

    def forward(self, x):  # x: (B, T, F)
        h = F.relu(self.inp(x)).transpose(1, 2)  # (B, C, T)
        for b in self.blocks:
            h = h + F.relu(b["bn"](b["conv"](h)))
        pooled = torch.cat([h.mean(2), h.amax(2)], 1)
        return self.head(self.drop(pooled))


def augment(seq, rng):
    """Random changes a real webcam would cause. seq: (frames, 49, 2) with NaN."""
    seq = seq.astype(np.float32).copy()
    n = len(seq)
    if n > 8 and rng.random() < 0.8:  # take a random 80-100% slice of time
        keep = int(n * rng.uniform(0.8, 1.0))
        start = rng.integers(0, n - keep + 1)
        seq = seq[start:start + keep]
    if rng.random() < 0.5:  # left-handed signer
        seq = seq[:, MIRROR]
        seq[..., 0] = 1.0 - seq[..., 0]
    ang = rng.uniform(-0.17, 0.17)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=np.float32)
    scale = rng.uniform(0.85, 1.15, size=2).astype(np.float32)
    shift = rng.uniform(-0.08, 0.08, size=2).astype(np.float32)
    seq = (seq - 0.5) @ rot.T * scale + 0.5 + shift
    seq += rng.normal(0, 0.004, size=seq.shape).astype(np.float32)
    if rng.random() < 0.3:  # the tracker briefly loses a hand
        lost = rng.random(len(seq)) < 0.15
        hand = slice(7, 28) if rng.random() < 0.5 else slice(28, 49)
        seq[lost, hand] = np.nan
    return seq


def featurize(seqs, rng=None):
    out = np.empty((len(seqs), SEQ_LEN, FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = trim(np.asarray(s, dtype=np.float32))
        if rng is not None:
            s = augment(s, rng)
        out[i] = frames_features(resample(s))
    return out


def split(labels, val_frac, seed):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        k = max(1, int(round(len(idx) * val_frac))) if len(idx) >= 4 else 0
        va += idx[:k].tolist()
        tr += idx[k:].tolist()
    return np.array(tr), np.array(va)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    top = logits.topk(5, 1).indices.numpy()
    return float((top[:, 0] == y).mean()), float((top == y[:, None]).any(1).mean()), logits.softmax(1).numpy()


def export(model, mean, std, classes, categories, metrics):
    """Fold BatchNorm into the convolutions and write raw float32 weights + a JSON manifest."""
    EXPORT.mkdir(parents=True, exist_ok=True)
    model.eval()
    tensors, layout = [], []

    def add(name, t):
        a = t.detach().cpu().numpy().astype(np.float32)
        layout.append({"name": name, "shape": list(a.shape), "offset": sum(x.size for x in tensors)})
        tensors.append(a.reshape(-1))

    add("mean", torch.from_numpy(mean))
    add("std", torch.from_numpy(std))
    add("inp.w", model.inp.weight)  # (C, F)
    add("inp.b", model.inp.bias)
    for i, b in enumerate(model.blocks):
        conv, bn = b["conv"], b["bn"]
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        add(f"b{i}.w", conv.weight * g[:, None, None])  # (C, C, k)
        add(f"b{i}.b", (conv.bias - bn.running_mean) * g + bn.bias)
    add("head.w", model.head.weight)  # (classes, 2C)
    add("head.b", model.head.bias)

    blob = np.concatenate(tensors)
    (EXPORT / "weights.bin").write_bytes(blob.tobytes())
    manifest = {
        "version": time.strftime("%Y%m%d-%H%M"),
        "seq_len": SEQ_LEN, "feat_dim": FEAT_DIM,
        "blocks": [{"kernel": b["conv"].kernel_size[0], "dilation": b["conv"].dilation[0]} for b in model.blocks],
        "tensors": layout,
        "classes": classes, "labels": [display(c) for c in classes], "categories": categories,
        "metrics": metrics,
    }
    (EXPORT / "model.json").write_text(json.dumps(manifest))
    print(f"exported {blob.size * 4 / 1e6:.2f} MB to {EXPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--full", action="store_true", help="train on all data after validating (for export)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = np.load(CACHE, allow_pickle=True)
    seqs, labels = d["seqs"], d["labels"]
    classes, categories = d["classes"].tolist(), d["categories"].tolist()
    tr, va = split(labels, args.val, args.seed)
    if args.full:
        tr = np.arange(len(labels))
    print(f"train {len(tr)}  val {len(va)}  classes {len(classes)}")

    base = featurize(seqs[tr])
    mean = base.reshape(-1, FEAT_DIM).mean(0)
    std = base.reshape(-1, FEAT_DIM).std(0) + 1e-4
    Xva = (featurize(seqs[va]) - mean) / std
    yva = labels[va]

    model = SignNet(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(tr) / 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * steps_per_epoch)
    rng = np.random.default_rng(args.seed)
    best = (0.0, None)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        X = (featurize(seqs[tr], rng) - mean) / std
        y = labels[tr]
        perm = rng.permutation(len(tr))
        total = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            msg = f"epoch {epoch + 1:3d}  loss {total / len(tr):.3f}"
            if len(va) and not args.full:
                top1, top5, _ = evaluate(model, Xva, yva)
                msg += f"  val top1 {top1:.3f}  top5 {top5:.3f}"
                if top1 >= best[0]:
                    best = (top1, {k: v.clone() for k, v in model.state_dict().items()})
            print(msg, f"({time.time() - t0:.0f}s)", flush=True)

    if best[1] is not None:
        model.load_state_dict(best[1])
    metrics = {"train_samples": int(len(tr)), "classes": len(classes)}
    if len(va):
        top1, top5, probs = evaluate(model, Xva, yva)
        conf = probs.max(1)
        right = probs.argmax(1) == yva
        print(f"final val top1 {top1:.3f}  top5 {top5:.3f}")
        for th in (0.3, 0.5, 0.7):
            m = conf >= th
            print(f"  confidence >= {th}: answers {m.mean():.2f} of signs, correct {right[m].mean() if m.any() else 0:.3f}")
        if not args.full:
            metrics.update({"val_top1": round(top1, 4), "val_top5": round(top5, 4), "val_samples": int(len(va))})
    export(model, mean, std, classes, categories, metrics)


if __name__ == "__main__":
    main()
