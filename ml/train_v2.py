"""v2 sign models: 127 points (hands, arms, lips, eyes, nose), CNN + transformer, GPU training.

  python ml/prepare_v2.py --lang asl
  python ml/train_v2.py --lang asl --epochs 200

The whole dataset sits on the GPU and every augmentation runs there, so a laptop RTX GPU
trains ASL (94k recordings) in a few hours. Features are computed inside the network from raw
landmarks + a presence mask, so the browser only has to pass points (static/js/sign-model-v2.js).
Exports static/models/<lang>/model.onnx + model.json, and keeps the v1 files untouched until
--install is given.
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
from ml import landmarks_v2 as L  # noqa: E402
from ml.labels import display  # noqa: E402
from ml.train import LANGS, split  # noqa: E402

T_IN, T = 48, 32                    # frames stored / frames the model sees
P = L.N_POINTS
FEAT = P * 2 + 84 + 80 + P * 2 + 2  # global, hand shapes, lip shape, motion, hand flags = 674


# ---------------------------------------------------------------- features (inside the model)
def features(x, m):
    """x: (B, T, 127, 2) raw image coords (0 where missing), m: (B, T, 127) 1 = present."""
    eps = 1e-4
    ls, rs = x[:, :, 1], x[:, :, 2]
    pres = (m[:, :, 1] * m[:, :, 2]).unsqueeze(-1)                       # (B, T, 1)
    c_f, s_f = (ls + rs) / 2, (ls - rs).norm(dim=-1, keepdim=True)
    n = pres.sum(1, keepdim=True)
    c_seq = (c_f * pres).sum(1, keepdim=True) / n.clamp(min=1)
    s_seq = (s_f * pres).sum(1, keepdim=True) / n.clamp(min=1)
    has = (n > 0).float()
    c_seq = has * c_seq + (1 - has) * 0.5
    s_seq = has * s_seq + (1 - has) * 0.25
    c = pres * c_f + (1 - pres) * c_seq
    s = (pres * s_f + (1 - pres) * s_seq).clamp(min=1e-3)
    mm = m.unsqueeze(-1)
    g = (x - c.unsqueeze(2)) / s.unsqueeze(2) * mm                       # (B, T, P, 2)

    def local(block, ref_a, ref_b, mask_pt):
        a, b = block[:, :, ref_a:ref_a + 1], block[:, :, ref_b:ref_b + 1]
        size = (b - a).norm(dim=-1, keepdim=True).clamp(min=eps)
        return (block - a) / size * mask_pt

    lh, rh = x[:, :, L.SL_LH], x[:, :, L.SL_RH]
    lh_on, rh_on = m[:, :, 9:10].unsqueeze(-1), m[:, :, 30:31].unsqueeze(-1)
    lhl = local(lh, 0, 9, lh_on * m[:, :, L.SL_LH].unsqueeze(-1))
    rhl = local(rh, 0, 9, rh_on * m[:, :, L.SL_RH].unsqueeze(-1))
    lips = x[:, :, L.SL_LIPS]
    lm = m[:, :, L.SL_LIPS].unsqueeze(-1)
    lc = (lips * lm).sum(2, keepdim=True) / lm.sum(2, keepdim=True).clamp(min=1)
    lw = (lips[:, :, 0:1] - lips[:, :, 10:11]).norm(dim=-1, keepdim=True).clamp(min=eps)
    lipl = (lips - lc) / lw * lm

    vel = torch.zeros_like(g)
    both = mm[:, 1:] * mm[:, :-1]
    vel[:, 1:] = (g[:, 1:] - g[:, :-1]) * both
    flags = torch.cat([m[:, :, 9:10], m[:, :, 30:31]], -1)
    B, Tn = x.shape[:2]
    f = torch.cat([g.reshape(B, Tn, -1), lhl.reshape(B, Tn, -1), rhl.reshape(B, Tn, -1),
                   lipl.reshape(B, Tn, -1), vel.reshape(B, Tn, -1) * 4, flags], -1)
    return f.clamp(-10, 10)


# ---------------------------------------------------------------- model
class ECA(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):                     # x: (B, C, T)
        w = self.conv(x.mean(-1, keepdim=True).transpose(1, 2)).transpose(1, 2)
        return x * torch.sigmoid(w)


class ConvBlock(nn.Module):
    def __init__(self, d, k=15, drop=0.2, expand=2):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.inp = nn.Linear(d, d * expand)
        self.dw = nn.Conv1d(d * expand, d * expand, k, padding=k // 2, groups=d * expand)
        self.bn = nn.BatchNorm1d(d * expand)
        self.eca = ECA()
        self.out = nn.Linear(d * expand, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                     # (B, T, D)
        h = F.silu(self.inp(self.ln(x))).transpose(1, 2)
        h = self.eca(F.silu(self.bn(self.dw(h)))).transpose(1, 2)
        return x + self.drop(self.out(h))


class TransBlock(nn.Module):
    def __init__(self, d, heads=4, drop=0.2):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Dropout(drop), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.att(h, h, h, need_weights=False)[0])
        return x + self.drop(self.mlp(self.ln2(x)))


class SignNetV2(nn.Module):
    def __init__(self, n_classes, d=192, repeats=2, drop=0.2, head_drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(FEAT, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        blocks = []
        for _ in range(repeats):
            blocks += [ConvBlock(d, drop=drop), ConvBlock(d, drop=drop), ConvBlock(d, drop=drop), TransBlock(d, drop=drop)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(head_drop)
        self.head = nn.Linear(2 * d, n_classes)

    def embed(self, x, m):
        h = self.stem(features(x, m)) + self.pos
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return torch.cat([h.mean(1), h.amax(1)], -1)

    def forward(self, x, m):
        return self.head(self.drop(self.embed(x, m)))


class Exported(nn.Module):
    """What the browser runs: logits + unit-length fingerprint (for taught signs)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, m):
        e = self.net.embed(x, m)
        return self.net.head(e), e / e.norm(dim=-1, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------- augmentation (GPU)
def mirror_perm(X, M):
    """Point order after a left-right flip, found from the data (faces are symmetric)."""
    perm = np.arange(P)
    for a, b in ((1, 2), (3, 4), (5, 6), (7, 8)):
        perm[a], perm[b] = b, a
    perm[9:30], perm[30:51] = np.arange(30, 51), np.arange(9, 30)
    face = slice(51, P)
    xs = X[:4000:7, ::6, face].astype(np.float32)
    ms = M[:4000:7, ::6, face]
    rel = xs - np.nanmean(np.where(ms[..., None], xs, np.nan), axis=2, keepdims=True)
    mean = np.nanmean(np.where(ms[..., None], rel, np.nan).reshape(-1, P - 51, 2), axis=0)
    flipped = mean * np.array([-1, 1])
    d = ((flipped[:, None] - mean[None]) ** 2).sum(-1)
    perm[face] = 51 + d.argmin(1)
    return torch.as_tensor(perm)


def augment(x, m, perm, g):
    """x: (B, 48, P, 2), m: (B, 48, P) on the GPU -> (B, 32, P, 2), (B, 32, P)."""
    B, dev = x.shape[0], x.device
    # time: random crop of 75-100% and random speed, resampled to 32 frames
    length = (torch.rand(B, 1, device=dev, generator=g) * 0.25 + 0.75) * (T_IN - 1)
    start = torch.rand(B, 1, device=dev, generator=g) * (T_IN - 1 - length)
    idx = (start + torch.linspace(0, 1, T, device=dev) * length).round().long().clamp(0, T_IN - 1)
    x = torch.gather(x, 1, idx[:, :, None, None].expand(-1, -1, P, 2))
    m = torch.gather(m, 1, idx[:, :, None].expand(-1, -1, P))
    # left-handed signer
    flip = torch.rand(B, device=dev, generator=g) < 0.5
    xf = x[:, :, perm].clone()
    xf[..., 0] = 1 - xf[..., 0]
    x = torch.where(flip[:, None, None, None], xf, x)
    m = torch.where(flip[:, None, None], m[:, :, perm], m)
    # camera: scale, aspect, rotation, shear, shift
    sc = torch.rand(B, 1, 1, 1, device=dev, generator=g) * 0.4 + 0.8
    asp = torch.rand(B, 1, 1, 1, device=dev, generator=g) * 0.2 + 0.9
    ang = (torch.rand(B, device=dev, generator=g) - 0.5) * math.radians(30)
    shear = (torch.rand(B, device=dev, generator=g) - 0.5) * 0.2
    sh = (torch.rand(B, 1, 1, 2, device=dev, generator=g) - 0.5) * 0.16
    cos, sin = torch.cos(ang), torch.sin(ang)
    R = torch.stack([torch.stack([cos, -sin + shear], -1), torch.stack([sin, cos], -1)], -2)  # (B, 2, 2)
    c = x - 0.5
    c = torch.einsum("btpi,bji->btpj", c, R)
    c[..., :1] = c[..., :1] * asp
    x = c * sc + 0.5 + sh
    # laptop webcam: zoom in so hands can leave the frame
    zoom = (torch.rand(B, device=dev, generator=g) < 0.4)
    k = (torch.rand(B, 1, 1, 1, device=dev, generator=g) * 1.5 + 1.0)
    ctr = ((x[:, :, 1:2] + x[:, :, 2:3]) / 2 * m[:, :, 1:2, None]).sum(1, keepdim=True) / m[:, :, 1:2, None].sum(1, keepdim=True).clamp(min=1)
    xz = (x - ctr) * k + torch.tensor([0.5, 0.6], device=dev)
    x = torch.where(zoom[:, None, None, None], xz, x)
    inside = ((x > 0) & (x < 1)).all(-1).float()
    for sl in (L.SL_LH, L.SL_RH):
        frac = (inside[:, :, sl] * m[:, :, sl]).sum(-1) / m[:, :, sl].sum(-1).clamp(min=1)
        lost = (frac < 0.6) & zoom[:, None]
        m[:, :, sl] = m[:, :, sl] * (~lost).float().unsqueeze(-1)
    # tracker: noise, a hand dropping out, missing frames
    x = x + torch.randn(x.shape, device=dev, generator=g) * 0.004
    for sl in (L.SL_LH, L.SL_RH):
        drop = (torch.rand(B, T, device=dev, generator=g) < 0.12) & (torch.rand(B, 1, device=dev, generator=g) < 0.3)
        m[:, :, sl] = m[:, :, sl] * (~drop).float().unsqueeze(-1)
    # face not tracked (mask, poor light, turned away): hide lips, eyes and nose for some samples
    noface = (torch.rand(B, device=dev, generator=g) < 0.2).float()[:, None, None]
    m[:, :, 51:] = m[:, :, 51:] * (1 - noface)
    gone = (torch.rand(B, T, device=dev, generator=g) < 0.08).float().unsqueeze(-1)
    m = m * (1 - gone)
    return x * m.unsqueeze(-1), m


def evenly(x, m):
    idx = torch.floor(torch.linspace(0, T_IN - 1, T, device=x.device) + 0.5).long()
    return x[:, idx] * m[:, idx].unsqueeze(-1), m[:, idx]


# ---------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, X, M, y, bs=1024):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        xb, mb = evenly(X[i:i + bs].float(), M[i:i + bs].float())
        with torch.autocast("cuda", dtype=torch.float16, enabled=X.is_cuda):
            probs.append(model(xb, mb).float().softmax(-1))
    p = torch.cat(probs)
    top = p.topk(5, -1).indices
    yy = y[:, None]
    return (top[:, 0] == y).float().mean().item(), (top == yy).any(-1).float().mean().item(), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="asl", choices=sorted(LANGS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--head-drop", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="", help="export folder (default data/models_v2/<lang>)")
    ap.add_argument("--all-data", action="store_true",
                    help="final model: also train on the test recordings (report the earlier test score)")
    ap.add_argument("--reported", default="", help="model.json of the tested run, whose scores are kept")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    d = np.load(ROOT / "data" / "cache_v2" / f"{args.lang}.npz", allow_pickle=True)
    Xn, labels = d["X"], d["labels"]
    classes, cats = d["classes"].tolist(), d["categories"].tolist()
    signers = d["signers"]
    tr, va = split(labels, 0.2, args.seed, signers if (signers >= 0).all() else None)
    if args.all_data:
        tr = np.arange(len(labels))
    Mn = np.isfinite(Xn[..., 0])
    perm = mirror_perm(Xn, Mn).to(dev)
    X = torch.as_tensor(np.nan_to_num(Xn, nan=0.0), dtype=torch.float16, device=dev)
    M = torch.as_tensor(Mn, dtype=torch.bool, device=dev)
    y = torch.as_tensor(labels, device=dev)
    del Xn, Mn
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    print(f"{args.lang}: train {len(tr)} val {len(va)} ({'new signers' if (signers >= 0).all() else 'random split'}), "
          f"{len(classes)} signs, device {dev}", flush=True)

    model = SignNetV2(len(classes), d=args.dim, repeats=args.repeats, drop=args.drop, head_drop=args.head_drop).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    g = torch.Generator(device=dev).manual_seed(args.seed)
    best, best_state, t0 = -1.0, None, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), device=dev, generator=g)]
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            xb, mb = augment(X[b].float(), M[b].float(), perm, g)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(xb, mb), y[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            top1, top5, _ = evaluate(model, X[va_t], M[va_t], y[va_t])
            if args.all_data:  # test recordings are in training now: keep the last model
                top1 = epoch / args.epochs
            if top1 > best:
                best, best_state = top1, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch + 1:3d}  loss {total / len(tr):.3f}  val top1 {top1:.3f}  top5 {top5:.3f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    model.load_state_dict(best_state)
    top1, top5, p = evaluate(model, X[va_t], M[va_t], y[va_t])
    conf, right = p.max(-1).values, p.argmax(-1) == y[va_t]
    print(f"final val top1 {top1:.4f}  top5 {top5:.4f}")
    for th in (0.5, 0.7):
        k = conf >= th
        print(f"  confidence >= {th}: answers {k.float().mean():.2f} of signs, correct {right[k].float().mean():.3f}")

    out = Path(args.out) if args.out else ROOT / "data" / "models_v2" / args.lang
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "classes": classes, "args": vars(args)}, out / "model.pt")
    exp = Exported(model.float().cpu().eval())
    dummy = (torch.zeros(1, T, P, 2), torch.ones(1, T, P))
    torch.onnx.export(exp, dummy, out / "model.onnx", input_names=["points", "mask"],
                      output_names=["logits", "embedding"], opset_version=17, dynamo=False)
    manifest = {
        "format": 2, "version": time.strftime("%Y%m%d-%H%M"), "lang": args.lang, "language": LANGS[args.lang],
        "frames": T, "points": P,
        "pose": L.POSE, "lips": L.LIPS, "eye_r": L.EYE_R, "eye_l": L.EYE_L, "nose": L.NOSE,
        "classes": classes, "labels": [display(c) for c in classes], "categories": cats,
        "metrics": ({**json.loads(Path(args.reported).read_text())["metrics"], "train_samples": len(tr),
                     "trained_on_all_data": True} if args.reported else
                    {"val_top1": round(top1, 4), "val_top5": round(top5, 4), "val_samples": len(va),
                     "train_samples": len(tr), "val_new_signers": bool((signers >= 0).all())}),
    }
    (out / "model.json").write_text(json.dumps(manifest))
    print("exported", out / "model.onnx", f"{(out / 'model.onnx').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
