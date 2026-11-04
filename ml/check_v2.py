"""Check an exported v2 model: ONNX vs PyTorch outputs, accuracy, and stress conditions.

  python ml/check_v2.py --lang isl [--dir data/models_v2/isl] [--stress] [--parity]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml import landmarks_v2 as L  # noqa: E402
from ml.train import split  # noqa: E402
from ml.train_v2 import T, T_IN, Exported, SignNetV2  # noqa: E402

TRAIN_MIRROR = None


def even(n, k):
    return np.floor(np.linspace(0, n - 1, k) + 0.5).astype(int)


def to_inputs(seq48):
    s = np.asarray(seq48, dtype=np.float32)[even(T_IN, T)]
    m = np.isfinite(s[..., 0]).astype(np.float32)
    return np.nan_to_num(s, nan=0.0)[None], m[None]


# ---- conditions (on 48-frame arrays, NaN = missing)
def zoom(s, r):
    s = s.copy()
    ls, rs = s[:, 1], s[:, 2]
    ok = np.isfinite(ls).all(1) & np.isfinite(rs).all(1)
    if not ok.any():
        return s
    sw = np.median(np.abs(ls[ok, 0] - rs[ok, 0]))
    k = np.clip(r.uniform(0.35, 0.6) / max(sw, 1e-3), 1, 4)
    c = np.median((ls[ok] + rs[ok]) / 2, 0)
    s = (s - c) * k + np.array([0.5, 0.6])
    for sl in (L.SL_LH, L.SL_RH):
        h = s[:, sl]
        inside = ((h > 0) & (h < 1)).all(-1).mean(1)
        s[(inside < 0.6), sl] = np.nan
    return s


def noise(s, r):
    s = s + r.normal(0, 0.006, s.shape)
    drop = r.random(len(s)) < 0.15
    s[drop, L.SL_LH if r.random() < 0.5 else L.SL_RH] = np.nan
    return s


def low_fps(s, r):
    k = s[::3]
    return k[even(len(k), T_IN)]


def mirror(s, r, perm):
    s = s[:, perm].copy()
    s[..., 0] = 1 - s[..., 0]
    return s


def tilt(s, r):
    a = np.radians(12) * (1 if r.random() < 0.5 else -1)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (s - 0.5) @ R.T + 0.5


def speed(s, r, f):
    idx = np.clip(np.round(np.linspace(0, T_IN - 1, max(6, int(T_IN * f)))).astype(int), 0, T_IN - 1)
    k = s[idx]
    return k[even(len(k), T_IN)]


def no_face(s, r):
    s = s.copy()
    s[:, 51:] = np.nan
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="isl")
    ap.add_argument("--dir", default="")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--max", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.dir) if args.dir else ROOT / "data" / "models_v2" / args.lang
    m = json.loads((out / "model.json").read_text())
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    ck = torch.load(out / "model.pt", map_location="cpu")
    net = SignNetV2(len(m["classes"]), d=ck["args"]["dim"], repeats=ck["args"]["repeats"])
    net.load_state_dict(ck["state"])
    exp = Exported(net.eval())

    d = np.load(ROOT / "data" / "cache_v2" / f"{args.lang}.npz", allow_pickle=True)
    X, labels, signers, cats = d["X"], d["labels"], d["signers"], d["categories"].tolist()
    _, va = split(labels, 0.2, 7, signers if (signers >= 0).all() else None)
    va = va[:: max(1, len(va) // args.max)]

    # parity: ONNX vs PyTorch on a few recordings
    diffs = []
    for i in va[:8]:
        x, mk = to_inputs(X[i])
        a = sess.run(None, {"points": x, "mask": mk})[0]
        with torch.no_grad():
            b = exp(torch.from_numpy(x), torch.from_numpy(mk))[0].numpy()
        diffs.append(np.abs(a - b).max())
    print(f"ONNX vs PyTorch: max logit difference {max(diffs):.2e} on 8 recordings")

    from ml.train_v2 import mirror_perm
    perm = mirror_perm(X, np.isfinite(X[..., 0])).numpy()
    tests = {"clean": lambda s, r: s}
    if args.stress:
        tests.update({
            "close to camera": zoom, "noisy tracking": noise, "weak device (10 fps)": low_fps,
            "left-handed signer": lambda s, r: mirror(s, r, perm), "camera tilted 12 degrees": tilt,
            "signing 1.5x faster": lambda s, r: speed(s, r, 0.66), "signing 1.5x slower": lambda s, r: speed(s, r, 1.5),
            "face not tracked": no_face,
            "all webcam effects": lambda s, r: noise(zoom(low_fps(s, r) if r.random() < 0.5 else s, r), r),
        })
    print(f"{len(va)} held-out recordings ({'new signers' if (signers >= 0).all() else 'random split'})")
    for name, fn in tests.items():
        r = np.random.default_rng(5)
        t1 = t3 = 0
        for i in va:
            x, mk = to_inputs(fn(np.asarray(X[i], dtype=np.float32), r))
            p = sess.run(None, {"points": x.astype(np.float32), "mask": mk})[0][0]
            o = np.argsort(-p)
            t1 += o[0] == labels[i]
            t3 += labels[i] in o[:3]
        print(f"  {name:28s} top-1 {t1 / len(va):6.1%}   top-3 {t3 / len(va):6.1%}")


if __name__ == "__main__":
    main()
