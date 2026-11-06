"""Does smoothing the landmarks ourselves (before reading a sign) help when the tracker's own
smoothing is weak, e.g. on a slow phone? Scores the shipped model on ASL Citizen landmarks
extracted with and without MediaPipe's smoothing, after each candidate filter.

  python ml/smooth_test.py data/asl_citizen data/asl_citizen_ns

A sign is read only after it ends, so the filter may look forwards and backwards in time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.holistic_extract import load  # noqa: E402
from ml.prepare_v2 import fixed  # noqa: E402


def box(seq, w):
    """Centred moving average over w frames, per point, skipping missing frames."""
    if w <= 1:
        return seq
    x = np.nan_to_num(seq)
    m = np.isfinite(seq[..., :1]).astype(np.float32)
    k = np.ones(w, np.float32)
    pad = w // 2
    def conv(a):
        a = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1))
        return sum(a[i:i + len(seq)] * k[i] for i in range(w))
    s, c = conv(x * m), conv(m)
    out = s / np.maximum(c, 1e-6)
    out[m[..., 0] == 0] = np.nan            # keep missing points missing
    return out


def gauss(seq, sigma):
    w = int(sigma * 3) * 2 + 1
    k = np.exp(-0.5 * ((np.arange(w) - w // 2) / sigma) ** 2)
    x = np.nan_to_num(seq)
    m = np.isfinite(seq[..., :1]).astype(np.float32)
    pad = w // 2
    def conv(a):
        a = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1))
        return sum(a[i:i + len(seq)] * k[i] for i in range(w))
    out = conv(x * m) / np.maximum(conv(m), 1e-6)
    out[m[..., 0] == 0] = np.nan
    return out


FILTERS = {"none": lambda s: s, "box3": lambda s: box(s, 3), "box5": lambda s: box(s, 5),
           "gauss1": lambda s: gauss(s, 1.0), "gauss1.5": lambda s: gauss(s, 1.5), "gauss2": lambda s: gauss(s, 2.0)}


def main():
    folders = sys.argv[1:] or ["data/asl_citizen", "data/asl_citizen_ns"]
    m = json.loads((ROOT / "static/models/asl/model.json").read_text())
    sess = ort.InferenceSession(str(ROOT / "static/models/asl/model.onnx"), providers=["CPUExecutionProvider"])
    idx = np.floor(np.linspace(0, 47, m["frames"]) + 0.5).astype(int)
    for folder in folders:
        lm = load(folder)
        picked = [p for p in json.loads((Path(folder) / "picked.json").read_text()) if p["video"] in lm and len(lm[p["video"]])]
        y = np.array([m["classes"].index(p["sign"]) for p in picked])
        print(f"\n{folder}: {len(picked)} videos")
        for name, f in FILTERS.items():
            right = 0
            for p, yy in zip(picked, y):
                s = fixed(f(lm[p["video"]].astype(np.float32)).astype(np.float32))[idx][None].astype(np.float32)
                mk = np.isfinite(s[..., 0]).astype(np.float32)
                lg = sess.run(None, {"points": np.nan_to_num(s), "mask": mk})[0][0]
                right += lg.argmax() == yy
            print(f"  {name:9s} top-1 {right / len(picked):.1%}")


if __name__ == "__main__":
    main()
