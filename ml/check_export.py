"""Re-run the exported browser model with plain numpy and measure its accuracy.

Confirms that the exported weights (BatchNorm folded) behave like the trained network.
With --parity it also writes a few test cases for comparing against the JavaScript
implementation in the browser (static/models/isl/_parity.json, delete after use).

  python ml/check_export.py [--parity]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.features import sequence_features  # noqa: E402
from ml.train import CACHE, EXPORT, split  # noqa: E402


def load():
    m = json.loads((EXPORT / "model.json").read_text())
    blob = np.frombuffer((EXPORT / "weights.bin").read_bytes(), dtype=np.float32)
    t = {x["name"]: blob[x["offset"]:x["offset"] + int(np.prod(x["shape"]))].reshape(x["shape"]) for x in m["tensors"]}
    return m, t


def forward(m, t, feats):
    x = (feats - t["mean"]) / t["std"]                      # (T, F)
    h = np.maximum(x @ t["inp.w"].T + t["inp.b"], 0).T      # (C, T)
    T = h.shape[1]
    for i, blk in enumerate(m["blocks"]):
        w, b, k, d = t[f"b{i}.w"], t[f"b{i}.b"], blk["kernel"], blk["dilation"]
        pad = d * (k // 2)
        hp = np.pad(h, ((0, 0), (pad, pad)))
        out = np.zeros_like(h) + b[:, None]
        for j in range(k):
            out += w[:, :, j] @ hp[:, j * d:j * d + T]
        h = h + np.maximum(out, 0)
    pooled = np.concatenate([h.mean(1), h.max(1)])
    logits = t["head.w"] @ pooled + t["head.b"]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    m, t = load()
    d = np.load(CACHE, allow_pickle=True)
    seqs, labels = d["seqs"], d["labels"]
    _, va = split(labels, 0.2, 7)
    probs = np.stack([forward(m, t, sequence_features(seqs[i])) for i in va])
    top1 = (probs.argmax(1) == labels[va]).mean()
    print(f"exported model, numpy: val top1 {top1:.3f} on {len(va)} held-out recordings")

    if args.parity:
        cases = []
        for i in va[:: max(1, len(va) // 6)][:6]:
            s = np.asarray(seqs[i], dtype=np.float32)
            frames = [[None if np.isnan(p).any() else [float(p[0]), float(p[1])] for p in f] for f in s]
            p = forward(m, t, sequence_features(s))
            top = np.argsort(-p)[:3]
            cases.append({"label": m["classes"][labels[i]], "frames": frames,
                          "expect": [{"index": int(k), "p": float(p[k])} for k in top]})
        (EXPORT / "_parity.json").write_text(json.dumps(cases))
        print("wrote", EXPORT / "_parity.json")


if __name__ == "__main__":
    main()
