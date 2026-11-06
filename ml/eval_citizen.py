"""Score sign models on ASL Citizen: 11 Deaf/hard-of-hearing signers filmed on their own webcams,
none of whom are in our training data. This is the closest test we have to real patients.

  python ml/asl_citizen.py --fetch-test
  data/venv_mp/Scripts/python ml/holistic_extract.py data/asl_citizen
  python ml/eval_citizen.py static/models/asl data/models_v2/asl_AB ...

Only the signs our model knows are scored (exact gloss matches, 208 of 250).
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.prepare_v2 import fixed  # noqa: E402

CIT = ROOT / "data" / "asl_citizen"


def load_set(folder=CIT):
    from ml.holistic_extract import load
    lm = load(folder)
    picked = json.loads((Path(folder) / "picked.json").read_text())
    items = [p for p in picked if p["video"] in lm and len(lm[p["video"]])]
    X = np.stack([fixed(lm[p["video"]].astype(np.float32)) for p in items])
    return X, items


def score(folder, X, items):
    m = json.loads((Path(folder) / "model.json").read_text())
    sess = ort.InferenceSession(str(Path(folder) / "model.onnx"), providers=["CPUExecutionProvider"])
    t = m["frames"]
    idx = np.floor(np.linspace(0, 47, t) + 0.5).astype(int)
    y = np.array([m["classes"].index(p["sign"]) for p in items])
    P = []
    for i in range(0, len(X), 64):
        s = X[i:i + 64][:, idx]
        mk = np.isfinite(s[..., 0]).astype(np.float32)
        lg = np.concatenate([sess.run(None, {"points": np.nan_to_num(s[j:j + 1]).astype(np.float32),
                                             "mask": mk[j:j + 1]})[0] for j in range(len(s))])
        P.append(np.exp(lg - lg.max(1, keepdims=True)) / np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True))
    P = np.concatenate(P)
    top = np.argsort(-P, 1)
    r = {"top1": float((top[:, 0] == y).mean()), "top5": float((top[:, :5] == y[:, None]).any(1).mean())}
    conf = P.max(1)
    for th in (0.5, 0.7):
        k = conf >= th
        r[f"at{th}"] = {"answers": float(k.mean()), "correct": float((top[k, 0] == y[k]).mean()) if k.any() else None}
    signers = np.array([p["signer"] for p in items])
    r["per_signer"] = {s: round(float((top[signers == s, 0] == y[signers == s]).mean()), 3) for s in sorted(set(signers))}
    return r, P


def main():
    X, items = load_set()
    print(f"ASL Citizen: {len(items)} videos, {len({p['sign'] for p in items})} signs, "
          f"{len({p['signer'] for p in items})} new signers")
    probs = {}
    for folder in sys.argv[1:]:
        r, P = score(folder, X, items)
        probs[folder] = P
        print(f"\n{folder}: top-1 {r['top1']:.1%}  top-5 {r['top5']:.1%}")
        for th in (0.5, 0.7):
            a = r[f"at{th}"]
            print(f"  confidence >= {th}: answers {a['answers']:.0%}, correct {a['correct']:.1%}" if a["correct"] is not None
                  else f"  confidence >= {th}: never")
        print("  per signer:", r["per_signer"])


if __name__ == "__main__":
    main()
