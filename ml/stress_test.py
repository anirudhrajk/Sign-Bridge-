"""Stress test an exported sign model under many real-world conditions.

Every test uses held-out recordings the model never trained on, changed to imitate a
situation a patient might be in. Prints top-1 / top-3 accuracy per condition, per word
category, and the most-confused pairs.

  python ml/stress_test.py [--lang isl] [--model static/models/isl] [--full]
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.check_export import forward, load  # noqa: E402
from ml.features import sequence_features  # noqa: E402
from ml.train import MIRROR, cache_path, export_dir, split  # noqa: E402
from ml.webcam_sim import lower_fps, tracker_noise, webcam, zoom  # noqa: E402


def mirror(s, r):
    """A left-handed signer: the whole body mirrored."""
    s = s[:, MIRROR].copy()
    s[..., 0] = 1 - s[..., 0]
    return s


def tilt(s, r, deg=12):
    """Phone or laptop camera tilted."""
    a = math.radians(deg) * (1 if r.random() < 0.5 else -1)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
    return ((s - 0.5) @ R.T + 0.5).astype(np.float32)


def speed(s, r, factor):
    """Signing faster (factor < 1 keeps fewer frames) or slower (> 1 repeats frames)."""
    n = len(s)
    idx = np.clip(np.round(np.linspace(0, n - 1, max(4, int(n * factor)))).astype(int), 0, n - 1)
    return s[idx]


def very_close(s, r):
    s = zoom(s, r)
    return zoom(s, r) if r.random() < 0.5 else s


def one_hand_lost(s, r):
    s = s.copy()
    hand = slice(7, 28) if r.random() < 0.5 else slice(28, 49)
    lost = r.random(len(s)) < 0.4
    s[lost, hand] = np.nan
    return s


def weak_device(s, r):
    s = s[:: 3] if len(s) > 12 else s
    return tracker_noise(s, r)


TESTS = {
    "clean (dataset video)": lambda s, r: s,
    "close to camera": zoom,
    "very close to camera": very_close,
    "low frame rate": lower_fps,
    "weak device (10 fps, noisy)": weak_device,
    "noisy tracking": tracker_noise,
    "a hand lost 40% of the time": one_hand_lost,
    "left-handed signer": mirror,
    "camera tilted 12 degrees": tilt,
    "signing 1.5x faster": lambda s, r: speed(s, r, 0.66),
    "signing 1.5x slower": lambda s, r: speed(s, r, 1.5),
    "all webcam effects": webcam,
    "webcam + left-handed": lambda s, r: webcam(mirror(s, r), r),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="isl")
    ap.add_argument("--model", default="")
    ap.add_argument("--max", type=int, default=900)
    ap.add_argument("--full", action="store_true", help="every condition, categories and confusions")
    args = ap.parse_args()
    m, t = load(Path(args.model) if args.model else export_dir(args.lang))
    vel = m.get("velocity", False)
    d = np.load(cache_path(args.lang), allow_pickle=True)
    seqs, labels, classes, cats = d["seqs"], d["labels"], d["classes"].tolist(), d["categories"].tolist()
    _, va = split(labels, 0.2, 7, d["signers"] if "signers" in d.files else None)
    va = va[:: max(1, len(va) // args.max)]
    names = list(TESTS) if args.full else ["clean (dataset video)", "close to camera", "low frame rate",
                                            "noisy tracking", "all webcam effects"]
    print(f"{len(va)} held-out recordings, {len(classes)} signs\n")
    for name in names:
        rng = np.random.default_rng(5)
        top1 = top3 = 0
        per_cat, confusions = defaultdict(lambda: [0, 0]), Counter()
        for i in va:
            p = forward(m, t, sequence_features(TESTS[name](np.asarray(seqs[i], dtype=np.float32), rng), vel))
            order = np.argsort(-p)
            ok = order[0] == labels[i]
            top1 += ok
            top3 += labels[i] in order[:3]
            c = cats[labels[i]]
            per_cat[c][0] += ok
            per_cat[c][1] += 1
            if not ok:
                confusions[(classes[labels[i]], classes[order[0]])] += 1
        print(f"{name:30s} top-1 {top1 / len(va):6.1%}   top-3 {top3 / len(va):6.1%}")
        if args.full and name == "all webcam effects":
            webcam_cats, webcam_conf = per_cat, confusions
    if args.full:
        print("\nWebcam conditions, by category (top-1):")
        for c, (ok, n) in sorted(webcam_cats.items(), key=lambda kv: kv[1][0] / kv[1][1]):
            print(f"  {c:26s} {ok / n:6.1%}  ({n} recordings)")
        print("\nMost confused under webcam conditions (signed -> read as):")
        for (a, b), n in webcam_conf.most_common(10):
            print(f"  {a} -> {b}  x{n}")


if __name__ == "__main__":
    main()
