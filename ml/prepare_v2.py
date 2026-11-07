"""Build the compact v2 training caches: every recording as 48 frames x 127 points.

  python ml/prepare_v2.py --lang isl    # from data/keypoints/**.parquet
  python ml/prepare_v2.py --lang asl    # from data/asl/shards_v2 (py -3.11 ml/download_asl.py --v2)

Output data/cache_v2/<lang>.npz: X (N, 48, 127, 2) float16 (NaN = missing), labels, classes,
categories, signers (-1 if unknown), ids. Frames where neither hand is visible are trimmed from
the start and end first, as in the browser.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.landmarks_v2 import SL_LH, SL_RH, load  # noqa: E402

FRAMES = 48
OUT = ROOT / "data" / "cache_v2"


def fixed(seq):
    seq = np.asarray(seq, dtype=np.float32)
    hand = np.isfinite(seq[:, SL_LH, 0]).any(1) | np.isfinite(seq[:, SL_RH, 0]).any(1)
    idx = np.flatnonzero(hand)
    if len(idx):
        seq = seq[idx[0]:idx[-1] + 1]
    pos = np.floor(np.linspace(0, len(seq) - 1, FRAMES) + 0.5).astype(int)
    return seq[pos].astype(np.float16)


def isl():
    files = sorted(glob.glob(str(ROOT / "data" / "keypoints" / "**" / "*.parquet"), recursive=True))
    classes = sorted({Path(f).parent.name for f in files})
    cats = {Path(f).parent.name: Path(f).parent.parent.name for f in files}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        X = np.stack([fixed(s) for s in ex.map(load, files, chunksize=16)])
    labels = np.array([classes.index(Path(f).parent.name) for f in files])
    return X, labels, classes, [cats[c] for c in classes], np.full(len(files), -1), np.arange(len(files))


def asl():
    shards = sorted((ROOT / "data" / "asl" / "shards_v2").glob("shard_*.npz"))
    if not shards:
        sys.exit("No v2 ASL shards. Run: py -3.11 ml/download_asl.py --v2")
    # Fill one preallocated array shard by shard, so memory stays near the final size.
    readable = []
    for s in shards:  # skip files a running download is still writing
        try:
            readable.append((s, len(np.load(s, allow_pickle=True)["sign"])))
        except Exception:
            print("skipping incomplete", s.name)
    shards, counts = [s for s, _ in readable], [n for _, n in readable]
    X = np.empty((sum(counts), FRAMES, 127, 2), dtype=np.float16)
    signs, signers, ids, seen, i = [], [], [], set(), 0
    for s in shards:
        d = np.load(s, allow_pickle=True)
        for q, sg, who, sid in zip(d["seqs"], d["sign"], d["participant_id"], d["sequence_id"]):
            if int(sid) in seen:  # a recording saved twice (interrupted runs): keep one
                continue
            seen.add(int(sid))
            X[i] = fixed(q)
            signs.append(sg); signers.append(int(who)); ids.append(int(sid)); i += 1
    X = X[:i]
    classes = sorted(set(signs))
    return (X, np.array([classes.index(x) for x in signs]), classes, ["asl"] * len(classes),
            np.array(signers), np.array(ids))


def trim_rest(seq, margin=3):
    """Drop the start and end where the arms hang down, as a seated webcam user's hands leave the frame."""
    shoulder = np.nanmean(seq[:, 1:3, 1], axis=1)
    hip = np.nanmean(seq[:, 7:9, 1], axis=1)
    wrist = np.nanmin(seq[:, 5:7, 1], axis=1)     # the higher wrist (y grows downwards)
    up = np.flatnonzero(wrist < (shoulder + hip) / 2)
    if len(up) < 8:
        return seq
    return seq[max(0, up[0] - margin):up[-1] + margin + 1]


def extra(folders, classes, only=None, x_scale=1.0, rest_copy=False):
    """Recordings extracted from videos (ml/holistic_extract.py), e.g. PopSign. Signer ids get
    900000 added so they never clash with Kaggle's. `only` keeps the videos with these name starts.
    `x_scale` puts narrow videos into a 16:9 frame (640x720 -> 0.5), and `rest_copy` also keeps a
    copy of each clip with the resting arms trimmed off."""
    from ml.holistic_extract import load
    import json
    Xs, ys, ws = [], [], []
    low = {c.lower(): i for i, c in enumerate(classes)}
    for f in folders:
        lm = load(f)
        for p in json.loads((Path(f) / "picked.json").read_text()):
            if only and not p["video"].startswith(tuple(only)):
                continue
            a = lm.get(p["video"])
            k = low.get(p["sign"].lower())
            if a is None or k is None or not len(a) or not np.isfinite(a[:, 9:51, 0]).any():
                continue
            a = a.astype(np.float32)
            if x_scale != 1:
                a = a.copy()
                a[..., 0] = (a[..., 0] - 0.5) * x_scale + 0.5
            Xs.append(fixed(a))
            ys.append(k)
            ws.append(900000 + int(p["signer"]))
            if rest_copy:
                Xs.append(fixed(trim_rest(a)))
                ys.append(k)
                ws.append(900000 + int(p["signer"]))
    return np.stack(Xs), np.array(ys), np.array(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["isl", "asl"], default="isl")
    ap.add_argument("--extra", nargs="*", default=[], help="folders of extracted videos to add (ASL)")
    ap.add_argument("--hold", default="", help="comma-separated extra signer ids to hold out for testing")
    ap.add_argument("--name", default="", help="cache name (default: --lang)")
    ap.add_argument("--new-words", default="", help="comma-separated words to add after the existing ones")
    ap.add_argument("--only", default="", help="comma-separated video name starts to keep from --extra")
    ap.add_argument("--x-scale", type=float, default=1.0, help="squeeze x to fit a 16:9 frame (640x720 videos: 0.5)")
    ap.add_argument("--rest-copy", action="store_true", help="also keep a copy of each clip without the resting arms")
    args = ap.parse_args()
    lang = args.lang
    name = args.name or lang
    X, labels, classes, cats, signers, ids = isl() if lang == "isl" else asl()
    more = {}
    if args.new_words:                               # new words go last, so old word numbers stay the same
        added = args.new_words.split(",")
        classes, cats = list(classes) + added, list(cats) + ["medical"] * len(added)
    if args.extra:
        from ml.train import split
        _, va = split(labels, 0.2, 7, signers if (signers >= 0).all() else None)
        sp = np.zeros(len(labels), np.int8)
        sp[va] = 1                                   # Kaggle: the same 4 held-out signers as before
        ex, ey, ew = extra(args.extra, classes, args.only.split(",") if args.only else None,
                           args.x_scale, args.rest_copy)
        hold = {900000 + int(s) for s in args.hold.split(",") if s}
        X = np.concatenate([X, ex])
        labels = np.concatenate([labels, ey])
        signers = np.concatenate([signers, ew])
        ids = np.concatenate([ids, -1 - np.arange(len(ey))])
        more["split"] = np.concatenate([sp, np.isin(ew, list(hold)).astype(np.int8)])
        print(f"added {len(ey)} recordings from {len(set(ew))} signers "
              f"({int(np.isin(ew, list(hold)).sum())} held out)")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", X=X, labels=labels, classes=np.array(classes),
             categories=np.array(cats), signers=signers, ids=ids, **more)
    print(f"{name}: {X.shape} ({X.nbytes / 1e9:.2f} GB), {len(classes)} signs -> {OUT / (name + '.npz')}")


if __name__ == "__main__":
    main()
