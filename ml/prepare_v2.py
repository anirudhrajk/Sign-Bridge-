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


def extra(folders, classes):
    """Recordings extracted from videos (ml/holistic_extract.py), e.g. PopSign. Signer ids get
    900000 added so they never clash with Kaggle's."""
    from ml.holistic_extract import load
    import json
    Xs, ys, ws = [], [], []
    low = {c.lower(): i for i, c in enumerate(classes)}
    for f in folders:
        lm = load(f)
        for p in json.loads((Path(f) / "picked.json").read_text()):
            a = lm.get(p["video"])
            k = low.get(p["sign"].lower())
            if a is None or k is None or not len(a) or not np.isfinite(a[:, 9:51, 0]).any():
                continue
            Xs.append(fixed(a.astype(np.float32)))
            ys.append(k)
            ws.append(900000 + int(p["signer"]))
    return np.stack(Xs), np.array(ys), np.array(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["isl", "asl"], default="isl")
    ap.add_argument("--extra", nargs="*", default=[], help="folders of extracted videos to add (ASL)")
    ap.add_argument("--hold", default="", help="comma-separated extra signer ids to hold out for testing")
    ap.add_argument("--name", default="", help="cache name (default: --lang)")
    args = ap.parse_args()
    lang = args.lang
    name = args.name or lang
    X, labels, classes, cats, signers, ids = isl() if lang == "isl" else asl()
    more = {}
    if args.extra:
        from ml.train import split
        _, va = split(labels, 0.2, 7, signers if (signers >= 0).all() else None)
        sp = np.zeros(len(labels), np.int8)
        sp[va] = 1                                   # Kaggle: the same 4 held-out signers as before
        ex, ey, ew = extra(args.extra, classes)
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
