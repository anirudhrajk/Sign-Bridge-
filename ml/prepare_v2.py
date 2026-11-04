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
    X, signs, signers, ids = [], [], [], []
    for s in shards:
        d = np.load(s, allow_pickle=True)
        X += [fixed(q) for q in d["seqs"]]
        signs += d["sign"].tolist()
        signers += d["participant_id"].tolist()
        ids += d["sequence_id"].tolist()
    classes = sorted(set(signs))
    return (np.stack(X), np.array([classes.index(x) for x in signs]), classes, ["asl"] * len(classes),
            np.array(signers), np.array(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["isl", "asl"], default="isl")
    lang = ap.parse_args().lang
    X, labels, classes, cats, signers, ids = isl() if lang == "isl" else asl()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{lang}.npz", X=X, labels=labels, classes=np.array(classes),
             categories=np.array(cats), signers=signers, ids=ids)
    print(f"{lang}: {X.shape} ({X.nbytes / 1e9:.2f} GB), {len(classes)} signs -> {OUT / (lang + '.npz')}")


if __name__ == "__main__":
    main()
