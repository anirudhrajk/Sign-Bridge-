"""Convert long-format parquet recordings into one compact cache per sign language.

Each parquet file holds one sign video as rows of (frame, type, landmark_index, x, y, z)
with 543 landmarks per frame (face 468, pose 33, left_hand 21, right_hand 21).
We keep the upper-body pose points and both hands, x/y only, as float16.

Languages:
  --lang isl  (default) Indian Sign Language, from data/keypoints/<category>/<word>/*.parquet
  --lang asl  American Sign Language, merges shards written by ml/download_asl.py

Output: data/cache/<lang>/landmarks.npz with
  seqs     object array of (frames, 49, 2) float16 arrays (NaN = not detected)
  labels   int array, index into `classes`
  classes, categories  string arrays
  signers  int array (participant id, -1 when unknown) for signer-independent testing

Run:  python ml/prepare_data.py [--lang asl]
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.features import POSE_IDS  # noqa: E402

DATA = ROOT / "data" / "keypoints"
CACHE_DIR = ROOT / "data" / "cache"


def load(path):
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y"])
    frames = np.sort(df.frame.unique())
    fidx = {f: i for i, f in enumerate(frames)}
    out = np.full((len(frames), 49, 2), np.nan, dtype=np.float32)

    pose = df[(df.type == "pose") & (df.landmark_index.isin(POSE_IDS))]
    slot = {pid: i for i, pid in enumerate(POSE_IDS)}
    out[pose.frame.map(fidx).values, pose.landmark_index.map(slot).values] = pose[["x", "y"]].values
    for offset, kind in ((7, "left_hand"), (28, "right_hand")):
        h = df[df.type == kind]
        out[h.frame.map(fidx).values, offset + h.landmark_index.values] = h[["x", "y"]].values
    return out.astype(np.float16)


def save(lang, seqs, labels, classes, categories, signers, ids=None):
    out = CACHE_DIR / lang / "landmarks.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(seqs), dtype=object)
    arr[:] = list(seqs)
    np.savez_compressed(out, seqs=arr, labels=np.asarray(labels), classes=np.array(classes),
                        categories=np.array(categories), signers=np.asarray(signers),
                        ids=np.asarray(ids if ids is not None else np.arange(len(seqs))))
    print(f"saved {len(seqs)} recordings, {len(classes)} signs -> {out}")


def build_isl():
    files = sorted(glob.glob(str(DATA / "**" / "*.parquet"), recursive=True))
    classes = sorted({Path(f).parent.name for f in files})
    cats = {Path(f).parent.name: Path(f).parent.parent.name for f in files}
    print(f"{len(files)} recordings, {len(classes)} signs")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        seqs = list(ex.map(load, files, chunksize=16))
    labels = [classes.index(Path(f).parent.name) for f in files]
    save("isl", seqs, labels, classes, [cats[c] for c in classes], [-1] * len(files))


def build_asl():
    shards = sorted((ROOT / "data" / "asl" / "shards").glob("shard_*.npz"))
    if not shards:
        sys.exit("No ASL shards yet. Run: py -3.11 ml/download_asl.py")
    seqs, signs, signers, ids = [], [], [], []
    for s in shards:
        d = np.load(s, allow_pickle=True)
        seqs += list(d["seqs"])
        signs += d["sign"].tolist()
        signers += d["participant_id"].tolist()
        ids += d["sequence_id"].tolist()
    classes = sorted(set(signs))
    labels = [classes.index(x) for x in signs]
    save("asl", seqs, labels, classes, ["asl"] * len(classes), signers, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="isl", choices=["isl", "asl"])
    lang = ap.parse_args().lang
    if lang == "isl":
        build_isl()
    else:
        build_asl()


if __name__ == "__main__":
    main()
