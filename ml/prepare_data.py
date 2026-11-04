"""Convert the long-format parquet recordings into one compact cache.

Each parquet file holds one sign video as rows of (frame, type, landmark_index, x, y, z)
with 543 landmarks per frame (face 468, pose 33, left_hand 21, right_hand 21).
We keep the upper-body pose points and both hands, x/y only, as float16.

Output: data/cache/landmarks.npz with
  seqs   object array of (frames, 49, 2) float16 arrays (NaN = not detected)
  labels int array, index into `classes`
  classes, categories  string arrays

Run:  python ml/prepare_data.py
"""
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
OUT = ROOT / "data" / "cache" / "landmarks.npz"


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


def main():
    files = sorted(glob.glob(str(DATA / "**" / "*.parquet"), recursive=True))
    classes = sorted({Path(f).parent.name for f in files})
    cats = {Path(f).parent.name: Path(f).parent.parent.name for f in files}
    print(f"{len(files)} recordings, {len(classes)} signs")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        seqs = list(ex.map(load, files, chunksize=16))
    labels = np.array([classes.index(Path(f).parent.name) for f in files])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(seqs), dtype=object)
    arr[:] = seqs
    np.savez_compressed(OUT, seqs=arr, labels=labels, classes=np.array(classes),
                        categories=np.array([cats[c] for c in classes]), files=np.array(files))
    print("saved", OUT)


if __name__ == "__main__":
    main()
