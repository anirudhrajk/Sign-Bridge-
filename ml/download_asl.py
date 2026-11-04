"""Download Google's ASL Signs dataset (Kaggle competition "asl-signs") and shrink it.

Each recording is fetched, reduced to the 49 points the model uses (x/y, float16), and the
original parquet is deleted right away. About 68 GB is downloaded, but only about 1 GB stays
on disk. Safe to stop and re-run: finished recordings are skipped.

Needs your own Kaggle token in %USERPROFILE%\\.kaggle\\access_token and the competition rules
accepted on kaggle.com. Run with Python 3.11+ (kaggle 2.x):

  py -3.11 ml/download_asl.py [--workers 8] [--limit N]

Output: data/asl/raw_index/train.csv and data/asl/shards/shard_*.npz, then
`python ml/prepare_data.py --lang asl` merges the shards into data/cache/asl/landmarks.npz.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.prepare_data import load  # noqa: E402

COMP = "asl-signs"
OUT = ROOT / "data" / "asl"
SHARDS = OUT / "shards"
SHARD_SIZE = 2000


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def fetch(a, name, dest):
    """Download one competition file into dest dir; unzip if Kaggle zipped it. Returns the path."""
    a.competition_download_file(COMP, name, path=str(dest), quiet=True)
    base = Path(name).name
    p = dest / base
    if p.exists():
        return p
    z = dest / (base + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
        return dest / base
    raise FileNotFoundError(name)


def done_ids():
    ids = set()
    for s in SHARDS.glob("shard_*.npz"):
        ids.update(np.load(s, allow_pickle=True)["sequence_id"].tolist())
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only this many recordings (for testing)")
    args = ap.parse_args()

    a = api()
    (OUT / "raw_index").mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    index = OUT / "raw_index" / "train.csv"
    if not index.exists():
        for f in ("train.csv", "sign_to_prediction_index_map.json"):
            shutil.move(str(fetch(a, f, Path(tempfile.mkdtemp()))), OUT / "raw_index" / f)

    rows = list(csv.DictReader(index.open()))
    have = done_ids()
    todo = [r for r in rows if int(r["sequence_id"]) not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} recordings, {len(have)} already done, {len(todo)} to fetch", flush=True)

    lock = threading.Lock()
    buf = {"seqs": [], "sequence_id": [], "participant_id": [], "sign": []}
    stats = {"ok": 0, "fail": 0, "bytes": 0, "t0": time.time()}
    local = threading.local()

    def flush():
        if not buf["seqs"]:
            return
        n = len(list(SHARDS.glob("shard_*.npz")))
        arr = np.empty(len(buf["seqs"]), dtype=object)
        arr[:] = buf["seqs"]
        np.savez_compressed(SHARDS / f"shard_{n:04d}.npz", seqs=arr,
                            sequence_id=np.array(buf["sequence_id"]),
                            participant_id=np.array(buf["participant_id"]),
                            sign=np.array(buf["sign"]))
        for k in buf:
            buf[k].clear()

    def work(r):
        if not hasattr(local, "api"):
            local.api = api()
        tmp = Path(tempfile.mkdtemp(prefix="asl_"))
        try:
            for attempt in range(4):
                try:
                    p = fetch(local.api, r["path"], tmp)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            size = p.stat().st_size
            seq = load(p)
            return r, seq, size
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r, seq, size = fut.result()
            except Exception as e:  # keep going; a re-run retries failures
                stats["fail"] += 1
                if stats["fail"] <= 5:
                    print("failed:", repr(e)[:200], flush=True)
                continue
            with lock:
                buf["seqs"].append(seq)
                buf["sequence_id"].append(int(r["sequence_id"]))
                buf["participant_id"].append(int(r["participant_id"]))
                buf["sign"].append(r["sign"])
                stats["ok"] += 1
                stats["bytes"] += size
                if len(buf["seqs"]) >= SHARD_SIZE:
                    flush()
            if i % 500 == 0 or i == len(todo):
                el = time.time() - stats["t0"]
                rate = stats["ok"] / el if el else 0
                left = (len(todo) - i) / rate / 3600 if rate else 0
                print(f"{i}/{len(todo)}  ok {stats['ok']}  failed {stats['fail']}  "
                      f"{stats['bytes'] / 1e9:.1f} GB  {rate:.1f}/s  ~{left:.1f} h left", flush=True)
    with lock:
        flush()
    print("finished. Re-run to retry any failures." if stats["fail"] else "finished.", flush=True)


if __name__ == "__main__":
    main()
