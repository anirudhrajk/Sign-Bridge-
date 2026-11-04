"""Download Google's ASL Signs dataset (Kaggle competition "asl-signs") and shrink it.

Kaggle serves the competition as one large zip from cloud storage. Instead of saving the
whole ~60 GB zip, we read its table of contents with HTTP range requests, then stream the
archive in large chunks. Each recording is shrunk in memory to the 49 points the model uses
(x/y, float16). Only about 1 GB is written to disk, and Kaggle's API is called just once
(per-file API downloads hit its rate limit). Safe to stop and re-run: finished recordings
are skipped.

Needs your own Kaggle token in %USERPROFILE%\\.kaggle\\access_token, and the competition
rules accepted on kaggle.com. Run with Python 3.11+ (kaggle 2.x):

  py -3.11 ml/download_asl.py [--workers 6] [--limit N]

Then: python ml/prepare_data.py --lang asl  &&  python ml/train.py --lang asl
"""
import argparse
import csv
import io
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.prepare_data import load as load_v1  # noqa: E402
from ml import landmarks_v2  # noqa: E402

COMP = "asl-signs"
OUT = ROOT / "data" / "asl"
SHARDS = OUT / "shards"  # replaced by --v2 (127 points) below
SHARD_SIZE = 2000
CHUNK = 48 * 1024 * 1024  # bytes fetched per range request


class Archive:
    """Read a remote zip through HTTP range requests. Refreshes the signed URL when it expires."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.refresh()

    def refresh(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
        api = KaggleApi()
        api.authenticate()
        for attempt in range(16):  # Kaggle rate limits can last an hour or more
            try:
                with api.build_kaggle_client() as k:
                    req = ApiDownloadDataFilesRequest()
                    req.competition_name = COMP
                    resp = k.competitions.competition_api_client.download_data_files(req)
                    self.url = resp.url
                    resp.close()
                break
            except requests.HTTPError as e:
                if attempt == 15:
                    raise
                wait = min(600, 60 * (attempt + 1))
                print(f"Kaggle said {e.response.status_code if e.response is not None else e}; waiting {wait}s", flush=True)
                time.sleep(wait)
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def read(self, start, length):
        for attempt in range(6):
            url = self.url
            try:
                r = self.session.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=300)
                if r.status_code in (400, 401, 403):  # signed URL expired
                    with self.lock:
                        if self.url == url:
                            self.refresh()
                    continue
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError("short read")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))

    def entries(self):
        """Parse the central directory: [(name, header_offset, comp_size, method)]."""
        tail_len = min(self.size, 1 << 20)
        tail = self.read(self.size - tail_len, tail_len)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        z64 = tail.rfind(b"PK\x06\x06")
        if z64 != -1:  # zip64 end of central directory
            cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40:z64 + 56])
        cd = self.read(cd_off, cd_size)
        out, p = [], 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            off, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            if 0xFFFFFFFF in (csize, usize, off):  # zip64 sizes/offset live in the extra field
                q = 0
                while q + 4 <= len(extra):
                    hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                    if hid == 1:
                        vals, r = [], q + 4
                        for present in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                            if present:
                                vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                                r += 8
                            else:
                                vals.append(None)
                        usize = vals[0] if vals[0] is not None else usize
                        csize = vals[1] if vals[1] is not None else csize
                        off = vals[2] if vals[2] is not None else off
                    q += 4 + hlen
            out.append((name, off, csize, method))
            p += 46 + nlen + xlen + clen
        return out


def member_bytes(blob, base, off, csize, method):
    """Extract one member's data from a fetched span that starts at byte `base`."""
    h = off - base
    nlen, xlen = struct.unpack("<HH", blob[h + 26:h + 30])
    data = blob[h + 30 + nlen + xlen:h + 30 + nlen + xlen + csize]
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported zip method {method}")


def done_ids():
    ids = set()
    for s in SHARDS.glob("shard_*.npz"):
        ids.update(np.load(s, allow_pickle=True)["sequence_id"].tolist())
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only this many recordings (for testing)")
    ap.add_argument("--v2", action="store_true", help="keep 127 points (lips, eyes, nose) in data/asl/shards_v2")
    ap.add_argument("--clips", action="store_true",
                    help="fetch the full recordings listed in data/asl/clip_choices.json (for the avatar)")
    args = ap.parse_args()
    global SHARDS, load
    load = landmarks_v2.load if args.v2 else load_v1
    if args.v2:
        SHARDS = OUT / "shards_v2"

    SHARDS.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_index").mkdir(parents=True, exist_ok=True)
    arc = Archive()
    print(f"archive size {arc.size / 1e9:.1f} GB", flush=True)
    entries = arc.entries()
    by_name = {e[0]: e for e in entries}

    if args.clips:
        import json
        wanted = set(json.loads((OUT / "clip_choices.json").read_text()))
        raw_dir = OUT / "raw_clips"
        raw_dir.mkdir(exist_ok=True)
        picks = [e for e in entries if e[0].endswith(".parquet") and int(Path(e[0]).stem) in wanted]
        for n, (name, off, cs, m) in enumerate(picks, 1):
            dest = raw_dir / f"{Path(name).stem}.parquet"
            if dest.exists():
                continue
            head = arc.read(off, 30)
            nlen, xlen = struct.unpack("<HH", head[26:30])
            blob = arc.read(off, 30 + nlen + xlen + cs)
            dest.write_bytes(member_bytes(blob, off, off, cs, m))
            if n % 50 == 0:
                print(f"{n}/{len(picks)} full recordings", flush=True)
        print(f"saved {len(picks)} full recordings to {raw_dir}")
        return

    index = OUT / "raw_index" / "train.csv"
    if not index.exists():
        for f in ("train.csv", "sign_to_prediction_index_map.json"):
            n, off, cs, m = by_name[f]
            head = arc.read(off, 30 + 1024)
            nlen, xlen = struct.unpack("<HH", head[26:30])
            blob = arc.read(off, 30 + nlen + xlen + cs)
            (OUT / "raw_index" / f).write_bytes(member_bytes(blob, off, off, cs, m))
    meta = {int(r["sequence_id"]): r for r in csv.DictReader(index.open())}

    have = done_ids()
    todo = sorted((e for e in entries if e[0].endswith(".parquet")
                   and int(Path(e[0]).stem) in meta and int(Path(e[0]).stem) not in have),
                  key=lambda e: e[1])
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(meta)} recordings, {len(have)} already done, {len(todo)} to fetch", flush=True)

    # Group recordings into contiguous spans of about CHUNK bytes.
    batches, cur = [], []
    for e in todo:
        if cur and (e[1] + e[2] + 1024 - cur[0][1] > CHUNK):
            batches.append(cur)
            cur = []
        cur.append(e)
    if cur:
        batches.append(cur)

    lock = threading.Lock()
    buf = {"seqs": [], "sequence_id": [], "participant_id": [], "sign": []}
    stats = {"ok": 0, "fail": 0, "bytes": 0, "t0": time.time()}

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

    def work(batch):
        base = batch[0][1]
        last = batch[-1]
        blob = arc.read(base, last[1] + last[2] + 1024 - base if last[1] + last[2] + 1024 <= arc.size
                        else arc.size - base)
        results = []
        for name, off, cs, m in batch:
            try:
                seq = load(io.BytesIO(member_bytes(blob, base, off, cs, m)))
                results.append((int(Path(name).stem), seq))
            except Exception:
                results.append((int(Path(name).stem), None))
        return results, len(blob)

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(work, b) for b in batches]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results, nbytes = fut.result()
            except Exception as e:  # a re-run retries anything that failed
                stats["fail"] += 1
                print("batch failed:", repr(e)[:200], flush=True)
                continue
            with lock:
                stats["bytes"] += nbytes
                for sid, seq in results:
                    if seq is None:
                        stats["fail"] += 1
                        continue
                    r = meta[sid]
                    buf["seqs"].append(seq)
                    buf["sequence_id"].append(sid)
                    buf["participant_id"].append(int(r["participant_id"]))
                    buf["sign"].append(r["sign"])
                    stats["ok"] += 1
                    if len(buf["seqs"]) >= SHARD_SIZE:
                        flush()
            if i % 10 == 0 or i == len(batches):
                el = time.time() - stats["t0"]
                rate = stats["bytes"] / el if el else 0
                left_gb = sum(b[-1][1] + b[-1][2] - b[0][1] for b in batches[i:]) / 1e9
                eta = left_gb * 1e9 / rate / 3600 if rate else 0
                print(f"{stats['ok']}/{len(todo)} recordings  {stats['bytes'] / 1e9:.1f} GB  "
                      f"{rate / 1e6:.1f} MB/s  ~{eta:.1f} h left  failed {stats['fail']}", flush=True)
    with lock:
        flush()
    print("finished. Re-run to retry any failures." if stats["fail"] else "finished.", flush=True)


if __name__ == "__main__":
    main()
