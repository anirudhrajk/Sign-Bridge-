"""ASL Citizen (Microsoft Research, 52 Deaf/hard-of-hearing signers, webcam videos) as a test set.

The licence allows non-commercial research only and does not allow sharing the data, so SignBridge
uses it only to *measure* how the shipped models do on new signers filmed with webcams. Nothing
from it is trained into the shipped models or stored in the repo. Delete data/asl_citizen when done.

  python ml/asl_citizen.py --list                    # list the archive (reads only its directory)
  python ml/asl_citizen.py --fetch-test       # test-split videos for the signs our model knows
  python ml/asl_citizen.py --landmarks               # MediaPipe Holistic -> data/asl_citizen/test_v2.npz

Reads single files out of the 42.8 GB zip with HTTP range requests, so only the needed videos
are downloaded.
"""
import argparse
import csv
import io
import json
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
OUT = ROOT / "data" / "asl_citizen"


class RemoteZip:
    def __init__(self, url=URL):
        self.url, self.s = url, requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def read(self, start, length):
        for attempt in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=300)
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError("short read")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 7:
                    raise
                time.sleep(4 * (attempt + 1))

    def entries(self):
        tail_len = min(self.size, 1 << 20)
        tail = self.read(self.size - tail_len, tail_len)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        z64 = tail.rfind(b"PK\x06\x06")
        if z64 != -1:
            cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40:z64 + 56])
        cd = self.read(cd_off, cd_size)
        out, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            off, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            if 0xFFFFFFFF in (csize, usize, off):
                q = 0
                while q + 4 <= len(extra):
                    hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                    if hid == 1:
                        vals, r = [], q + 4
                        for present in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                            if present:
                                vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                            else:
                                vals.append(None)
                        usize = vals[0] if vals[0] is not None else usize
                        csize = vals[1] if vals[1] is not None else csize
                        off = vals[2] if vals[2] is not None else off
                    q += 4 + hlen
            out[name] = (off, csize, method)
            p += 46 + nlen + xlen + clen
        return out

    def member(self, info):
        off, csize, method = info
        head = self.read(off, 30)
        nlen, xlen = struct.unpack("<HH", head[26:30])
        data = self.read(off + 30 + nlen + xlen, csize) if csize else b""
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise ValueError(f"zip method {method}")


def our_signs():
    m = json.loads((ROOT / "static" / "models" / "asl" / "model.json").read_text())
    return m["classes"]


def norm(g):
    """ASL Citizen glosses look like 'DOG' or 'CRY2'; Kaggle classes like 'dog'. Numbered glosses are
    different variants (often different signs), so only exact matches count."""
    return g.strip().lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch-test", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-per-sign", type=int, default=1000)
    ap.add_argument("--landmarks", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.list or args.fetch_test:
        z = RemoteZip()
        ents = z.entries()
        print(f"archive {z.size / 1e9:.1f} GB, {len(ents)} entries")
        if args.list:
            dirs = {}
            for n in ents:
                d = n.rsplit("/", 1)[0] if "/" in n else ""
                dirs[d] = dirs.get(d, 0) + 1
            for d, k in sorted(dirs.items()):
                print(f"  {k:6d}  {d}/")
            for n in ents:
                if not n.lower().endswith(".mp4"):
                    print("  file:", n, ents[n][1], "bytes")
            for n in ents:  # keep the small metadata files (split lists) locally
                if n.lower().endswith((".csv", ".txt", ".json")) and ents[n][1] < 50_000_000:
                    (OUT / Path(n).name).write_bytes(z.member(ents[n]))
            return
        rows = list(csv.DictReader(io.StringIO((OUT / f"{args.split}.csv").read_text(encoding="utf-8-sig"))))
        keys = list(rows[0].keys())
        vcol = next(k for k in keys if "video" in k.lower() or "file" in k.lower())
        gcol = next(k for k in keys if "gloss" in k.lower())
        ucol = next(k for k in keys if "user" in k.lower() or "participant" in k.lower() or "signer" in k.lower())
        mine = {c.lower(): c for c in our_signs()}
        by_name = {Path(n).name: n for n in ents if n.lower().endswith(".mp4")}
        picked, count = [], {}
        for r in rows:
            c = mine.get(norm(r[gcol]))
            name = by_name.get(Path(r[vcol]).name)
            if c is None or name is None or count.get(c, 0) >= args.max_per_sign:
                continue
            count[c] = count.get(c, 0) + 1
            picked.append((name, c, r[ucol]))
        print(f"{len(picked)} videos, {len({p[1] for p in picked})} of our {len(mine)} signs, "
              f"{len({p[2] for p in picked})} signers")
        vids = OUT / "videos"
        vids.mkdir(exist_ok=True)

        def get(p):
            dst = vids / Path(p[0]).name
            if not dst.exists():
                dst.write_bytes(z.member(ents[p[0]]))
            return dst.name

        with ThreadPoolExecutor(args.workers) as ex:
            for i, _ in enumerate(ex.map(get, picked)):
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(picked)}", flush=True)
        (OUT / "picked.json").write_text(json.dumps([{"video": Path(n).name, "sign": c, "signer": u} for n, c, u in picked]))
        print("saved", vids)

    if args.landmarks:
        from ml.holistic_extract import extract_many  # noqa: E402
        extract_many(OUT)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
