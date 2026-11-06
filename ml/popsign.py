"""PopSign ASL v1.0 (Georgia Tech / NTID, CC BY 4.0): 250 signs filmed on phone selfie cameras.

The Kaggle competition data (data/asl) came from PopSign. The Kaggle uploads by mrgeislinger hold
the raw videos (test set 131 GB, train 670 GB), so this reads chosen videos straight out of the
archives with HTTP range requests instead of downloading them whole.

  py -3.11 ml/popsign.py --set test --per-sign-signer 2      # sample, keyed by signer and sign
  data/venv_mp/Scripts/python ml/holistic_extract.py data/popsign/test

Needs your Kaggle token in %USERPROFILE%\\.kaggle\\access_token.
Attribution: PopSign ASL v1.0, Starner et al., NeurIPS 2023 Datasets and Benchmarks (CC BY 4.0).
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.asl_citizen import RemoteZip  # noqa: E402

SETS = {"test": "popsign-asl-v1-0-game-test", "train-ae": "popsign-asl-v1-0-game-train-a-e-signs",
        "train-fm": "popsign-asl-v1-0-game-train-f-m-signs", "train-ns": "popsign-asl-v1-0-game-train-n-s-signs",
        "train-tz": "popsign-asl-v1-0-game-train-t-z-signs"}
OWNER = "mrgeislinger"
# e.g. TV/TV/gtsignstudy4a.8031-TV-2023_01_24_20_22_42.559-0.mp4  -> signer 8031, sign TV
NAME = re.compile(r"(?:^|/)[^/]*?\.(\d+)-(.+?)-\d{4}_\d\d_\d\d_[\d_.]+-\d+\.mp4$")


def signed_url(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        req = ApiDownloadDatasetRequest()
        req.owner_slug, req.dataset_slug = OWNER, slug
        resp = k.datasets.dataset_api_client.download_dataset(req)
        url = resp.url
        resp.close()
    return url


class KaggleZip(RemoteZip):
    """RemoteZip whose signed URL is renewed when it expires."""

    def __init__(self, slug):
        self.slug, self.lock = slug, threading.Lock()
        super().__init__(signed_url(slug))

    def read(self, start, length):
        for attempt in range(8):
            url = self.url
            try:
                r = self.s.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=300)
                if r.status_code in (400, 401, 403):
                    with self.lock:
                        if self.url == url:
                            self.url = signed_url(self.slug)
                    continue
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError("short read")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 7:
                    raise
                time.sleep(4 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test", choices=sorted(SETS))
    ap.add_argument("--per-sign-signer", type=int, default=2)
    ap.add_argument("--signers", default="", help="comma-separated signer ids to keep (default all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ROOT / "data" / "popsign" / args.set
    (out / "videos").mkdir(parents=True, exist_ok=True)
    z = KaggleZip(SETS[args.set])
    ents = z.entries()
    vids = [(n, NAME.search(n)) for n in ents if n.lower().endswith(".mp4")]
    vids = [(n, mm.group(1), mm.group(2)) for n, mm in vids if mm]
    print(f"{args.set}: archive {z.size / 1e9:.1f} GB, {len(vids)} videos, "
          f"{len({v[1] for v in vids})} signers, {len({v[2] for v in vids})} signs", flush=True)
    keep = set(args.signers.split(",")) if args.signers else None
    groups = {}
    for n, who, sign in vids:
        if keep is None or who in keep:
            groups.setdefault((who, sign), []).append(n)
    rng = random.Random(args.seed)
    picked = []
    for (who, sign), names in sorted(groups.items()):
        rng.shuffle(names)
        picked += [(n, who, sign) for n in names[:args.per_sign_signer]]
    print(f"fetching {len(picked)} videos", flush=True)

    from ml.holistic_extract import load
    have = set(load(out))  # landmarks already extracted (their videos may have been deleted)

    def get(p):
        dst = out / "videos" / Path(p[0]).name
        if not dst.exists() and dst.name not in have:
            dst.write_bytes(z.member(ents[p[0]]))

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(get, picked)):
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(picked)}", flush=True)
    old = json.loads((out / "picked.json").read_text()) if (out / "picked.json").exists() else []
    seen = {o["video"] for o in old}
    old += [{"video": Path(n).name, "sign": s, "signer": w} for n, w, s in picked if Path(n).name not in seen]
    (out / "picked.json").write_text(json.dumps(old))
    print("saved", out, len(old), "videos listed")


if __name__ == "__main__":
    main()
