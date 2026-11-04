"""Build the avatar's sign animations from real recordings.

For every sign, pick the clearest recording (the one the recognition model is most
confident about, with the hands tracked most of the time), then keep the points the avatar
draws: upper body, both hands (with depth), and a light face outline (eyes, brows, lips),
which carry the facial grammar of sign language.

Points are measured from the middle of the shoulders in shoulder widths, so every clip
lines up on the same body. Missing hands are filled: short gaps are interpolated, and
longer ones move the hand to a relaxed rest pose at the tracked wrist.

  python ml/build_sign_clips.py [--lang isl] [--fps 15]

Output: static/signs/<lang>/clips.json (index) + clips.bin (int16 frames).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.check_export import forward, load as load_model  # noqa: E402
from ml.features import sequence_features  # noqa: E402
from ml.labels import display  # noqa: E402
from ml.train import cache_path, export_dir  # noqa: E402

POSE = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
FACE_GROUPS = {
    "oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "lips_outer": [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
    "lips_inner": [78, 80, 82, 13, 312, 310, 308, 318, 317, 14, 87, 88],
    "eye_r": [33, 160, 158, 133, 153, 144],
    "eye_l": [362, 385, 387, 263, 373, 380],
    "brow_r": [70, 63, 105, 66, 107],
    "brow_l": [336, 296, 334, 293, 300],
}
FACE = [i for g in FACE_GROUPS.values() for i in g]
SRC_FPS = 30
SCALE = 1000  # int16 units per shoulder width
MAX_GAP = 8   # source frames of missing hand that we interpolate across
# Width/height of the face (outer eye corners vs eye line to mouth) in the ISL videos, whose avatar
# looks right. Other datasets (e.g. upright phone videos) are rescaled sideways to match.
REFERENCE_FACE_RATIO = 1.408


def read_raw(path):
    """-> dict of arrays (frames, n, 2|3) in image coords, NaN when missing."""
    df = pd.read_parquet(path)
    frames = np.sort(df.frame.unique())
    fi = {f: i for i, f in enumerate(frames)}
    T = len(frames)

    def grab(kind, ids, dims):
        out = np.full((T, len(ids), dims), np.nan, dtype=np.float32)
        d = df[(df.type == kind) & df.landmark_index.isin(ids)]
        slot = {v: i for i, v in enumerate(ids)}
        out[d.frame.map(fi).values, d.landmark_index.map(slot).values] = d[["x", "y", "z"][:dims]].values
        return out

    return {"pose": grab("pose", POSE, 2), "lh": grab("left_hand", list(range(21)), 3),
            "rh": grab("right_hand", list(range(21)), 3), "face": grab("face", FACE, 2)}


def ffill(a):
    """Fill NaN frames with the nearest earlier (then later) valid frame."""
    a = a.copy()
    ok = ~np.isnan(a).reshape(len(a), -1).any(1)
    if not ok.any():
        return a
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(ok)
    idx[:first] = first
    return a[idx]


def rest_hand(template, wrist, elbow, size):
    """Place the relaxed template hand at the wrist, pointing along the forearm."""
    d = wrist - elbow
    ang = np.arctan2(d[1], d[0]) - np.pi / 2  # template points down (+y)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = template[:, :2] * size @ R.T + wrist
    return np.concatenate([xy, template[:, 2:3] * size], 1)


def match_aspect(raw):
    """Rescale x so the face has the reference proportions (fixes portrait/landscape stretch)."""
    f = raw["face"]
    i = {v: k for k, v in enumerate(FACE)}
    er, el, mt, mb = (f[:, i[33]], f[:, i[263]], f[:, i[13]], f[:, i[14]])
    ratio = np.abs(el[:, 0] - er[:, 0]) / ((mt[:, 1] + mb[:, 1]) / 2 - (el[:, 1] + er[:, 1]) / 2)
    ratio = np.nanmedian(ratio[np.isfinite(ratio) & (ratio > 0)]) if np.isfinite(ratio).any() else np.nan
    if not np.isfinite(ratio):
        return raw
    k = REFERENCE_FACE_RATIO / ratio
    out = {}
    for key, a in raw.items():
        a = a.copy()
        a[..., 0] = 0.5 + (a[..., 0] - 0.5) * k
        out[key] = a
    return out


def normalise(raw, template):
    pose = ffill(raw["pose"])
    face = ffill(raw["face"])
    ls, rs = pose[:, 0], pose[:, 1]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True)
    scale = np.where(np.isnan(scale) | (scale < 1e-3), np.nanmedian(scale), scale)
    sc = float(np.nanmedian(scale))

    def norm2(a):
        return (a - center[:, None, :2]) / sc

    out = {"pose": norm2(pose), "face": norm2(face)}
    hand_size = 0.62  # wrist-to-middle-knuckle in shoulder widths, typical adult
    present = {}
    for key, wrist_i, elbow_i in (("lh", 4, 2), ("rh", 5, 3)):
        h = raw[key].copy()
        h[..., :2] = (h[..., :2] - center[:, None]) / sc
        h[..., 2] = h[..., 2] / sc
        ok = ~np.isnan(h).reshape(len(h), -1).any(1)
        present[key] = ok
        sizes = np.linalg.norm(h[ok, 9, :2] - h[ok, 0, :2], axis=1)
        size = float(np.median(sizes)) if ok.any() else hand_size
        filled = h.copy()
        idx = np.flatnonzero(ok)
        for t in range(len(h)):
            if ok[t]:
                continue
            prev = idx[idx < t]
            nxt = idx[idx > t]
            if len(prev) and len(nxt) and nxt[0] - prev[-1] <= MAX_GAP:
                a, b = prev[-1], nxt[0]
                w = (t - a) / (b - a)
                filled[t] = h[a] * (1 - w) + h[b] * w
            else:
                filled[t] = rest_hand(template, out["pose"][t, wrist_i], out["pose"][t, elbow_i], size)
        # ease between tracked and rest so the hand never jumps
        k = np.ones(5) / 5
        for j in range(21):
            for c in range(3):
                col = filled[:, j, c]
                pad = np.pad(col, 2, mode="edge")
                filled[:, j, c] = np.convolve(pad, k, mode="valid")
        out[key] = filled
    return out, present


def build_template(cache):
    """Median relaxed hand shape (relative to wrist, unit size, pointing down)."""
    shapes = []
    for s in cache["seqs"][::25]:
        s = np.asarray(s, dtype=np.float32)
        for start in (7, 28):
            h = s[:, start:start + 21]
            ok = ~np.isnan(h).any((1, 2))
            for f in h[ok][::6]:
                rel = f - f[0]
                size = np.linalg.norm(rel[9])
                if size < 1e-4:
                    continue
                ang = np.arctan2(rel[9, 1], rel[9, 0]) - np.pi / 2
                c, s_ = np.cos(-ang), np.sin(-ang)
                shapes.append(rel / size @ np.array([[c, -s_], [s_, c]]).T)
    t = np.median(np.stack(shapes), 0)
    return np.concatenate([t, np.zeros((21, 1))], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="isl")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    npz = np.load(cache_path(args.lang), allow_pickle=True)
    # Read each array once: indexing an NpzFile re-reads and decompresses the whole array.
    cache = {k: npz[k] for k in ("seqs", "labels", "classes", "categories")}
    cache["ids"] = npz["ids"] if "ids" in npz.files else np.arange(len(cache["seqs"]))
    m, t = load_model(export_dir(args.lang))
    classes, labels = cache["classes"].tolist(), cache["labels"]
    rng = np.random.default_rng(0)

    # 1. Pick the clearest recording of each sign.
    choices = []
    for c, name in enumerate(classes):
        cand = np.flatnonzero(labels == c)
        if len(cand) > 80:  # large datasets: score a random sample
            cand = rng.choice(cand, 80, replace=False)
        best, best_score = None, -1
        for i in cand:
            seq = np.asarray(cache["seqs"][i], dtype=np.float32)
            p = forward(m, t, sequence_features(seq))[c]
            hands = (~np.isnan(seq[:, 7:49, 0]).all(1)).mean()
            score = p * (0.5 + 0.5 * hands)
            if score > best_score:
                best, best_score = i, score
        choices.append((c, name, int(best), float(best_score)))

    # 2. Find the full recording (with face points) for each choice.
    if args.lang == "isl":
        files = sorted(glob.glob(str(ROOT / "data" / "keypoints" / "**" / "*.parquet"), recursive=True))
        assert len(files) == len(cache["seqs"]), "cache and files out of sync; rerun prepare_data.py"
        source = {best: files[best] for _, _, best, _ in choices}
    else:
        raw_dir = ROOT / "data" / args.lang / "raw_clips"
        want = {best: int(cache["ids"][best]) for _, _, best, _ in choices}
        source = {b: raw_dir / f"{sid}.parquet" for b, sid in want.items()}
        missing = [sid for b, sid in want.items() if not source[b].exists()]
        if missing:
            (ROOT / "data" / args.lang / "clip_choices.json").write_text(json.dumps(sorted(want.values())))
            sys.exit(f"{len(missing)} full recordings needed. Run: py -3.11 ml/download_asl.py --clips, then rerun this.")

    # The relaxed hand comes from the ISL recordings, which are not stretched.
    isl = np.load(cache_path("isl"), allow_pickle=True)
    template = build_template({"seqs": isl["seqs"]})
    step = max(1, round(SRC_FPS / args.fps))
    out_dir = ROOT / "static" / "signs" / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)
    blobs, index, offset = [], [], 0
    for c, name, best, best_score in choices:
        raw = read_raw(source[best])
        if args.lang != "isl":
            raw = match_aspect(raw)
        norm, present = normalise(raw, template)
        active = np.flatnonzero(present["lh"] | present["rh"])
        a = max(0, active[0] - 6) if len(active) else 0
        b = min(len(norm["pose"]), active[-1] + 7) if len(active) else len(norm["pose"])
        frame = np.concatenate([norm["pose"].reshape(len(norm["pose"]), -1),
                                norm["lh"].reshape(len(norm["lh"]), -1),
                                norm["rh"].reshape(len(norm["rh"]), -1),
                                norm["face"].reshape(len(norm["face"]), -1)], 1)[a:b:step]
        q = np.clip(np.round(np.nan_to_num(frame) * SCALE), -32767, 32767).astype(np.int16)
        blobs.append(q.reshape(-1))
        index.append({"name": name, "label": display(name), "category": cache["categories"][c],
                      "offset": offset, "frames": int(len(q)), "confidence": round(float(best_score), 3)})
        offset += q.size

    idle = np.concatenate([b.reshape(-1, blobs[0].size // index[0]["frames"])[:2] for b in blobs]).mean(0)
    data = np.concatenate(blobs)
    (out_dir / "clips.bin").write_bytes(data.tobytes())
    sizes = {"pose": len(POSE) * 2, "lh": 63, "rh": 63, "face": len(FACE) * 2}
    face_slices, k = {}, 0
    for g, ids in FACE_GROUPS.items():
        face_slices[g] = [k, k + len(ids)]
        k += len(ids)
    manifest = {
        "lang": args.lang, "fps": SRC_FPS / step,
        "scale": SCALE, "frame_size": int(sum(sizes.values())), "layout": sizes,
        "pose_ids": POSE, "face_groups": face_slices,
        "idle": [int(v) for v in np.round(idle)],
        "clips": index,
    }
    (out_dir / "clips.json").write_text(json.dumps(manifest))
    print(f"{len(index)} signs, {data.size * 2 / 1e6:.1f} MB, avg {np.mean([c['frames'] for c in index]):.0f} frames "
          f"at {manifest['fps']:.0f} fps -> {out_dir}")


if __name__ == "__main__":
    main()
