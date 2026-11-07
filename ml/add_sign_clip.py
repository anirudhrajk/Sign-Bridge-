"""Add one sign to the signer's clips from a single video, without rebuilding the others.

  data/venv_mp/Scripts/python ml/add_sign_clip.py track video.mp4 out.npz --x-scale 0.5
  python ml/add_sign_clip.py add out.npz --lang isl --name fever --label Fever --category medical

The tracking step runs in the venv that has MediaPipe. The add step uses the same code as
ml/build_sign_clips.py, so the new clip lines up with the existing ones. Re-adding a name replaces it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSE = [11, 12, 13, 14, 15, 16, 23, 24]


def track(path, out, x_scale, face_ids):
    import cv2
    import mediapipe as mp
    h = mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=True, refine_face_landmarks=False,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(path))
    raw = {"pose": [], "lh": [], "rh": [], "face": []}

    def grab(lms, ids, dims):
        a = np.full((len(ids), dims), np.nan, np.float32)
        if lms:
            for i, k in enumerate(ids):
                p = lms.landmark[k]
                a[i] = (p.x, p.y, p.z)[:dims]
        return a

    while True:
        ok, img = cap.read()
        if not ok:
            break
        r = h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        raw["pose"].append(grab(r.pose_landmarks, POSE, 2))
        raw["lh"].append(grab(r.left_hand_landmarks, range(21), 3))
        raw["rh"].append(grab(r.right_hand_landmarks, range(21), 3))
        raw["face"].append(grab(r.face_landmarks, face_ids, 2))
    arrays = {k: np.stack(v) for k, v in raw.items()}
    for a in arrays.values():
        a[..., 0] = (a[..., 0] - 0.5) * x_scale + 0.5
    np.savez(out, **arrays)
    print(path, "->", out, {k: v.shape for k, v in arrays.items()})


def add(npz, lang, name, label, category):
    sys.path.insert(0, str(ROOT))
    from ml.build_sign_clips import SCALE, build_template, match_aspect, normalise
    from ml.train import cache_path
    d = np.load(npz)
    raw = {k: d[k] for k in d.files}
    isl = np.load(cache_path("isl"), allow_pickle=True)
    template = build_template({"seqs": isl["seqs"]})
    raw = match_aspect(raw)
    norm, _ = normalise(raw, template)
    # The hands stay in view at rest here, so the clip runs from when a hand rises until it drops.
    pose = norm["pose"]
    waist = (pose[:, 6:8, 1].mean(1) + pose[:, 0:2, 1].mean(1)) / 2     # between shoulders and hips
    up = np.flatnonzero(np.minimum(pose[:, 4, 1], pose[:, 5, 1]) < waist)      # the higher wrist
    a, b = max(0, up[0] - 6), min(len(pose), up[-1] + 7)
    folder = ROOT / "static" / "signs" / lang
    m = json.loads((folder / "clips.json").read_text(encoding="utf8"))
    step = round(30 / m["fps"])
    frame = np.concatenate([norm[k].reshape(len(pose), -1) for k in ("pose", "lh", "rh", "face")], 1)[a:b:step]
    if frame.shape[1] != m["frame_size"]:
        raise SystemExit(f"frame size {frame.shape[1]}, clips use {m['frame_size']}")
    q = np.clip(np.round(np.nan_to_num(frame) * SCALE), -32767, 32767).astype(np.int16).reshape(-1)
    data = np.fromfile(folder / "clips.bin", dtype=np.int16)
    for c in [c for c in m["clips"] if c["name"] == name]:      # replacing: cut the old frames out
        n = c["frames"] * m["frame_size"]
        data = np.delete(data, np.arange(c["offset"], c["offset"] + n))
        for other in m["clips"]:
            if other["offset"] > c["offset"]:
                other["offset"] -= n
        m["clips"].remove(c)
    m["clips"].append({"name": name, "label": label, "category": category, "offset": int(len(data)),
                       "frames": int(len(q) // m["frame_size"]), "confidence": 1.0})
    np.concatenate([data, q]).tofile(folder / "clips.bin")
    (folder / "clips.json").write_text(json.dumps(m), encoding="utf8")
    print(f"{name}: {len(q) // m['frame_size']} frames, {len(m['clips'])} signs")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track"); t.add_argument("video"); t.add_argument("out"); t.add_argument("--x-scale", type=float, default=1.0)
    a = sub.add_parser("add"); a.add_argument("npz"); a.add_argument("--lang", default="isl")
    a.add_argument("--name", required=True); a.add_argument("--label", required=True); a.add_argument("--category", default="medical")
    args = ap.parse_args()
    if args.cmd == "track":
        src = (ROOT / "ml" / "build_sign_clips.py").read_text(encoding="utf8")     # same face points, no pandas needed here
        groups = src[src.index("FACE_GROUPS = {"):src.index("FACE = [")]
        scope = {}
        exec(groups, scope)
        track(args.video, args.out, args.x_scale, [i for g in scope["FACE_GROUPS"].values() for i in g])
    else:
        add(args.npz, args.lang, args.name, args.label, args.category)


if __name__ == "__main__":
    main()
