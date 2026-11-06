"""Video -> 127 landmark points per frame with MediaPipe Holistic (legacy solution, like the website).

Runs in the separate MediaPipe environment (data/venv_mp, `pip install mediapipe==0.10.14
opencv-python-headless`), because mediapipe pins its own numpy/protobuf:

  data/venv_mp/Scripts/python ml/holistic_extract.py data/asl_citizen --parts 6

Uses the same settings as static/js/camera.js (model complexity 0, smoothing on) and gives each
hand to the arm it is attached to, like static/js/sign-model-v2.js, so the points match what the
app sees. Large videos are scaled to at most 960 px first (the app's camera gives 640x480).

--parts N starts N independent processes; each writes <folder>/landmarks_pK.npz every 20 videos,
so a crash loses little and a rerun continues where it stopped. load(folder) merges the parts.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSE = [0, 11, 12, 13, 14, 15, 16, 23, 24]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
NOSE = [1, 2, 98, 327]
FACE = LIPS + EYE_R + EYE_L + NOSE
MAX_SIDE = 960


def points(res):
    pts = np.full((127, 2), np.nan, dtype=np.float32)
    pose = res.pose_landmarks.landmark if res.pose_landmarks else None
    if pose:
        for i, k in enumerate(POSE):
            pts[i] = (pose[k].x, pose[k].y)
    hands = [res.left_hand_landmarks.landmark if res.left_hand_landmarks else None,
             res.right_hand_landmarks.landmark if res.right_hand_landmarks else None]
    if pose:  # give each hand to the arm it is attached to (as the browser does)
        lw, rw = pose[15], pose[16]
        d = lambda h, w: np.hypot(h[0].x - w.x, h[0].y - w.y) if h else np.inf
        a, b = hands
        if a and b:
            if d(a, rw) + d(b, lw) < d(a, lw) + d(b, rw):
                hands = [b, a]
        elif a or b:
            h = a or b
            hands = [h, None] if d(h, lw) <= d(h, rw) else [None, h]
    for off, h in ((9, hands[0]), (30, hands[1])):
        if h:
            for i in range(21):
                pts[off + i] = (h[i].x, h[i].y)
    if res.face_landmarks:
        f = res.face_landmarks.landmark
        for i, k in enumerate(FACE):
            pts[51 + i] = (f[k].x, f[k].y)
    return pts


def one(path, mp, cv2, step=1, smooth=True):
    holistic = mp.solutions.holistic.Holistic(model_complexity=0, smooth_landmarks=smooth,
                                              refine_face_landmarks=False,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % step:          # skip frames without decoding them (step 2 = 15 fps, like a phone)
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            k = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        frames.append(points(holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    cap.release()
    holistic.close()  # a fresh tracker per video, so smoothing does not carry over
    return (np.stack(frames) if frames else np.zeros((0, 127, 2), np.float32)), fps


def load(folder):
    """All extracted videos of a folder: {video name: (frames, 127, 2) float16}."""
    out = {}
    for f in sorted(Path(folder).glob("landmarks*.npz")):
        try:
            d = np.load(f)
            out.update({k: d[k] for k in d.files})
        except Exception:
            print("skipping unreadable", f.name)
    return out


def run_part(folder, k, n, step=1, smooth=True):
    import cv2
    import mediapipe as mp
    folder = Path(folder)
    vids = sorted((folder / "videos").glob("*.mp4"))
    mine = vids[k::n]
    path = folder / f"landmarks_p{k}.npz"
    done = load(folder)
    part = dict(np.load(path)) if path.exists() else {}
    todo = [v for v in mine if v.name not in done]
    print(f"part {k}/{n}: {len(mine)} videos, {len(todo)} to do", flush=True)
    for i, v in enumerate(todo):
        try:
            arr, _ = one(v, mp, cv2, step, smooth)
        except Exception as e:  # a damaged video: record it as empty and move on
            print("failed", v.name, e, flush=True)
            arr = np.zeros((0, 127, 2), np.float32)
        part[v.name] = arr.astype(np.float16)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **part)
            tmp.replace(path)
            print(f"part {k}: {i + 1}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(ROOT / "data" / "asl_citizen"))
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--part", type=int, default=-1, help="(internal) run one part")
    ap.add_argument("--step", type=int, default=1, help="track every Nth frame (2 = 15 fps from 30)")
    ap.add_argument("--no-smooth", action="store_true", help="turn off the tracker's landmark smoothing")
    args = ap.parse_args()
    if args.part >= 0:
        run_part(args.folder, args.part, args.parts, args.step, not args.no_smooth)
        return
    procs = [subprocess.Popen([sys.executable, __file__, args.folder, "--parts", str(args.parts), "--part", str(k),
                               "--step", str(args.step)] + (["--no-smooth"] if args.no_smooth else []))
             for k in range(args.parts)]
    codes = [p.wait() for p in procs]
    total = len(load(args.folder))
    vids = len(list((Path(args.folder) / "videos").glob("*.mp4")))
    print(f"done: {total}/{vids} videos extracted, exit codes {codes}")
    (Path(args.folder) / "extract_status.json").write_text(json.dumps({"extracted": total, "videos": vids}))


if __name__ == "__main__":
    main()
