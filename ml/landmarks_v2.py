"""The 127 points used by the v2 sign models (MediaPipe Holistic indices).

Order in every v2 array: pose (9) | left hand (21) | right hand (21) | lips (40) |
right eye (16) | left eye (16) | nose (4). "Left"/"right" are the signer's own sides.
Must match static/js/sign-model-v2.js.
"""
import numpy as np
import pandas as pd

POSE = [0, 11, 12, 13, 14, 15, 16, 23, 24]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
NOSE = [1, 2, 98, 327]
FACE = LIPS + EYE_R + EYE_L + NOSE

N_POINTS = len(POSE) + 42 + len(FACE)  # 127
SL_POSE = slice(0, 9)
SL_LH = slice(9, 30)
SL_RH = slice(30, 51)
SL_LIPS = slice(51, 91)
SL_EYE_R = slice(91, 107)
SL_EYE_L = slice(107, 123)
SL_NOSE = slice(123, 127)
# pose slots: 0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow, 5 L wrist, 6 R wrist, 7 L hip, 8 R hip


def load(path):
    """parquet (path or file-like) -> (frames, 127, 2) float16 with NaN for missing points."""
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y"])
    frames = np.sort(df.frame.unique())
    fidx = {f: i for i, f in enumerate(frames)}
    out = np.full((len(frames), N_POINTS, 2), np.nan, dtype=np.float32)

    def put(kind, ids, offset):
        slot = {v: i for i, v in enumerate(ids)}
        d = df[(df.type == kind) & df.landmark_index.isin(ids)]
        out[d.frame.map(fidx).values, offset + d.landmark_index.map(slot).values] = d[["x", "y"]].values

    put("pose", POSE, 0)
    put("left_hand", list(range(21)), 9)
    put("right_hand", list(range(21)), 30)
    put("face", FACE, 51)
    return out.astype(np.float16)
