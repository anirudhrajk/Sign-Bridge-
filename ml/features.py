"""Feature extraction shared by training and (mirrored in static/js/sign-model.js) the browser.

Per frame we take 49 points (x, y in image coordinates, 0..1):
  0..6   pose: nose, left/right shoulder, left/right elbow, left/right wrist
  7..27  left hand (21 MediaPipe hand landmarks)
  28..48 right hand
"left" and "right" are the signer's own sides, as MediaPipe Holistic labels them.

Everything is expressed relative to the shoulders, so it does not matter where the
person sits in the frame or how far they are from the camera. Each hand's shape is also
given relative to its own wrist, so hand shape is learned independently of position.
Any change here must be copied to the JavaScript version.
"""
import numpy as np

POSE_IDS = [0, 11, 12, 13, 14, 15, 16]
N_POINTS = 49
SEQ_LEN = 32
FEAT_DIM = 49 * 2 + 42 * 2 + 2  # 184


def frames_features(P):
    """P: (frames, 49, 2) float array with NaN for missing points -> (frames, 184) float32."""
    P = np.asarray(P, dtype=np.float32)
    ls, rs = P[:, 1], P[:, 2]
    ok = ~(np.isnan(ls).any(1) | np.isnan(rs).any(1))
    center = np.where(ok[:, None], (ls + rs) / 2, 0.5)
    scale = np.where(ok, np.linalg.norm(ls - rs, axis=1), 0.25)
    scale = np.maximum(np.nan_to_num(scale, nan=0.25), 1e-3)
    glob = (P - center[:, None]) / scale[:, None, None]

    locs, flags = [], []
    for start in (7, 28):
        hand = P[:, start:start + 21]
        present = ~np.isnan(hand).any((1, 2))
        rel = hand - hand[:, :1]
        size = np.maximum(np.nan_to_num(np.linalg.norm(rel[:, 9], axis=1), nan=1.0), 1e-3)  # wrist to middle knuckle
        rel = rel / size[:, None, None]
        rel[~present] = 0.0
        locs.append(rel.reshape(len(P), -1))
        flags.append(present.astype(np.float32))

    feat = np.concatenate([glob.reshape(len(P), -1), *locs, np.stack(flags, 1)], axis=1)
    return np.nan_to_num(feat, nan=0.0).astype(np.float32)


def frame_features(p):
    """Single frame version: (49, 2) -> (184,)."""
    return frames_features(np.asarray(p)[None])[0]


def trim(seq):
    """Drop leading/trailing frames where neither hand is visible."""
    has_hand = ~(np.isnan(seq[:, 7:28, 0]).all(1) & np.isnan(seq[:, 28:49, 0]).all(1))
    idx = np.flatnonzero(has_hand)
    if len(idx) == 0:
        return seq
    return seq[idx[0]:idx[-1] + 1]


def resample(seq, length=SEQ_LEN):
    """Pick `length` frames evenly across the sequence."""
    n = len(seq)
    # floor(x + 0.5) matches JavaScript's Math.round (numpy's round is half-to-even)
    pos = np.floor(np.linspace(0, n - 1, length) + 0.5).astype(int)
    return seq[pos]


def sequence_features(seq):
    """seq: (frames, 49, 2) -> (SEQ_LEN, FEAT_DIM)."""
    return frames_features(resample(trim(np.asarray(seq, dtype=np.float32))))
