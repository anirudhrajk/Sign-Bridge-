"""Make dataset recordings look like a laptop webcam, for testing and training.

The INCLUDE videos show a standing signer with the whole upper body in frame, filmed at
about 30 fps. On a laptop the person sits close, so the frame is "zoomed in": shoulders near
the edges, hands often leave the bottom or sides of the picture, frame rates drop to 10-20 fps,
and tracking is noisier. These functions reproduce that on (frames, 49, 2) landmark arrays.
"""
import numpy as np


def zoom(seq, rng, strength=1.0):
    """Zoom in around the face/shoulders like a close webcam. Hands outside the frame vanish."""
    seq = seq.copy()
    ls, rs, nose = seq[:, 1], seq[:, 2], seq[:, 0]
    ok = ~np.isnan(ls).any(1) & ~np.isnan(rs).any(1)
    if not ok.any():
        return seq
    sw = np.nanmedian(np.abs(ls[ok, 0] - rs[ok, 0]))
    # In the dataset the shoulders span ~0.15-0.3 of the width; at a laptop they span ~0.35-0.6.
    target = rng.uniform(0.35, 0.6) if strength >= 1 else rng.uniform(0.25, 0.45)
    k = np.clip(target / max(sw, 1e-3), 1.0, 4.0)
    cx = np.nanmedian((ls[ok, 0] + rs[ok, 0]) / 2) + rng.uniform(-0.05, 0.05)
    cy = np.nanmedian(nose[:, 1]) if np.isfinite(nose[:, 1]).any() else np.nanmedian(ls[ok, 1]) - 0.1
    # Webcam framing puts the face in the upper third.
    seq[..., 0] = 0.5 + (seq[..., 0] - cx) * k
    seq[..., 1] = rng.uniform(0.28, 0.4) + (seq[..., 1] - cy) * k
    # MediaPipe loses a hand once its wrist or most of it leaves the picture.
    for start in (7, 28):
        h = seq[:, start:start + 21]
        inside = ((h[..., 0] > 0) & (h[..., 0] < 1) & (h[..., 1] > 0) & (h[..., 1] < 1)).mean(1)
        lost = (inside < 0.6) | (h[:, 0, 1] > 1.0)
        seq[lost, start:start + 21] = np.nan
    return seq


def lower_fps(seq, rng):
    """Keep every 2nd or 3rd frame (webcams with slow tracking run at 10-15 fps)."""
    step = rng.choice([1, 2, 2, 3])
    return seq[rng.integers(0, step)::step] if len(seq) > 3 * step else seq


def tracker_noise(seq, rng):
    """More jitter, and frames where a hand briefly drops out."""
    seq = seq + rng.normal(0, rng.uniform(0.002, 0.008), size=seq.shape).astype(np.float32)
    drop = rng.random(len(seq)) < rng.uniform(0.0, 0.2)
    hand = slice(7, 28) if rng.random() < 0.5 else slice(28, 49)
    seq[drop, hand] = np.nan
    return seq


def webcam(seq, rng, p_zoom=1.0):
    seq = np.asarray(seq, dtype=np.float32)
    if rng.random() < p_zoom:
        seq = zoom(seq, rng)
    return tracker_noise(lower_fps(seq, rng), rng)
