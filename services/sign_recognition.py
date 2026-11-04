from collections import Counter

# MediaPipe Hand Landmarks
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

SUPPORTED_SIGNS = {
    "hello":  "Hello!",
    "help":   "I need help.",
    "water":  "I need water.",
    "doctor": "I need a doctor.",
    "good":   "I am feeling good.",
    "bad":    "I am feeling bad.",
    "family": "I am here with my family.",
}

# These are demo hand shapes for the prototype, not certified ISL signs.
SIGN_GESTURES = {
    "hello":  "Open palm",
    "help":   "Closed fist",
    "water":  "Index and middle fingers up",
    "doctor": "Index finger up",
    "good":   "Thumbs up",
    "bad":    "Little finger up",
    "family": "Index and little fingers up",
}

MIN_FRAMES = 5
WINDOW = 15


def supported_signs():
    return sorted(SUPPORTED_SIGNS.keys())


def sign_guide():
    return [{"key": k, "gesture": SIGN_GESTURES[k], "message": SUPPORTED_SIGNS[k]} for k in supported_signs()]


def get_pt(frame, idx):
    pt = frame[idx]
    if isinstance(pt, dict):
        return (float(pt.get('x', 0)), float(pt.get('y', 0)))
    return (float(pt[0]), float(pt[1]))


def classify_frame(frame):
    """Return the sign for one frame of 21 hand landmarks, or None."""
    index_up = get_pt(frame, INDEX_TIP)[1] < get_pt(frame, INDEX_PIP)[1]
    middle_up = get_pt(frame, MIDDLE_TIP)[1] < get_pt(frame, MIDDLE_PIP)[1]
    ring_up = get_pt(frame, RING_TIP)[1] < get_pt(frame, RING_PIP)[1]
    pinky_up = get_pt(frame, PINKY_TIP)[1] < get_pt(frame, PINKY_PIP)[1]
    thumb_up = get_pt(frame, THUMB_TIP)[1] < get_pt(frame, INDEX_PIP)[1]

    if index_up and middle_up and ring_up and pinky_up:
        return "hello"
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "good" if thumb_up else "help"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "water"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "doctor"
    if not index_up and not middle_up and not ring_up and pinky_up:
        return "bad"
    if index_up and not middle_up and not ring_up and pinky_up:
        return "family"
    return None


def _result(sign, message, confidence):
    return {"sign": sign, "message": message, "confidence": confidence, "supported": supported_signs()}


def predict_signs(landmarks_sequence):
    """Vote across recent frames. Confidence is the share of frames that agree."""
    if not isinstance(landmarks_sequence, list) or len(landmarks_sequence) < MIN_FRAMES:
        return _result("none", "Waiting for movement...", 0.0)

    frames = landmarks_sequence[-WINDOW:]
    votes = Counter()
    for frame in frames:
        try:
            votes[classify_frame(frame)] += 1
        except (IndexError, KeyError, TypeError, ValueError):
            votes[None] += 1

    sign_key, count = votes.most_common(1)[0]
    if sign_key is None:
        return _result("none", "No sign detected. Try a different hand shape.", 0.0)
    return _result(sign_key, SUPPORTED_SIGNS[sign_key], round(count / len(frames), 2))
