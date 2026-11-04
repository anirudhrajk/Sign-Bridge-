import os
import math
from datetime import datetime

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
    "home":   "I want to go home.",
    "good":   "I am feeling good.",
    "bad":    "I am feeling bad.",
    "name":   "What is your name?",
    "time":   "What time is it?",
    "family": "I am here with my family."
}

def supported_signs():
    return sorted(SUPPORTED_SIGNS.keys())

def get_pt(frame, idx):
    pt = frame[idx]
    if isinstance(pt, dict):
        return (float(pt.get('x', 0)), float(pt.get('y', 0)))
    return (float(pt[0]), float(pt[1]))

def predict_signs(landmarks_sequence):
    if not landmarks_sequence or len(landmarks_sequence) < 5:
        return {
            "sign": "none",
            "message": "Waiting for movement...",
            "confidence": 0.0,
            "supported": list(SUPPORTED_SIGNS.keys())
        }

    # Get the most recent frame
    frame = landmarks_sequence[-1]

    try:
        # Check if fingers are pointing UP (Tip Y < PIP Y)
        index_up = get_pt(frame, INDEX_TIP)[1] < get_pt(frame, INDEX_PIP)[1]
        middle_up = get_pt(frame, MIDDLE_TIP)[1] < get_pt(frame, MIDDLE_PIP)[1]
        ring_up = get_pt(frame, RING_TIP)[1] < get_pt(frame, RING_PIP)[1]
        pinky_up = get_pt(frame, PINKY_TIP)[1] < get_pt(frame, PINKY_PIP)[1]
        
        # Check thumb (Tip X compared to Index base)
        thumb_up = get_pt(frame, THUMB_TIP)[1] < get_pt(frame, INDEX_PIP)[1]

        sign_key = "none"
        confidence = 0.95

        # --- RULE BASED AI LOGIC ---
        if index_up and middle_up and ring_up and pinky_up:
            sign_key = "hello" # Open palm
        elif not index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_up:
                sign_key = "good" # Thumbs up
            else:
                sign_key = "help" # Fist
        elif index_up and middle_up and not ring_up and not pinky_up:
            sign_key = "water" # Peace sign
        elif index_up and not middle_up and not ring_up and not pinky_up:
            sign_key = "doctor" # Pointing
        elif not index_up and not middle_up and not ring_up and pinky_up:
            sign_key = "bad" # Pinky up
        elif index_up and not middle_up and not ring_up and pinky_up:
            sign_key = "family" # Rock sign

        if sign_key != "none":
            return {
                "sign": sign_key,
                "message": SUPPORTED_SIGNS[sign_key],
                "confidence": confidence,
                "supported": list(SUPPORTED_SIGNS.keys())
            }
            
    except Exception as e:
        print(f"Geometry error: {e}")

    return {
        "sign": "none",
        "message": "No sign detected. Try a different hand shape.",
        "confidence": 0.0,
        "supported": list(SUPPORTED_SIGNS.keys())
    }