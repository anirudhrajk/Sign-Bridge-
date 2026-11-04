"""
Language processing helpers for ISL sentence construction.
Real ISL grammar differs from English; this module provides a contract
for future linguistic processing reviewed by fluent signers.
"""


import re


def normalize(text):
    return " ".join((text or "").strip().split())


def signs_to_sentence(sign_words):
    if not sign_words: return ""
    return " ".join(sign_words)


def sentence_to_signs(sentence, phrases=None):
    from services.sign_recognition import supported_signs
    text = (sentence or "").lower()
    hits = [k for k in supported_signs() if re.search(rf"\b{re.escape(k.lower())}\b", text)]
    if phrases:
        for p in phrases:
            label = p["text"] if isinstance(p, dict) else p.text
            url = p.get("video_url") if isinstance(p, dict) else p.video_url
            if label and label.lower() in text:
                return {"match": label, "signs": hits, "video_url": url}
    return {"match": None, "signs": hits, "video_url": None}