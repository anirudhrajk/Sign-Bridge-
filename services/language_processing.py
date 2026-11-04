"""
Language processing helpers for ISL sentence construction.
Real ISL grammar differs from English; this module provides a contract
for future linguistic processing reviewed by fluent signers.
"""


def normalize(text):
    return " ".join((text or "").strip().split())


def signs_to_sentence(sign_keys):
    from services.sign_recognition import SUPPORTED_SIGNS
    if not sign_keys: return ""
    return " ".join(SUPPORTED_SIGNS.get(k, k) for k in sign_keys)


def sentence_to_signs(sentence, phrases=None):
    from services.sign_recognition import SUPPORTED_SIGNS
    text = (sentence or "").lower()
    hits = [k for k, v in SUPPORTED_SIGNS.items() if v.lower().strip(".") in text]
    if phrases:
        for p in phrases:
            label = p["text"] if isinstance(p, dict) else p.text
            url = p.get("video_url") if isinstance(p, dict) else p.video_url
            if label and label.lower() in text:
                return {"match": label, "signs": hits, "video_url": url}
    return {"match": None, "signs": hits, "video_url": None}