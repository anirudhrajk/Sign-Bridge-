"""
Speech-to-text wrapper using Whisper-style processing.
Replace with actual Whisper model or cloud STT service.
"""


def transcribe(audio_bytes=None, locale="en"):
    if not audio_bytes:
        return {
            "text": "I will check your blood pressure now.",
            "confidence": 0.88,
            "engine": "demo",
            "locale": locale,
        }
    return {
        "text": "Please take a deep breath.",
        "confidence": 0.91,
        "engine": "stub",
        "locale": locale,
    }