"""Sign recognition runs in the browser (static/js/sign-model.js) on landmarks from
MediaPipe Holistic, so video never leaves the patient's device. The server only needs
to know which sign languages have a trained model and which signs each one knows.

Each language lives in static/models/<code>/ (model.json + model.onnx, or weights.bin for v1).
Train a new one with ml/prepare_v2.py and ml/train_v2.py (see ml/README.md).
"""
import json
from functools import lru_cache
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "static" / "models"
DEFAULT_LANGUAGE = "isl"
NAMES = {"isl": "Indian Sign Language", "asl": "American Sign Language"}


@lru_cache(maxsize=16)
def _manifest(code):
    try:
        return json.loads((MODELS_DIR / code / "model.json").read_text())
    except (OSError, ValueError):
        return None


def available_languages():
    """[{code, name, signs}] for every language with a trained model, ISL first."""
    out = []
    for d in sorted(MODELS_DIR.glob("*/model.json")):
        code = d.parent.name
        m = _manifest(code)
        if m:
            out.append({"code": code, "name": m.get("language") or NAMES.get(code, code.upper()),
                        "signs": len(m.get("classes", []))})
    out.sort(key=lambda x: (x["code"] != DEFAULT_LANGUAGE, x["name"]))
    return out


def resolve_language(code):
    """The requested language if a model exists for it, otherwise the default."""
    codes = {x["code"] for x in available_languages()}
    return code if code in codes else DEFAULT_LANGUAGE


def avatar_language(code):
    """The sign language the avatar should use: the requested one if its clips exist."""
    signs = MODELS_DIR.parent / "signs"
    return code if code and (signs / code / "clips.json").exists() else DEFAULT_LANGUAGE


def supported_signs(code=DEFAULT_LANGUAGE):
    m = _manifest(resolve_language(code)) or {}
    return m.get("labels") or m.get("classes", [])


def model_info(code=DEFAULT_LANGUAGE):
    m = _manifest(resolve_language(code)) or {}
    return {"lang": resolve_language(code), "version": m.get("version"),
            "signs": len(m.get("classes", [])), "metrics": m.get("metrics", {})}
