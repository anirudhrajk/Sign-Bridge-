"""Sign recognition runs in the browser (static/js/sign-model.js) on landmarks from
MediaPipe Holistic, so video never leaves the patient's device. The server only needs
to know which signs the model supports. Retrain with ml/prepare_data.py + ml/train.py.
"""
import json
from functools import lru_cache
from pathlib import Path

MODEL_MANIFEST = Path(__file__).resolve().parent.parent / "static" / "models" / "isl" / "model.json"


@lru_cache(maxsize=1)
def _manifest():
    try:
        return json.loads(MODEL_MANIFEST.read_text())
    except (OSError, ValueError):
        return {"classes": [], "labels": [], "categories": []}


def supported_signs():
    m = _manifest()
    return m.get("labels") or m.get("classes", [])


def model_info():
    m = _manifest()
    return {"version": m.get("version"), "signs": len(m.get("classes", [])), "metrics": m.get("metrics", {})}
