from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from extensions import db
import json

from models import Session, Message, Phrase, CustomSign
from services.sign_recognition import supported_signs, model_info, available_languages, resolve_language
from services.language_processing import sentence_to_signs
from services.audit import log_action
from services.consent import CONSENT_PURPOSES, grant_consent, revoke_consent

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _open_session_for_patient():
    s = Session.query.filter_by(patient_id=current_user.id, status="open").first()
    if not s:
        s = Session(patient_id=current_user.id, status="open")
        db.session.add(s)
        db.session.commit()
    return s


@api_bp.get("/signs")
@login_required
def signs_index():
    lang = resolve_language(request.args.get("lang") or current_user.sign_language)
    return jsonify({"signs": supported_signs(lang), "model": model_info(lang), "languages": available_languages()})


@api_bp.post("/preferences/sign-language")
@login_required
def set_sign_language():
    code = (request.get_json(silent=True) or {}).get("code", "")
    if code not in {x["code"] for x in available_languages()}:
        return jsonify({"error": "That sign language is not available yet"}), 400
    current_user.sign_language = code
    db.session.commit()
    log_action("sign_language_changed", code)
    return jsonify({"ok": True, "code": code})


@api_bp.post("/messages")
@login_required
def create_message():
    payload = request.get_json(silent=True) or {}
    body = (payload.get("body") or "").strip()
    if not body:
        return jsonify({"error": "body is required"}), 400
    if len(body) > 2000:
        return jsonify({"error": "message is too long"}), 400
    source = payload.get("source", "text")
    if source not in ("sign", "speech", "text", "emergency", "pain", "request"):
        source = "text"
    if current_user.role == "patient":
        s = _open_session_for_patient()
        direction = "patient_to_doctor"
    elif current_user.role in ("doctor", "admin"):
        s = db.session.get(Session, payload.get("session_id") or 0)
        if not s:
            return jsonify({"error": "Choose a patient session first"}), 404
        direction = "doctor_to_patient"
    else:
        return jsonify({"error": "forbidden"}), 403
    msg = Message(
        session_id=s.id,
        sender_id=current_user.id,
        direction=direction,
        source=source,
        body=body,
        confirmed=True,  # both senders review the text before sending
        confidence=payload.get("confidence"),
    )
    db.session.add(msg)
    db.session.commit()
    log_action("message_created", f"{direction} via {source}")
    return jsonify({"id": msg.id, "body": msg.body, "created_at": msg.created_at.isoformat()})


@api_bp.post("/speech/to-sign")
@login_required
def speech_to_sign():
    payload = request.get_json(silent=True) or {}
    result = sentence_to_signs(payload.get("text", ""), Phrase.query.all())
    return jsonify(result)


@api_bp.get("/messages/<int:session_id>")
@login_required
def list_messages(session_id):
    s = Session.query.get_or_404(session_id)
    if current_user.role == "patient" and s.patient_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403
    msgs = Message.query.filter_by(session_id=s.id).order_by(Message.created_at).all()
    return jsonify({"messages": [
        {"id": m.id, "direction": m.direction, "source": m.source, "body": m.body,
         "sender": m.sender.display_name, "sender_role": m.sender.role,
         "confirmed": m.confirmed, "understood": m.understood,
         "created_at": m.created_at.strftime("%d %b %H:%M")}
        for m in msgs
    ], "call": {"at": s.called_at.isoformat() + "Z", "room": s.call_room} if s.called_at else None})


FEEDBACK = {
    "understood": None,
    "repeat": "Please repeat that. I did not understand.",
    "simpler": "Please say that more simply.",
}


@api_bp.post("/messages/<int:message_id>/feedback")
@login_required
def message_feedback(message_id):
    """The patient says whether they understood a doctor's message."""
    m = Message.query.get_or_404(message_id)
    if current_user.role != "patient" or m.session.patient_id != current_user.id or m.direction != "doctor_to_patient":
        return jsonify({"error": "forbidden"}), 403
    answer = (request.get_json(silent=True) or {}).get("answer")
    if answer not in FEEDBACK:
        return jsonify({"error": "unknown answer"}), 400
    m.understood = answer
    if FEEDBACK[answer]:  # also tell the doctor in the conversation
        db.session.add(Message(session_id=m.session_id, sender_id=current_user.id, direction="patient_to_doctor",
                               source="request", body=FEEDBACK[answer], confirmed=True))
    db.session.commit()
    log_action("message_feedback", f"{answer} on message {m.id}")
    return jsonify({"ok": True})


@api_bp.get("/me/call")
@login_required
def my_call():
    """The current patient's latest 'your turn' call, polled from every page."""
    if current_user.role != "patient":
        return jsonify({"call": None})
    s = Session.query.filter_by(patient_id=current_user.id, status="open").first()
    return jsonify({"call": {"at": s.called_at.isoformat() + "Z", "room": s.call_room} if s and s.called_at else None})


@api_bp.post("/sessions/<int:session_id>/call")
@login_required
def call_patient(session_id):
    """Show the patient a full-screen, vibrating 'it's your turn' alert."""
    if current_user.role not in ("doctor", "admin"):
        return jsonify({"error": "forbidden"}), 403
    s = Session.query.get_or_404(session_id)
    s.called_at = datetime.utcnow()
    s.call_room = ((request.get_json(silent=True) or {}).get("room") or "").strip()[:40] or None
    db.session.commit()
    log_action("patient_called", f"session {s.id} room {s.call_room or '-'}")
    return jsonify({"ok": True, "at": s.called_at.isoformat() + "Z"})


@api_bp.post("/consent/<purpose>")
@login_required
def toggle_consent(purpose):
    if purpose not in CONSENT_PURPOSES:
        return jsonify({"error": "unknown purpose"}), 404
    payload = request.get_json(silent=True) or {}
    if payload.get("granted"):
        grant_consent(current_user.id, purpose)
    else:
        revoke_consent(current_user.id, purpose)
    return jsonify({"ok": True})

# ---------- taught signs ----------
def _valid_frames(frames):
    if not isinstance(frames, list) or not 8 <= len(frames) <= 150:
        return False
    for f in frames:
        if not isinstance(f, list) or len(f) != 49:
            return False
        for p in f:
            if p is not None and not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)):
                return False
    return True


@api_bp.get("/custom-signs")
@login_required
def list_custom_signs():
    lang = request.args.get("lang", "isl")
    rows = CustomSign.query.filter_by(lang=lang).order_by(CustomSign.label, CustomSign.id).all()
    return jsonify({"signs": [
        {"id": r.id, "label": r.label, "frames": json.loads(r.frames), "by": r.author.display_name,
         "mine": r.created_by == current_user.id} for r in rows
    ]})


@api_bp.post("/custom-signs")
@login_required
def add_custom_sign():
    data = request.get_json(silent=True) or {}
    label = " ".join((data.get("label") or "").strip().lower().split())[:60]
    lang = data.get("lang") or "isl"
    frames = data.get("frames")
    if not label or not all(ch.isalnum() or ch in " '-" for ch in label):
        return jsonify({"error": "Use a short word or phrase (letters and numbers)."}), 400
    if lang not in {x["code"] for x in available_languages()}:
        return jsonify({"error": "Unknown sign language"}), 400
    if not _valid_frames(frames):
        return jsonify({"error": "That recording was too short or too long. Try again."}), 400
    if CustomSign.query.filter_by(lang=lang, label=label).count() >= 20:
        return jsonify({"error": "This sign already has enough examples."}), 400
    frames = [[None if p is None else [round(float(p[0]), 4), round(float(p[1]), 4)] for p in f] for f in frames]
    row = CustomSign(lang=lang, label=label, frames=json.dumps(frames, separators=(",", ":")), created_by=current_user.id)
    db.session.add(row)
    db.session.commit()
    log_action("custom_sign_added", f"{lang}: {label}")
    return jsonify({"id": row.id})


@api_bp.delete("/custom-signs/<int:sign_id>")
@login_required
def delete_custom_sign(sign_id):
    row = CustomSign.query.get_or_404(sign_id)
    if row.created_by != current_user.id and current_user.role not in ("doctor", "admin"):
        return jsonify({"error": "forbidden"}), 403
    db.session.delete(row)
    db.session.commit()
    log_action("custom_sign_deleted", f"{row.lang}: {row.label}")
    return jsonify({"ok": True})
