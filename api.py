from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from extensions import db
from models import Session, Message, Phrase
from services.sign_recognition import predict_signs, supported_signs
from services.language_processing import sentence_to_signs
from services.speech import transcribe
from services.audit import log_action

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
    return jsonify({"signs": supported_signs()})


@api_bp.post("/signs/recognize")
@login_required
def signs_recognize():
    payload = request.get_json(silent=True) or {}
    result = predict_signs(payload.get("landmarks"))
    log_action("sign_recognized", f"{result['sign']} ({result['confidence']})")
    return jsonify(result)


@api_bp.post("/messages")
@login_required
def create_message():
    payload = request.get_json(silent=True) or {}
    body = (payload.get("body") or "").strip()
    if not body:
        return jsonify({"error": "body is required"}), 400
    if current_user.role == "patient":
        s = _open_session_for_patient()
        direction = "patient_to_doctor"
        sender_role = "patient"
    else:
        s = Session.query.get(payload.get("session_id"))
        if not s:
            return jsonify({"error": "session not found"}), 404
        direction = "doctor_to_patient"
        sender_role = "doctor"
    msg = Message(
        session_id=s.id,
        sender_id=current_user.id,
        direction=direction,
        source=payload.get("source", "text"),
        body=body,
        confirmed=bool(payload.get("confirmed", sender_role == "patient")),
        confidence=payload.get("confidence"),
    )
    db.session.add(msg)
    db.session.commit()
    log_action("message_created", f"{direction}: {body[:40]}")
    return jsonify({"id": msg.id, "body": msg.body, "created_at": msg.created_at.isoformat()})


@api_bp.post("/speech/transcribe")
@login_required
def speech_transcribe():
    audio = request.files.get("audio")
    locale = request.form.get("locale", "en")
    data = audio.read() if audio else None
    result = transcribe(data, locale=locale)
    log_action("speech_transcribed", result["text"][:40])
    return jsonify(result)


@api_bp.post("/speech/to-sign")
@login_required
def speech_to_sign():
    payload = request.get_json(silent=True) or {}
    result = sentence_to_signs(payload.get("text", ""))
    return jsonify(result)


@api_bp.get("/messages/<int:session_id>")
@login_required
def list_messages(session_id):
    s = Session.query.get_or_404(session_id)
    if current_user.role == "patient" and s.patient_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403
    msgs = Message.query.filter_by(session_id=s.id).order_by(Message.created_at).all()
    return jsonify({"messages": [
        {"id": m.id, "direction": m.direction, "source": m.source,
         "body": m.body, "confirmed": m.confirmed, "created_at": m.created_at.isoformat()}
        for m in msgs
    ]})


@api_bp.post("/consent/<purpose>")
@login_required
def toggle_consent(purpose):
    from services.consent import grant_consent, revoke_consent
    payload = request.get_json(silent=True) or {}
    if payload.get("granted"):
        grant_consent(current_user.id, purpose)
    else:
        revoke_consent(current_user.id, purpose)
    return jsonify({"ok": True})