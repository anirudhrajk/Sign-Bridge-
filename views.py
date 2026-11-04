from flask import Blueprint, render_template, redirect, url_for, request, flash, abort, jsonify
from flask_login import login_required, current_user
from extensions import db
from models import Session, Message, MedicalSign, Phrase, AuditLog, User
from services.sign_recognition import sign_guide
from services.emergency import get_default_phrases
from services.dictionary import get_local_seed
from services.audit import log_action
from services.consent import consent_status, has_consent
from datetime import datetime, timedelta

views_bp = Blueprint("views", __name__)


def _require_role(*roles):
    if current_user.role not in roles:
        abort(403)


@views_bp.route("/")
def welcome():
    return render_template("welcome.html")


@views_bp.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("views.admin_home"))
    if current_user.role == "doctor":
        return redirect(url_for("views.doctor_home"))
    return redirect(url_for("views.patient_home"))


# ---------- patients ----------
@views_bp.route("/patient")
@login_required
def patient_home():
    _require_role("patient")
    open_session = Session.query.filter_by(patient_id=current_user.id, status="open").first()
    if not open_session:
        open_session = Session(patient_id=current_user.id, status="open")
        db.session.add(open_session)
        db.session.commit()
    messages = Message.query.filter_by(session_id=open_session.id).order_by(Message.created_at.desc()).limit(10).all()
    consent_camera = has_consent(current_user.id, "camera_capture")
    return render_template("patient_home.html", session=open_session, messages=messages,
                           emergency=get_default_phrases(), consent_camera=consent_camera)


@views_bp.route("/patient/sign")
@login_required
def sign_translator():
    _require_role("patient")
    if not has_consent(current_user.id, "camera_capture"):
        flash("Please grant camera consent first.", "warning")
        return redirect(url_for("views.consent"))
    return render_template("sign_translator.html", guide=sign_guide())


@views_bp.route("/patient/emergency")
@login_required
def emergency():
    _require_role("patient")
    phrases = Phrase.query.filter_by(is_emergency=True).order_by(Phrase.text).all()
    if not phrases:
        phrases = [Phrase(text=p["text"], category=p["category"], is_emergency=True)
                   for p in get_default_phrases()]
    return render_template("emergency.html", phrases=phrases)


# ---------- doctors ----------
@views_bp.route("/doctor")
@login_required
def doctor_home():
    _require_role("doctor", "admin")
    sessions = Session.query.filter_by(status="open").order_by(Session.created_at.desc()).all()
    counts = {s.id: len(s.messages) for s in sessions}
    return render_template("doctor_home.html", sessions=sessions, counts=counts)


@views_bp.route("/doctor/session/<int:session_id>")
@login_required
def session_detail(session_id):
    _require_role("doctor", "admin")
    s = Session.query.get_or_404(session_id)
    if s.doctor_id is None and current_user.role == "doctor":
        s.doctor_id = current_user.id
        db.session.commit()
    messages = Message.query.filter_by(session_id=s.id).order_by(Message.created_at).all()
    return render_template("session_detail.html", session=s, messages=messages)


@views_bp.route("/doctor/voice")
@login_required
def voice_translator():
    _require_role("doctor", "admin")
    phrases = Phrase.query.order_by(Phrase.category, Phrase.text).all()
    sessions = Session.query.filter_by(status="open").order_by(Session.created_at.desc()).all()
    return render_template("voice_translator.html", phrases=phrases, sessions=sessions,
                           selected_id=request.args.get("session_id", type=int))


@views_bp.route("/doctor/phrases", methods=["GET", "POST"])
@login_required
def phrase_library():
    _require_role("doctor", "admin")
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        category = (request.form.get("category") or "general").strip()
        is_emergency = bool(request.form.get("is_emergency"))
        video_url = (request.form.get("video_url") or "").strip() or None
        if text:
            db.session.add(Phrase(text=text, category=category, is_emergency=is_emergency,
                                  video_url=video_url, created_by=current_user.id))
            db.session.commit()
            log_action("phrase_created", text)
            flash("Phrase added.", "success")
        else:
            flash("Phrase text is required.", "error")
        return redirect(url_for("views.phrase_library"))
    phrases = Phrase.query.order_by(Phrase.category, Phrase.text).all()
    return render_template("phrase_library.html", phrases=phrases)


@views_bp.route("/doctor/phrases/<int:phrase_id>/delete", methods=["POST"])
@login_required
def delete_phrase(phrase_id):
    _require_role("doctor", "admin")
    phrase = Phrase.query.get_or_404(phrase_id)
    db.session.delete(phrase)
    db.session.commit()
    log_action("phrase_deleted", phrase.text)
    flash("Phrase removed.", "success")
    return redirect(url_for("views.phrase_library"))


# ---------- admin ----------
@views_bp.route("/admin")
@login_required
def admin_home():
    _require_role("admin")
    users = User.query.order_by(User.created_at.desc()).all()
    logs = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(50).all()
    return render_template("admin_home.html", users=users, logs=logs)


# ---------- shared ----------
@views_bp.route("/dictionary")
@login_required
def dictionary():
    signs = MedicalSign.query.order_by(MedicalSign.category, MedicalSign.label).all()
    if not signs:
        for s in get_local_seed():
            db.session.add(MedicalSign(label=s["label"], category=s["category"], description=s["description"]))
        db.session.commit()
        signs = MedicalSign.query.order_by(MedicalSign.category, MedicalSign.label).all()
    categories = sorted({s.category for s in signs})
    return render_template("dictionary.html", signs=signs, categories=categories)


@views_bp.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


@views_bp.route("/settings/delete-data", methods=["POST"])
@login_required
def delete_my_data():
    sessions = Session.query.filter_by(patient_id=current_user.id).all()
    for s in sessions:
        db.session.delete(s)
    db.session.commit()
    log_action("data_deleted", f"{len(sessions)} session(s) deleted by the user")
    flash("Your conversation history was deleted.", "success")
    return redirect(url_for("views.settings"))


@views_bp.route("/consent")
@login_required
def consent():
    return render_template("consent.html", consents=consent_status(current_user.id))


@views_bp.route("/analytics")
@login_required
def analytics():
    _require_role("doctor", "admin")
    total_sessions = Session.query.count()
    total_messages = Message.query.count()
    return render_template("analytics.html", total_sessions=total_sessions, total_messages=total_messages)