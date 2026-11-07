import hashlib
import json
from pathlib import Path

from flask import Blueprint, render_template, redirect, url_for, request, flash, abort, jsonify, Response, current_app
from flask_login import login_required, current_user
from extensions import db
from models import Session, Message, MedicalSign, Phrase, AuditLog, User
from services.emergency import get_default_phrases
from services.symptoms import symptom_groups
from services.dictionary import get_local_seed
from services.audit import log_action
from services.sign_recognition import available_languages, resolve_language, avatar_language, NAMES as SIGN_LANGUAGE_NAMES
from services.consent import consent_status, has_consent
from services import patient_card
from datetime import datetime, timedelta

views_bp = Blueprint("views", __name__)


def _require_role(*roles):
    if current_user.role not in roles:
        abort(403)


@views_bp.route("/")
def welcome():
    return render_template("landing.html")


@views_bp.route("/landing")
def landing():
    return redirect(url_for("views.welcome"), code=301)


# ---------- offline support ----------
OFFLINE_PAGES = ["/together", "/offline", "/"]
SKIP_FROM_PRECACHE = {"README.md", "README.txt", "sw.js"}


def _precache_files():
    static = Path(current_app.static_folder)
    globs = ["css/*.css", "js/*.js", "icons/*.png", "icon.svg", "manifest.webmanifest",
             "models/*/model.json", "models/*/clinic.json", "models/*/weights.bin", "models/*/model.onnx", "vendor/onnxruntime/*", "landing/avatar/signer.glb", "landing/vendor/signer3d.js", "landing/fonts/*", "signs/*/clips.json", "signs/*/clips.bin", "vendor/holistic/*"]
    files = sorted({f for g in globs for f in static.glob(g) if f.is_file() and f.name not in SKIP_FROM_PRECACHE})
    return static, files


@views_bp.route("/sw.js")
def service_worker():
    static, files = _precache_files()
    h = hashlib.sha1()
    templates = Path(current_app.root_path) / "templates"
    for f in files + sorted(templates.glob("*.html")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    urls = OFFLINE_PAGES + ["/static/" + f.relative_to(static).as_posix() for f in files]
    body = (static / "sw.js").read_text(encoding="utf-8")
    body = body.replace("'__VERSION__'", json.dumps(h.hexdigest()[:12])).replace("__PRECACHE__", json.dumps(urls))
    resp = Response(body, mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@views_bp.route("/offline")
def offline():
    return render_template("offline.html")


@views_bp.route("/together")
def together():
    """Face-to-face mode on one shared device. No login and no server data, so it works offline."""
    return render_template("together.html", lang="isl", avatar_lang="isl", languages=[])


@views_bp.route("/face-to-face")
@login_required
def video_call():
    """Face-to-face over video: the patient signs, the doctor reads it in English; the doctor speaks
    or types, and the patient sees it signed. Patients call; doctors see who is waiting and answer."""
    if current_user.role == "patient":
        if not has_consent(current_user.id, "camera_capture"):
            flash("Please grant camera consent first.", "warning")
            return redirect(url_for("views.consent"))
        lang = resolve_language(current_user.sign_language)
        role = "patient"
    elif current_user.role in ("doctor", "admin"):
        lang, role = "isl", "doctor"
    else:
        abort(403)
    try:
        ice = json.loads(current_app.config["ICE_SERVERS"])
    except (ValueError, TypeError):
        ice = [{"urls": "stun:stun.l.google.com:19302"}]
    return render_template("video_call.html", role=role, lang=lang, avatar_lang=avatar_language(lang),
                           ice_servers=ice)


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
    # The visit-summary folder shows up to six visits, newest first.
    visits = Session.query.filter_by(patient_id=current_user.id).order_by(Session.created_at.desc()).limit(6).all()
    return render_template("patient_home.html", session=open_session, messages=messages, visits=visits,
                           avatar_lang=avatar_language(current_user.sign_language),
                           emergency=get_default_phrases(), consent_camera=consent_camera)


@views_bp.route("/patient/sign")
@login_required
def sign_translator():
    _require_role("patient")
    if not has_consent(current_user.id, "camera_capture"):
        flash("Please grant camera consent first.", "warning")
        return redirect(url_for("views.consent"))
    languages = available_languages()
    lang = resolve_language(current_user.sign_language)
    have = {l["code"] for l in languages}
    upcoming = [(code, name) for code, name in SIGN_LANGUAGE_NAMES.items() if code not in have]
    return render_template("sign_translator.html", languages=languages, lang=lang, upcoming=upcoming,
                           avatar_lang=avatar_language(lang))


@views_bp.route("/patient/pain")
@login_required
def pain_report():
    _require_role("patient")
    return render_template("pain_report.html")


@views_bp.route("/patient/symptoms")
@login_required
def symptoms():
    _require_role("patient")
    return render_template("symptoms.html", groups=symptom_groups())


@views_bp.route("/patient/card")
@login_required
def deaf_card():
    _require_role("patient")
    # The QR code on the back of the card: a signed link that staff open after signing in.
    qr_url = url_for("views.card_scan", token=patient_card.make_token(current_user.id), _external=True)
    return render_template("deaf_card.html", lang=resolve_language(current_user.sign_language),
                           languages={x["code"]: x["name"] for x in available_languages()},
                           qr_svg=patient_card.qr_svg(qr_url), qr_hours=patient_card.QR_HOURS,
                           shared_at=patient_card.last_share(current_user.id))


@views_bp.route("/card/<token>")
@login_required
def card_scan(token):
    """A doctor scanned a patient's card: unlock that patient's history in this browser, then open it."""
    pid = patient_card.read_token(token)
    if pid is None:
        flash("That card code has expired. Ask the patient to open their card again and show the new code.", "error")
        return redirect(url_for("views.dashboard"))
    if current_user.role not in ("doctor", "admin"):
        abort(403)
    if not db.session.get(User, pid):
        abort(404)
    patient_card.grant_from_scan(pid)
    log_action("card_scanned", f"patient {pid}")
    return redirect(url_for("views.patient_history", patient_id=pid))


@views_bp.route("/doctor/patient/<int:patient_id>/history")
@login_required
def patient_history(patient_id):
    """Every visit and message of one patient, once they shared their card or it was scanned."""
    _require_role("doctor", "admin")
    patient = db.session.get(User, patient_id)
    if not patient or patient.role != "patient":
        abort(404)
    if not patient_card.can_view_history(patient_id):
        abort(403)
    visits = Session.query.filter_by(patient_id=patient_id).order_by(Session.created_at.desc()).all()
    messages = {v.id: Message.query.filter_by(session_id=v.id).order_by(Message.created_at).all() for v in visits}
    urgent = sum(1 for ms in messages.values() for m in ms if m.source in ("emergency", "pain", "symptoms"))
    log_action("history_viewed", f"patient {patient_id}")
    return render_template("patient_history.html", patient=patient, visits=visits, messages=messages,
                           urgent=urgent, lang_name=SIGN_LANGUAGE_NAMES.get(resolve_language(patient.sign_language), "Sign language"),
                           shared_at=patient_card.last_share(patient_id), scanned=patient_card.scanned(patient_id))


@views_bp.route("/visit/<int:session_id>/summary")
@login_required
def visit_summary(session_id):
    s = Session.query.get_or_404(session_id)
    if current_user.role == "patient" and s.patient_id != current_user.id:
        abort(403)
    if current_user.role not in ("patient", "doctor", "admin"):
        abort(403)
    messages = Message.query.filter_by(session_id=s.id).order_by(Message.created_at).all()
    log_action("summary_viewed", f"session {s.id}")
    return render_template("visit_summary.html", session=s, messages=messages)


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
    # A patient whose latest message is an emergency, pain report or request needs attention first.
    urgent = {}
    for s in sessions:
        last = next((m for m in reversed(s.messages) if m.direction == "patient_to_doctor"), None)
        if last and last.source in ("emergency", "pain", "symptoms", "request", "card"):
            urgent[s.id] = {"emergency": "Emergency", "pain": "Pain report", "symptoms": "Symptoms",
                            "request": "Request", "card": "Card shared"}[last.source]
    sessions.sort(key=lambda s: s.id not in urgent)
    return render_template("doctor_home.html", sessions=sessions, counts=counts, urgent=urgent)


@views_bp.route("/doctor/session/<int:session_id>")
@login_required
def session_detail(session_id):
    _require_role("doctor", "admin")
    s = Session.query.get_or_404(session_id)
    if s.doctor_id is None and current_user.role == "doctor":
        s.doctor_id = current_user.id
        db.session.commit()
    messages = Message.query.filter_by(session_id=s.id).order_by(Message.created_at).all()
    return render_template("session_detail.html", session=s, messages=messages,
                           history_ok=patient_card.can_view_history(s.patient_id))


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
    return render_template("settings.html", languages=available_languages(),
                           lang=resolve_language(current_user.sign_language))


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