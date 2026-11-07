from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    display_name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="patient")  # patient|doctor|admin
    hospital = db.Column(db.String(120))
    preferred_language = db.Column(db.String(12), default="en")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    google_sub = db.Column(db.String(64), unique=True, index=True)  # Google account ID
    avatar_url = db.Column(db.String(512))
    sign_language = db.Column(db.String(8), default="isl")  # which sign language model to use

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def initials(self):
        parts = [p for p in self.display_name.split() if p]
        if not parts: return "?"
        if len(parts) == 1: return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Session(db.Model):
    __tablename__ = "sessions"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    status = db.Column(db.String(20), default="open")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime)
    # Visual "your turn" alert: deaf patients cannot hear their name being called.
    called_at = db.Column(db.DateTime)
    call_room = db.Column(db.String(40))

    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])
    messages = db.relationship("Message", backref="session", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    direction = db.Column(db.String(20), nullable=False)   # patient_to_doctor|doctor_to_patient
    source = db.Column(db.String(20), default="text")     # sign|speech|text|emergency
    body = db.Column(db.Text, nullable=False)
    confirmed = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.Float)
    # Patient's answer to a doctor's message: understood | repeat | simpler
    understood = db.Column(db.String(12))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship("User", foreign_keys=[sender_id])


class MedicalSign(db.Model):
    __tablename__ = "medical_signs"
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(60), nullable=False)
    description = db.Column(db.Text)
    video_url = db.Column(db.String(255))
    islrtc_id = db.Column(db.String(60))   # ISLRTC dictionary ID
    approved = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Phrase(db.Model):
    __tablename__ = "phrases"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(60), default="general")
    is_emergency = db.Column(db.Boolean, default=False)
    video_url = db.Column(db.String(255))
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    action = db.Column(db.String(120), nullable=False)
    detail = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ConsentRecord(db.Model):
    __tablename__ = "consent_records"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    purpose = db.Column(db.String(80), nullable=False)   # e.g. "camera_capture", "data_storage"
    granted = db.Column(db.Boolean, default=False)
    granted_at = db.Column(db.DateTime)
    revoked_at = db.Column(db.DateTime)
    version = db.Column(db.String(20), default="1.0")

class CustomSign(db.Model):
    """One recorded example of a sign taught by a user (for words the trained model lacks)."""
    __tablename__ = "custom_signs"
    id = db.Column(db.Integer, primary_key=True)
    lang = db.Column(db.String(8), nullable=False, index=True)
    label = db.Column(db.String(60), nullable=False)
    frames = db.Column(db.Text, nullable=False)   # JSON: [[ [x,y] | null ] * 49] per frame
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    author = db.relationship("User", foreign_keys=[created_by])


class VideoCall(db.Model):
    """A patient's video call to a doctor. The video goes browser to browser (WebRTC);
    the server only passes the connection details between them and records the call."""
    __tablename__ = "video_calls"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("sessions.id"), nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    status = db.Column(db.String(12), default="ringing")   # ringing|active|ended
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    answered_at = db.Column(db.DateTime)
    ended_at = db.Column(db.DateTime)

    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])


class CallSignal(db.Model):
    """One connection message (offer, answer or network candidate) from one side of a call to the other."""
    __tablename__ = "call_signals"
    id = db.Column(db.Integer, primary_key=True)
    call_id = db.Column(db.Integer, db.ForeignKey("video_calls.id"), nullable=False, index=True)
    sender_role = db.Column(db.String(10), nullable=False)   # patient|doctor
    kind = db.Column(db.String(12), nullable=False)          # offer|answer|ice
    payload = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
