"""The Deaf patient card: sharing it with staff, and the QR code on its back.

Staff can open a patient's full visit history only when the patient has agreed to it, in one of two ways:
  1. The patient tapped "Share my card" in the last SHARE_HOURS hours (a "card" message from them).
  2. A doctor scanned the QR code on the card. The code is signed with the app's secret key, names the
     patient, and expires after QR_HOURS hours, so an old screenshot stops working.
Either way, only signed-in doctors and admins can see the history, and every view is written to the audit log.
"""
from datetime import datetime, timedelta
from io import BytesIO

import segno
from flask import current_app, session as browser_session
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from models import Message, Session

SHARE_HOURS = 24
QR_HOURS = 12
_SALT = "signbridge-patient-card"
_GRANTS = "card_access"   # patient ids this browser unlocked by scanning a card, with expiry times


def _serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt=_SALT)


def make_token(patient_id):
    return _serializer().dumps({"p": int(patient_id)})


def read_token(token):
    """Patient id from a card QR code, or None if it is forged or older than QR_HOURS."""
    try:
        data = _serializer().loads(token, max_age=QR_HOURS * 3600)
        return int(data["p"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


def qr_svg(url):
    """The QR code as inline SVG: dark modules on white, so every phone camera reads it."""
    out = BytesIO()
    segno.make(url, error="m").save(out, kind="svg", scale=4, border=2, dark="#14221d", light="#ffffff",
                                    xmldecl=False, svgns=True, svgclass="card-qr", omitsize=True)
    return out.getvalue().decode("utf-8")


def last_share(patient_id):
    """When the patient last shared their card, if within SHARE_HOURS."""
    since = datetime.utcnow() - timedelta(hours=SHARE_HOURS)
    m = (Message.query.join(Session, Message.session_id == Session.id)
         .filter(Session.patient_id == patient_id, Message.sender_id == patient_id,
                 Message.direction == "patient_to_doctor", Message.source == "card",
                 Message.created_at >= since)
         .order_by(Message.created_at.desc()).first())
    return m.created_at if m else None


def grant_from_scan(patient_id):
    grants = dict(browser_session.get(_GRANTS) or {})
    now = datetime.utcnow().timestamp()
    grants = {k: v for k, v in grants.items() if v > now}
    grants[str(patient_id)] = now + QR_HOURS * 3600
    browser_session[_GRANTS] = grants


def scanned(patient_id):
    grants = browser_session.get(_GRANTS) or {}
    return grants.get(str(patient_id), 0) > datetime.utcnow().timestamp()


def can_view_history(patient_id):
    return last_share(patient_id) is not None or scanned(patient_id)
