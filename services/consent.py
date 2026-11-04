from extensions import db
from models import ConsentRecord
from datetime import datetime

CONSENT_PURPOSES = {
    "camera_capture": "Use the camera to read hand signs",
    "data_storage": "Store messages from my sessions",
}


def has_consent(user_id, purpose):
    rec = ConsentRecord.query.filter_by(user_id=user_id, purpose=purpose, granted=True, revoked_at=None).first()
    return rec is not None


def grant_consent(user_id, purpose, version="1.0"):
    if has_consent(user_id, purpose):
        return
    rec = ConsentRecord(user_id=user_id, purpose=purpose, granted=True, granted_at=datetime.utcnow(), version=version)
    db.session.add(rec)
    db.session.commit()


def revoke_consent(user_id, purpose):
    rec = ConsentRecord.query.filter_by(user_id=user_id, purpose=purpose, granted=True, revoked_at=None).first()
    if rec:
        rec.revoked_at = datetime.utcnow()
        rec.granted = False
        db.session.commit()


def consent_status(user_id):
    """Every purpose with its current state, including ones never answered."""
    rows = []
    for purpose, description in CONSENT_PURPOSES.items():
        rec = (ConsentRecord.query.filter_by(user_id=user_id, purpose=purpose, granted=True, revoked_at=None)
               .order_by(ConsentRecord.id.desc()).first())
        rows.append({"purpose": purpose, "description": description, "granted": rec is not None,
                     "granted_at": rec.granted_at if rec else None})
    return rows
