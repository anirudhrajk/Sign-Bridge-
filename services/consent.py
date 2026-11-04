from extensions import db
from models import ConsentRecord
from datetime import datetime


def has_consent(user_id, purpose):
    rec = ConsentRecord.query.filter_by(user_id=user_id, purpose=purpose, granted=True, revoked_at=None).first()
    return rec is not None


def grant_consent(user_id, purpose, version="1.0"):
    rec = ConsentRecord(user_id=user_id, purpose=purpose, granted=True, granted_at=datetime.utcnow(), version=version)
    db.session.add(rec)
    db.session.commit()


def revoke_consent(user_id, purpose):
    rec = ConsentRecord.query.filter_by(user_id=user_id, purpose=purpose, granted=True, revoked_at=None).first()
    if rec:
        rec.revoked_at = datetime.utcnow()
        rec.granted = False
        db.session.commit()