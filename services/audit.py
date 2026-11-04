from extensions import db
from models import AuditLog
from flask_login import current_user
from flask import request


def log_action(action, detail=""):
    try:
        log = AuditLog(
            user_id=current_user.id if current_user.is_authenticated else None,
            action=action,
            detail=detail,
            ip_address=request.remote_addr if request else None,
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()