import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from flask.cli import with_appcontext
import click

from config import Config
from extensions import db, login_manager
import models  # noqa: F401


def seed_defaults():
    from models import User, Phrase, MedicalSign
    from services.emergency import get_default_phrases
    from services.dictionary import get_local_seed

    if User.query.count() == 0:
        patient = User(email="patient@signbridge.app", display_name="Demo Patient", role="patient")
        patient.set_password("patient123")
        doctor = User(email="doctor@signbridge.app", display_name="Dr. Demo Clinician", role="doctor")
        doctor.set_password("doctor123")
        admin = User(email="admin@signbridge.app", display_name="System Admin", role="admin")
        admin.set_password("admin123")
        db.session.add_all([patient, doctor, admin])
        db.session.commit()

    if Phrase.query.count() == 0:
        for p in get_default_phrases():
            db.session.add(Phrase(text=p["text"], category=p["category"], is_emergency=True))
        for text, cat in [
            ("Please take a seat", "general"),
            ("I will check your blood pressure", "procedure"),
            ("Do you have any allergies?", "intake"),
            ("Where does it hurt?", "intake"),
            ("Take this medicine twice a day", "medication"),
        ]:
            db.session.add(Phrase(text=text, category=cat))
        db.session.commit()

    if MedicalSign.query.count() == 0:
        for s in get_local_seed():
            db.session.add(MedicalSign(label=s["label"], category=s["category"], description=s["description"]))
        db.session.commit()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from auth import auth_bp
    from views import views_bp
    from api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("500.html"), 500

    @app.context_processor
    def inject_globals():
        return {"app_name": "SignBridge", "now": datetime.utcnow}

    @app.cli.command("init-db")
    @with_appcontext
    def init_db():
        db.create_all()
        seed_defaults()
        click.echo("Initialized database.")

    @app.cli.command("purge-old-messages")
    @with_appcontext
    def purge_old_messages():
        cutoff = datetime.utcnow() - timedelta(days=app.config["MESSAGE_RETENTION_DAYS"])
        deleted = Message.query.filter(Message.created_at < cutoff).delete()
        db.session.commit()
        click.echo(f"Purged {deleted} messages older than {app.config['MESSAGE_RETENTION_DAYS']} days.")

    with app.app_context():
        db.create_all()
        seed_defaults()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)