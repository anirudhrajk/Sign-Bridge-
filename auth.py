from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from extensions import db
from models import User, ConsentRecord
from services.audit import log_action
from services.consent import grant_consent
from datetime import datetime

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password) and user.is_active:
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=bool(request.form.get("remember")))
            log_action("login", f"User {user.email} logged in")
            next_url = request.args.get("next") or ""
            # Only follow local paths, so a crafted link cannot redirect off-site.
            if not next_url.startswith("/") or next_url.startswith(("//", "/\\")):
                next_url = url_for("views.dashboard")
            return redirect(next_url)
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        name = (request.form.get("display_name") or "").strip()
        password = request.form.get("password") or ""
        role = request.form.get("role", "patient")
        if role not in ("patient", "doctor"):
            role = "patient"
        if not email or not name or len(password) < 8:
            flash("Provide a name, email, and a password of at least 8 characters.", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("register.html")
        user = User(email=email, display_name=name, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        # DPDP consent
        grant_consent(user.id, "data_storage")
        grant_consent(user.id, "camera_capture")
        log_action("register", f"New {role} registered: {email}")
        login_user(user)
        return redirect(url_for("views.dashboard"))
    return render_template("register.html")


@auth_bp.route("/logout")
@login_required
def logout():
    log_action("logout", f"User {current_user.email} logged out")
    logout_user()
    return redirect(url_for("views.welcome"))