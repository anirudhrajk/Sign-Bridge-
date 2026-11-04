import secrets
from datetime import datetime

from flask import Blueprint, render_template, redirect, url_for, request, flash, session, current_app
from flask_login import login_user, logout_user, login_required, current_user

from extensions import db, oauth
from models import User
from services.audit import log_action
from services.consent import grant_consent

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

ROLES = ("patient", "doctor")


def _safe_next(url):
    """Only follow local paths, so a crafted link cannot redirect off-site."""
    if not url or not url.startswith("/") or url.startswith(("//", "/\\")):
        return url_for("views.dashboard")
    return url


def _clinician_allowed(role):
    """Clinicians can read patient messages, so an optional clinic code guards that role."""
    code = current_app.config.get("CLINICIAN_CODE")
    if role == "doctor" and code and not secrets.compare_digest((request.form.get("clinic_code") or "").strip(), code):
        flash("That clinic code is not right. Ask your clinic administrator, or sign up as a patient.", "error")
        return False
    return True


def _finish_login(user, remember=True, method="password"):
    user.last_login = datetime.utcnow()
    db.session.commit()
    login_user(user, remember=remember)
    log_action("login", f"User {user.email} signed in with {method}")
    return redirect(_safe_next(session.pop("next_url", None) or request.args.get("next")))


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password) and user.is_active:
            return _finish_login(user, remember=bool(request.form.get("remember")))
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
        if role not in ROLES:
            role = "patient"
        if not _clinician_allowed(role):
            return render_template("register.html")
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


# ---------- Sign in with Google ----------
@auth_bp.route("/google")
def google_login():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    client = oauth.create_client("google")
    if client is None:
        flash("Google sign-in is not set up on this server yet. Use email for now.", "warning")
        return redirect(url_for("auth.login"))
    if request.args.get("next"):
        session["next_url"] = request.args["next"]
    redirect_uri = current_app.config["GOOGLE_REDIRECT_URI"] or url_for("auth.google_callback", _external=True)
    return client.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    client = oauth.create_client("google")
    if client is None:
        return redirect(url_for("auth.login"))
    try:
        token = client.authorize_access_token()
    except Exception:  # user cancelled, state mismatch, or network failure
        flash("Google sign-in was cancelled or failed. Please try again.", "error")
        return redirect(url_for("auth.login"))

    info = token.get("userinfo") or {}
    sub, email = info.get("sub"), (info.get("email") or "").lower()
    if not sub or not email or not info.get("email_verified"):
        flash("Your Google account needs a verified email address.", "error")
        return redirect(url_for("auth.login"))

    user = User.query.filter_by(google_sub=sub).first()
    if user is None:
        # Link an existing email account the first time its owner uses Google.
        user = User.query.filter_by(email=email).first()
        if user is not None:
            user.google_sub = sub
    if user is not None:
        if not user.is_active:
            flash("This account is disabled. Contact your administrator.", "error")
            return redirect(url_for("auth.login"))
        user.avatar_url = info.get("picture") or user.avatar_url
        return _finish_login(user, method="Google")

    # New person: ask whether they are a patient or clinician before creating the account.
    session["pending_google"] = {
        "sub": sub,
        "email": email,
        "name": info.get("name") or email.split("@")[0],
        "picture": info.get("picture"),
    }
    return redirect(url_for("auth.choose_role"))


@auth_bp.route("/welcome", methods=["GET", "POST"])
def choose_role():
    pending = session.get("pending_google")
    if not pending:
        return redirect(url_for("auth.login"))
    if request.method == "POST":
        role = request.form.get("role")
        if role not in ROLES:
            flash("Choose how you will use SignBridge.", "error")
            return render_template("choose_role.html", pending=pending)
        if not _clinician_allowed(role):
            return render_template("choose_role.html", pending=pending)
        if not request.form.get("agree"):
            flash("Please accept the privacy terms to continue.", "error")
            return render_template("choose_role.html", pending=pending)
        user = User(
            email=pending["email"],
            display_name=(request.form.get("display_name") or pending["name"]).strip()[:120] or pending["name"],
            role=role,
            google_sub=pending["sub"],
            avatar_url=pending.get("picture"),
        )
        # Google accounts never use this password; it only satisfies the column.
        user.set_password(secrets.token_urlsafe(32))
        db.session.add(user)
        db.session.commit()
        grant_consent(user.id, "data_storage")
        grant_consent(user.id, "camera_capture")
        session.pop("pending_google", None)
        log_action("register", f"New {role} registered with Google: {user.email}")
        return _finish_login(user, method="Google")
    return render_template("choose_role.html", pending=pending)


@auth_bp.route("/logout")
@login_required
def logout():
    log_action("logout", f"User {current_user.email} logged out")
    logout_user()
    return redirect(url_for("views.welcome"))
