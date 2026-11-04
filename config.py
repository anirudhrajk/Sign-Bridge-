import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///signbridge.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Set HTTPS=1 when the app is served over https:// (cookies are then never sent over http).
    SESSION_COOKIE_SECURE = REMEMBER_COOKIE_SECURE = os.environ.get("HTTPS") == "1"
    # Set TRUST_PROXY=1 behind a reverse proxy (Caddy, nginx, a cloud host) so https links are right.
    TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    MESSAGE_RETENTION_DAYS = int(os.environ.get("MESSAGE_RETENTION_DAYS", 90))
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    # Optional: force the exact callback URL registered in Google Cloud Console.
    GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")