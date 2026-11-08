import os
from dotenv import load_dotenv

load_dotenv()


def _database_url(url):
    """The database to use. Hosts hand out postgres:// or postgresql:// links; SQLAlchemy needs
    postgresql+psycopg:// to use the psycopg 3 driver. With no link, a local SQLite file."""
    url = (url or "").strip()
    # Forgive a whole .env line or quotes pasted into the host's settings.
    if url.startswith("DATABASE_URL="):
        url = url[len("DATABASE_URL="):].strip()
    url = url.strip("\"'").strip()
    if not url:
        return "sqlite:///signbridge.db"
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    # Accounts, chats and visits live here. Online, set DATABASE_URL to a PostgreSQL database
    # (Render Postgres, Neon, Supabase...): a SQLite file inside a cloud container is wiped on every
    # deploy and restart, which loses every account and conversation.
    SQLALCHEMY_DATABASE_URI = _database_url(os.environ.get("DATABASE_URL", ""))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Cloud databases close idle connections: check each one before use, and renew them every 5 minutes.
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 300, "pool_timeout": 10}
    if SQLALCHEMY_DATABASE_URI.startswith("postgresql"):
        # Never wait forever on the database: give up connecting after 10 s, notice a dead network
        # link within ~15 s (keepalives, unacknowledged sends), and stop any query after 15 s.
        SQLALCHEMY_ENGINE_OPTIONS["connect_args"] = {
            "connect_timeout": 10,
            "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3,
            "tcp_user_timeout": 15000,
            "options": "-c statement_timeout=15000 -c idle_in_transaction_session_timeout=30000",
        }
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Set HTTPS=1 when the app is served over https:// (cookies are then never sent over http).
    SESSION_COOKIE_SECURE = REMEMBER_COOKIE_SECURE = os.environ.get("HTTPS") == "1"
    # Set TRUST_PROXY=1 behind a reverse proxy (Caddy, nginx, a cloud host) so https links are right.
    TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"
    # Demo accounts have public passwords: keep them for local testing only (DEMO_ACCOUNTS=0 online).
    DEMO_ACCOUNTS = os.environ.get("DEMO_ACCOUNTS", "1") == "1"
    # If set, signing up as a clinician needs this code, so strangers cannot read patient messages.
    CLINICIAN_CODE = os.environ.get("CLINICIAN_CODE", "")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    # Video calls: how the two browsers find each other (a JSON list of WebRTC ICE servers).
    # Public STUN works on most networks; add a TURN server here for strict hospital networks.
    ICE_SERVERS = os.environ.get("ICE_SERVERS", '[{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]')
    MESSAGE_RETENTION_DAYS = int(os.environ.get("MESSAGE_RETENTION_DAYS", 90))
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    # Optional: force the exact callback URL registered in Google Cloud Console.
    GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")