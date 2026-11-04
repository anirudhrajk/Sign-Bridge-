"""Run SignBridge with a production web server (Waitress; works on Windows, macOS and Linux).

  python serve.py                 # http://0.0.0.0:8000
  python serve.py --port 8080

Before real use:
  * set SECRET_KEY in .env to a long random value (python -c "import secrets;print(secrets.token_hex(32))")
  * serve it over HTTPS. Browsers only allow the camera on https:// (or on localhost), and
    Google sign-in needs an https:// callback outside localhost. See DEPLOY.md.
"""
import argparse
import os
import sys

from waitress import serve

from app import app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if app.config["SECRET_KEY"] in ("dev-secret-change-me", "change-me-in-production", ""):
        sys.exit("Refusing to start: set SECRET_KEY in .env to a long random value first (see DEPLOY.md).")
    print(f"SignBridge running on http://{args.host}:{args.port}  (put HTTPS in front for camera access)")
    serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
