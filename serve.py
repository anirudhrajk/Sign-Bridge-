"""Run SignBridge with a production web server (Waitress; works on Windows, macOS and Linux).

  python serve.py                 # http://0.0.0.0:8000
  python serve.py --port 8080

Before real use:
  * set SECRET_KEY in .env to a long random value (python -c "import secrets;print(secrets.token_hex(32))")
  * serve it over HTTPS. Browsers only allow the camera on https:// (or on localhost), and
    Google sign-in needs an https:// callback outside localhost. See DEPLOY.md.

A watchdog checks /healthz every 30 s. If the app stops answering three times in a row (every
worker stuck, or the database gone), it prints what each worker was doing and exits, so the
host (Render, Docker) starts a fresh copy instead of leaving the site frozen.
"""
import argparse
import os
import sys
import threading
import time
import traceback
import urllib.request

from waitress import serve

from app import app

CHECK_EVERY = 30   # seconds between health checks
WAIT = 10          # seconds to wait for an answer
STRIKES = 3        # failed checks in a row before restarting
STARTUP = 60       # seconds to let the server start before the first check


def dump_threads():
    """Print every thread's stack, so the log shows what the stuck workers were waiting on."""
    names = {t.ident: t.name for t in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        print(f"--- thread {names.get(ident, ident)} ---", flush=True)
        print("".join(traceback.format_stack(frame)), flush=True)


def watchdog(port):
    url = f"http://127.0.0.1:{port}/healthz"
    time.sleep(STARTUP)
    strikes = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=WAIT) as r:
                ok = r.status == 200
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"watchdog: health check failed: {e!r}", flush=True)
        strikes = 0 if ok else strikes + 1
        if strikes >= STRIKES:
            print(f"watchdog: no healthy answer {STRIKES} times in a row; restarting. Stuck workers:", flush=True)
            dump_threads()
            os._exit(1)
        time.sleep(CHECK_EVERY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    if app.config["SECRET_KEY"] in ("dev-secret-change-me", "change-me-in-production", ""):
        sys.exit("Refusing to start: set SECRET_KEY in .env to a long random value first (see DEPLOY.md).")
    if os.environ.get("WATCHDOG", "1") != "0":
        threading.Thread(target=watchdog, args=(args.port,), name="watchdog", daemon=True).start()
    print(f"SignBridge running on http://{args.host}:{args.port}  (put HTTPS in front for camera access)", flush=True)
    serve(app, host=args.host, port=args.port, threads=args.threads, channel_timeout=60)


if __name__ == "__main__":
    main()
