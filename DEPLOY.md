# Deploying SignBridge

SignBridge is a web app that installs like a native app (a Progressive Web App). Sign
recognition and the signing avatar run on the user's device, so the server only stores
accounts and messages.

## 1. Run it on your computer (development)

```bash
pip install -r requirements.txt
python app.py              # http://localhost:5000
```

## 2. Run it for a clinic (production)

1. Create `.env` from `.env.example` and set a real secret key:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
2. Start the production server:
   ```bash
   python serve.py --port 8000
   ```
3. **Put HTTPS in front of it.** Browsers only allow the camera on `https://` or on
   `localhost`, so phones and tablets on the clinic Wi-Fi need HTTPS. The simplest option is
   [Caddy](https://caddyserver.com): `caddy reverse-proxy --from signbridge.yourclinic.in --to localhost:8000`
   gets a free certificate automatically. Then set `HTTPS=1` and `TRUST_PROXY=1` in `.env`.

### Cloud hosting

The included `Dockerfile` runs on any container host (Render, Railway, Fly.io, Google Cloud
Run, AWS). The host provides HTTPS. Set `SECRET_KEY`, `HTTPS=1`, and the Google keys as
environment variables.

**Set `DATABASE_URL` to a PostgreSQL database.** Without it, SignBridge keeps accounts and
conversations in a SQLite file inside the container, and cloud hosts wipe that file on every
deploy and restart: people who signed up cannot log in again and their chats are gone.

1. Create a PostgreSQL database: on Render (*New → Postgres*, same region as the web
   service), or on Neon or Supabase.
2. Copy its connection string (on Render, the *Internal Database URL*). It starts with
   `postgres://` or `postgresql://`; both work.
3. In the web service's *Environment*, add `DATABASE_URL` with that value and save. The
   service redeploys and creates its tables on first start.

Keep the connection string secret: it is the database's password.

### Google sign-in in production

In Google Cloud Console → Credentials → your OAuth client, add the production callback:
`https://<your-domain>/auth/google/callback`. Then publish the OAuth consent screen
(move it from "Testing" to "In production") so anyone can sign in.

## 3. Offline use

- After the first visit, the service worker (`static/sw.js`) keeps the app files, the sign
  model, the hand tracker and the avatar's sign library on the device, about 45 MB.
- **Face-to-face mode** (`/together`) works with no internet and no server. One device sits
  between patient and doctor. Nothing leaves the device.
- Clinic conversations (patient and doctor on separate devices) need the server, but not the
  internet: a laptop running `serve.py` on the clinic's local network is enough.
- Typing works offline everywhere. The doctor's *spoken* input uses the browser's speech
  service, which needs internet in most browsers.

## 4. Install it as an app

Once SignBridge is served over HTTPS:

- **Android / Chrome / Edge:** open the site, then choose *Install app* or *Add to Home screen*.
- **iPhone / iPad (Safari):** tap Share → *Add to Home Screen*.

## 5. Publish to the app stores

1. Host SignBridge on a public HTTPS domain (step 2).
2. Go to **pwabuilder.com**, enter your URL, and download the generated packages:
   - **Google Play:** an Android package (Trusted Web Activity). Upload the `assetlinks.json`
     it gives you to `static/.well-known/` and serve it at `/.well-known/assetlinks.json`.
     A Play developer account costs a one-time US$25.
   - **Apple App Store:** an Xcode project. This needs a Mac and an Apple Developer account
     (US$99 per year).
3. Store listings for medical apps need a privacy policy URL and clear wording that
   SignBridge is a communication aid, not a medical device.

## 6. Updating models

```bash
python ml/prepare_data.py --lang isl && python ml/train.py --lang isl   # recognition
python ml/build_sign_clips.py --lang isl                                 # avatar signs
```

Restart the server. The service worker notices the changed files and refreshes devices on
their next visit.
