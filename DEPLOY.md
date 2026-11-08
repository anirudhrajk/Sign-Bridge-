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

**Set the health check.** In the web service's *Settings → Health Checks*, set the path to
`/healthz`. It answers only when the app has a free worker and the database replies, so the
host replaces a frozen instance. `serve.py` also runs its own watchdog: if `/healthz` fails three
times in a row (about 90 s), it writes what each worker was stuck on to the log and exits, and
the host starts a fresh copy. Set `WATCHDOG=0` to turn it off.

### Google Cloud Run (recommended host)

Cloud Run sleeps when nobody is using it and wakes in a few seconds on the next visit. Its
monthly free allowance is far more than a clinic needs. Accounts and chats live in a separate
PostgreSQL database (Neon), so they survive restarts.

**1. Database (Neon, free, does not expire)**
1. Sign up at neon.tech and create a project in the region closest to India.
2. On the project dashboard, choose *Connect*, turn **Connection pooling off**, and copy the
   connection string (`postgresql://...`). This is `DATABASE_URL`. Keep it secret: it holds the password.

**2. Google Cloud**
1. Sign in at console.cloud.google.com, create a project (e.g. `signbridge`) and link a billing
   account. A card is required even within the free tier.
2. *Billing → Budgets & alerts*: create a budget of e.g. ₹100 with email alerts.

**3. Create the service** (*Cloud Run → Deploy container → Service*)
1. Choose **Continuously deploy from a repository (GitHub)** → *Set up with Cloud Build*.
   Connect GitHub, pick `anirudhrajk/Sign-Bridge-`, branch `^deploy$`, build type **Dockerfile**
   (path `/Dockerfile`). Every push to `deploy` then redeploys automatically.
2. Name `signbridge`, region **asia-south1 (Mumbai)**.
3. Authentication: **Allow public access** (unauthenticated invocations).
4. Billing **Request-based**. Minimum instances **0**, maximum instances **2**.
5. *Containers → Settings*: port **8080**, memory **512 MiB**, CPU **1**, maximum concurrent
   requests per instance **16**.
6. *Containers → Variables & Secrets*: add
   - `DATABASE_URL`: the Neon connection string
   - `SECRET_KEY`: a long random value (`python -c "import secrets;print(secrets.token_hex(32))"`)
   - `CLINICIAN_CODE`: the clinician sign-up code, if you use one
7. *Containers → Health checks*: add a **liveness** probe, HTTP, path `/healthz`, period 30 s.
8. Create. The first build takes a few minutes; the service URL appears at the top
   (`https://signbridge-....asia-south1.run.app`).

On Cloud Run, `serve.py` turns off its own watchdog (Cloud Run pauses the CPU between requests),
and the liveness probe takes its place. Old container images accumulate in Artifact Registry. Add a
cleanup policy there (keep the 3 most recent) to stay inside the free 0.5 GB of storage.

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
