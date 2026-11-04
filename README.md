# SignBridge

Clear conversations between deaf patients and their doctors.

- **Sign → text.** The patient signs to the camera. The phone or laptop tracks their body
  and hands (MediaPipe Holistic) and a small on-device model reads the sign. No video leaves
  the device.
- **Speech/text → sign.** The doctor speaks or types. A signing avatar, driven by real
  recorded sign movements, signs it back, with each word captioned.
- **Built for the clinic.** "Call patient in" alerts, understanding checks on every doctor
  message, a body map for pain, a Deaf patient card, interpreter requests, emergency phrases,
  and a printable visit summary.
- **Works offline.** Face-to-face mode (`/together`) needs no internet and no server. The
  app installs on phones like a native app.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Demo accounts: `patient@signbridge.app` / `patient123`,
`doctor@signbridge.app` / `doctor123`. To enable **Sign in with Google**, add
`GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to `.env` (see `.env.example`).

## Sign languages

| Code | Language | Recognition | Avatar | Data |
|------|----------|-------------|--------|------|
| `isl` | Indian Sign Language | 262 signs | 262 signs | AI4Bharat INCLUDE (landmarks) |
| `asl` | American Sign Language | 250 signs | 250 signs | Google Isolated Sign Language Recognition (Kaggle) |

Patients choose their sign language in Settings. Adding a language is a data and training job,
not a code change. See [ml/README.md](ml/README.md).

## Project layout

```
app.py, views.py, api.py, auth.py   Flask app, pages, JSON API, sign-in (email + Google)
templates/, static/css, static/js   Interface (design system in static/css/app.css)
static/js/sign-model.js             In-browser sign recognition
static/js/avatar.js, sign-planner.js  Signing avatar and English-to-sign planner
static/sw.js                        Offline support (service worker)
static/models/<lang>/               Trained recognition models
static/signs/<lang>/                Avatar sign library
ml/                                 Data prep, training, export checks, avatar clip builder
serve.py, Dockerfile, DEPLOY.md     Production server and deployment
```

## Honest limits

- Recognition reads one sign at a time and knows a few hundred everyday words, few of them
  medical. The patient always checks a message before it is sent.
- The avatar signs word by word in English order, not full sign-language grammar.
- SignBridge is a communication aid, not a replacement for a qualified interpreter. For
  consent, diagnosis and discharge, book an interpreter.
