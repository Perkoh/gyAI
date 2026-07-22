# gyAI — AI-Powered Domain Intelligence System (ADIS)

> An informed user is a protected user.

**gyAI** is a real-time, notification-only browser security system. As you browse, it analyses each domain you visit with a machine-learning model and warns you when a site looks suspicious or malicious — **without ever blocking, redirecting, or interrupting your browsing.** The choice to continue always stays yours.

gyAI is the public name for the system; internally the project is referred to as **ADIS (Artificial Domain Intelligence System)**.

---

## Why gyAI exists

Most web-protection tools take control away from the user — they block pages, throw up full-screen interstitials, or silently reroute traffic. gyAI takes the opposite stance: it **informs rather than restricts.** When a domain scores as risky, gyAI injects a small, dismissible banner explaining *why* in plain English, then gets out of the way. Safe domains produce zero interruption.

It is built to be private (only the bare domain name ever leaves the browser — never full URLs, paths, or history), free to use (no account required), and fast (a Redis cache returns repeat lookups in milliseconds).

---

## How it works

When you navigate to a site, gyAI runs the domain through a multi-stage pipeline:

1. **Local cache check** — the extension keeps a short-lived local cache; known-recent domains resolve instantly.
2. **Server cache (Redis)** — on a miss, the API checks a shared Redis cache before doing any real work.
3. **Curated ground truth** — domains on a curated allowlist/blocklist short-circuit straight to a verdict.
4. **Feature extraction** — for everything else, the API computes **48 features** from the domain: 30 *structural/lexical* features (length, entropy, character patterns, brand-impersonation and typosquatting signals — instant, pure-Python) and 18 *network* features (domain age, DNS records, WHOIS data — with graceful fallback when a lookup times out).
5. **ML inference** — a **LightGBM** classifier scores the domain from 0 to 1 and assigns a label of *safe*, *suspicious*, or *malicious*.
6. **Explanation** — **SHAP** values identify the top contributing features, which are mapped to human-readable reasons ("This domain was registered 4 days ago, which is common for phishing sites").
7. **Notification** — the extension shows nothing for safe domains, a dismissible yellow *caution* banner for suspicious ones, and a red *alert* banner for malicious ones.

---

## The machine-learning model

The classifier was trained in two phases to avoid a subtle data-bias trap. Because most malicious domains die quickly, their live network features are often unavailable — which could teach a model the shortcut "no network data ⇒ malicious." To prevent this, the training set was filtered to **live** domains (verified to resolve via the same DNS library used in production), and the model was trained in two stages:

- **Phase 1** — pretrained on structural features alone across the full corpus.
- **Phase 2** — fine-tuned with network features enabled on the live subset, continuing from the phase-1 model.

**Performance of the fine-tuned model (v1.1.0), on a held-out test set:**

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.983 |
| F1 | 0.938 |
| Precision | 0.962 |
| Recall | 0.915 |
| Accuracy | 0.940 |

Training data was balanced across benign and malicious classes.

---

## Architecture

```
Browser Extension (Manifest V3, TypeScript)
        │   sends { "domain": "example.com" }  (domain only — privacy first)
        ▼
Flask API  (Gunicorn, deployed on Fly.io)
        │
        ├─ Redis cache (Upstash)      — fast repeat lookups, 24h network-feature cache
        ├─ Feature extraction         — 30 structural + 18 network features
        ├─ LightGBM + SHAP            — score, label, and human-readable reasons
        └─ Supabase (PostgreSQL)      — logs flagged domains + user feedback
```

The extension and the API are decoupled: the extension only ever sends a domain and renders the response. All analysis, caching, and logging happen server-side.

---

## Tech stack

**Backend / ML:** Python, Flask, Gunicorn, LightGBM, SHAP, scikit-learn, dnspython, python-whois, tldextract
**Infrastructure:** Fly.io (API hosting, always-on), Upstash Redis (caching), Supabase / PostgreSQL (logging)
**Extension:** TypeScript, Manifest V3, Webpack
**Design constraints:** free-tier-friendly, no paid third-party APIs, HTTPS-only, privacy-preserving

---

## Project structure

```
gyAI/
├── api/                  # Flask application (routes, middleware, schemas)
│   ├── app.py            # application factory
│   └── routes/           # analyze, feedback, health, admin
├── cache/                # Redis client
├── config/               # settings + constants
├── database/             # Supabase client
├── features/             # feature extraction (structural + network) + assembler
├── ml/                   # model server, training, and saved models
│   ├── models/           # trained LightGBM model + label encoder
│   └── training/         # feature builder, preprocess, train, finetune
├── extension/            # browser extension (TypeScript, Manifest V3)
├── scripts/              # data-prep and utility scripts
├── Dockerfile
├── fly.toml
└── gunicorn.conf.py
```

---

## Getting started

### Prerequisites

- Python 3.12
- Node.js ≥ 18 (for the extension build)
- A Redis instance (Upstash) and a Supabase project

### Backend

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env                 # then fill in SUPABASE_URL, SUPABASE_KEY,
                                     # REDIS_URL, REDIS_SSL, API keys

# 4. Run locally
python -m api.app                    # dev server on :8080
# or, production-style:
gunicorn -c gunicorn.conf.py "api.app:create_app()"
```

### Browser extension

```bash
cd extension
npm install
npm run build                        # compiles TypeScript into dist/
```

Then load it in Chrome: `chrome://extensions` → enable **Developer Mode** → **Load unpacked** → select the `extension/dist` folder.

---

## API

Base URL: `https://gyai-api.fly.dev/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Score a single domain. Public. |
| `POST` | `/analyze/bulk` | Score up to 50 domains. Requires an API key. |
| `POST` | `/feedback` | Submit a false-positive / confirmation report. |
| `GET`  | `/health` | Liveness and model-status probe. |
| `GET`  | `/version` | Current model version. |

**Example:**

```bash
curl -X POST https://gyai-api.fly.dev/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"domain": "example.com"}'
```

```json
{
  "domain": "example.com",
  "score": 0.04,
  "label": "safe",
  "confidence": "high",
  "reasons": [],
  "model_version": "v1.1.0",
  "cached": false,
  "network_features_used": true
}
```

---

## Privacy

gyAI is designed to protect through awareness, not surveillance:

- Only the **bare domain name** is ever sent to the API — never full URLs, paths, query strings, or cookies.
- **No account** is required to use the extension.
- **No browsing history** is stored. Only domains scored as suspicious or malicious are logged server-side (for model improvement), never safe browsing.

---

## Roadmap

- Threat-intelligence feed ingestion (PhishTank / OpenPhish / URLhaus auto-sync)
- Firefox (WebExtensions) build
- RDAP fallback for TLDs deprecating legacy WHOIS
- Public API documentation portal

---

## Acknowledgements

Built on open threat-intelligence sources and open-source tooling. gyAI calls no paid third-party APIs — the intelligence comes from the model, not a subscription.
