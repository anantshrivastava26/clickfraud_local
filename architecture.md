# Architecture

How the Click Fraud Monitor is put together, why it is built that way, and where it is rough. For
setup and day-to-day commands see the [README](README.md); this document explains the design.

## 1. Overview

The system flags **bot-like traffic** in ad click data without any fraud labels. Two ideas drive it:

1. **Profile each device, then look for outliers.** An impression-level click log is collapsed into
   one row of behavioural features per device (volume, click-through rate, how many different sites
   and apps it touched, ...). An **Isolation Forest**, trained once offline, learns what the bulk of
   those profiles look like; devices that are easy to isolate are suspicious.
2. **Score against the training baseline, never against the upload.** A device gets the same score
   whatever else is in the file, so results are comparable between uploads.

There are two deployable services and one offline job:

| Piece | What it is |
| --- | --- |
| **backend** | FastAPI service. Loads the trained model at startup and scores uploaded CSVs. |
| **frontend** | React + TypeScript single-page app (built with Vite, served by nginx). Shows the model baseline and scan results. |
| **training** | `backend/train.py`, run by hand. Produces the model files that are committed under `backend/artifacts/` and baked into the backend image. |

## 2. System diagram

```text
  OFFLINE (by hand)
  ───────────────────────────────────────────────────────────────────────────
   Avazu click log ─► backend/train.py ─► backend/artifacts/
                       │  app.features      ├─ scaler.joblib          {scaler, features}
                       │  (same code as     ├─ isoforest_model.joblib
                       │   the API uses)    ├─ model_meta.json        provenance for the UI
                       │                    └─ session_features.csv   training table (51,052 rows)

  RUNTIME
  ───────────────────────────────────────────────────────────────────────────
   Browser ──► nginx (frontend container, :80)
                 │  /            static React app (index.html fallback for client routes)
                 │  /api/*  ───► FastAPI (backend container, :8000, not published to the host)
                 │                 │
                 │                 ├─ GET  /api/health
                 │                 ├─ GET  /api/model   ◄── precomputed baseline (Detector.summary)
                 │                 └─ POST /api/scan
                 │                       │ pandas.read_csv
                 │                       ▼
                 │                  Detector.scan ─► features ─► StandardScaler ─► IsolationForest
                 │                       │                                             │
                 │                       └──────── verdict · risk 0-100 · signals ◄────┘
                 ◄─────────────────────── JSON (counts, histogram, per-row results)

  DELIVERY
  ───────────────────────────────────────────────────────────────────────────
   git push to main ─► GitHub Actions ─► build both images ─► ECR (tag = commit SHA)
                                          │
                     OIDC (no stored keys)└─► SSM Run Command ─► EC2: deploy/ec2/run.sh ─► smoke test
```

## 3. Repository layout

| Path | Role |
| --- | --- |
| `backend/app/main.py` | HTTP layer: routes, upload validation, status codes, CORS. No modelling. |
| `backend/app/detector.py` | `Detector`: loads artifacts, scores frames, builds verdicts, risk, signals, histogram and the training baseline. |
| `backend/app/features.py` | Feature engineering shared by training and scoring; input-format detection and validation. |
| `backend/train.py` | Builds features (optionally from a raw log), fits scaler and model, writes `artifacts/`. |
| `backend/make_samples.py` | Regenerates the sample CSVs served by the frontend. |
| `backend/artifacts/` | Trained model, scaler, metadata and the training feature table. |
| `backend/data/` | Hand-written demo sessions used by `make_samples.py`. |
| `frontend/src/` | React app: `App.tsx` (shell, shared state), `pages/`, `components/`, `api.ts`, `types.ts`, `format.ts`, `styles.css`. |
| `frontend/nginx.conf.template` | Production web server: static files, `/api` proxy, SPA fallback. |
| `docker-compose.yml` | Local production-like stack (frontend on :8080). |
| `deploy/ec2/` | Shell scripts for the EC2 host: bootstrap, run, manual deploy, one-time CI/CD setup. |
| `infra/github-deploy-role.yaml` | CloudFormation: ECR repos, EC2 instance role, GitHub OIDC deploy role. |
| `.github/workflows/ci-cd.yml` | CI (tests, build) and CD (deploy to EC2). |

## 4. Backend

### 4.1 Request flow for `POST /api/scan`

1. **`main.scan`** checks the extension (`.csv`/`.txt` → else 415), reads at most
   `MAX_UPLOAD_MB` + 1 byte (larger → 413), rejects empty content (400) and parses with pandas
   (failure → 400).
2. **`Detector.scan`** normalises column names, then detects the format from the header:
   - **Raw click log** (has `click` and `device_id` or `device_ip`) → `aggregate_click_log`
     groups impressions by device and builds the features.
   - **Session features** (anything else) → `prepare_session_features` checks that every column
     the model was trained on is present (non-numeric cells become 0, with a warning). If columns
     are missing it raises `ValueError`, which `main.py` returns as **422** with a message naming
     what is missing.
3. **`_score_frame`** selects the model's columns in training order, standardises them, takes the
   IsolationForest decision score and assigns a verdict and a risk score.
4. The response is assembled: `counts`, a `histogram` of scores, up to 5,000 `results` (most
   anomalous first) and a `truncated` flag.

Handlers are plain `def` (not `async`): pandas and scikit-learn block the CPU, so FastAPI runs them
in its thread pool and one slow scan cannot stall health checks.

### 4.2 Features

Fifteen features per device or session (`features.py`, `SESSION_FEATURES`):

| Feature | Meaning |
| --- | --- |
| `impressions` | Rows in the log for this device. |
| `clicks`, `click_rate` | Sum of the `click` flag, and clicks / impressions. |
| `unique_sites`, `unique_apps` | Distinct `site_id` / `app_id` values. |
| `device_type`, `device_conn_type` | Most frequent value; `-1` if unknown. |
| `c14_nunique` … `c21_nunique` | Distinct values of Avazu's anonymised `C14`–`C21` columns (8 features). |

Training and scoring call the **same function**, so the model always sees identically built
features. If an optional column is missing from an upload, the feature defaults to *one distinct
value* (the typical case in training) and a warning is returned, so absent data is not mistaken for
suspicious behaviour.

### 4.3 Verdict, risk and signals

`IsolationForest.decision_function` is higher for normal points; **negative means anomalous** and
`0` is the model's decision boundary (placed by `contamination=0.02`).

| Verdict | Rule | Risk band |
| --- | --- | --- |
| **bot** | the model's own decision: score < 0 | 70 – 100 |
| **review** | not a bot, but in the riskiest 5 % of *training* scores (`REVIEW_QUANTILE`) | 40 – 69 |
| **human** | everything else (it only means "not flagged") | 0 – 39 |

- **Risk** is a piecewise-linear rescaling of the score through the points
  *(lowest training score → 100)*, *(0 → 70)*, *(review threshold → 40)*, *(highest training score
  → 0)*, so the verdict cut-offs land on round numbers. Scores outside the training range clamp to
  100 or 0. This needs `min < 0 < review threshold`, which holds only while `REVIEW_QUANTILE` is
  larger than `contamination`.
- **Signals** explain a row: the (up to) three features whose standardised value is furthest from
  the training mean, if at least 2.5 standard deviations away. They are read from the same
  `StandardScaler` output the model sees, so no extra computation is needed.
- **Histogram** buckets use fixed edges taken from the training score range, so every scan is drawn
  on the same axis.

### 4.4 Startup and the training baseline

`Detector.__init__` (run once in FastAPI's `lifespan`) loads the three model files, then **re-scores
the whole training table** to derive the review threshold, risk anchors, histogram edges, feature
medians and the baseline served by `GET /api/model`. Recomputing instead of storing guarantees the
baseline always matches the loaded model. A missing or incompatible artifact fails the deployment at
boot rather than on the first request.

### 4.5 Training (`backend/train.py`)

1. Optionally build features from a raw log in chunks (`--clicks`, default 200,000 rows per chunk).
2. Keep only `SESSION_FEATURES`, drop zero-variance columns, fit `StandardScaler`.
3. Fit `IsolationForest(n_estimators=200, contamination=0.02, random_state=42)`.
4. Write `scaler.joblib` as `{"scaler", "features"}` (the feature list is the contract with the
   detector), `isoforest_model.joblib`, and `model_meta.json` (training time, scikit-learn version,
   source, row count; display only).

## 5. API reference

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/health` | `{"status": "ok", "model_loaded": bool}` |
| GET | `/api/model` | `model` (algorithm, trees, contamination, thresholds, risk bands, features, provenance), `counts`, `histogram`, `profile` (human vs bot medians and p95 per feature), `top_anomalies` (25 rows) |
| POST | `/api/scan` | `file`, `mode` (`click_log` or `sessions`), `input_rows`, `warnings[]`, `counts`, `histogram`, `results[]`, `truncated` |

A result row: `id`, `verdict`, `risk`, `score`, `impressions`, `clicks`, `click_rate`,
`signals[] {feature, label, value, typical, z}`, `features {name: value}`.

`/api/scan` errors: **415** wrong extension, **413** larger than `MAX_UPLOAD_MB` (default 50),
**400** empty or unparseable, **422** readable but unrecognised format. The message is in
`detail` and is shown verbatim by the UI. Interactive docs: `/docs`.

## 6. Frontend

- **Routing and state.** `App.tsx` renders a sidebar and two routes: `/` (**Model baseline**) and
  `/scan` (**Scan traffic**). A React context above the router holds the model summary (fetched
  once) and the latest scan result, so navigating away and back keeps the scan. `/api/health` is
  polled every 30 s for the status dot.
- **Data access.** All requests go through `api.ts`, which turns failures into readable `Error`
  messages (network down, or the backend's `detail`). `types.ts` mirrors the backend JSON by hand.
- **Scan page.** Upload by drag-and-drop or file picker, or scan one of three bundled samples
  (fetched as static files and sent through the same upload path). Results show counts, a score
  histogram and a table, with an export-to-CSV button.
- **Results table.** Filtering, sorting and paging all run in the browser over the (at most 5,000)
  returned rows. Selecting a row opens a drawer with the risk, the signals and every feature value
  beside the training medians.
- **Charts** are hand-drawn SVG: a log-scaled stacked histogram with threshold markers and tooltip,
  and a stacked proportion bar. No chart library.
- **Accessibility and theming.** A verdict is always icon + colour + text. Colours come from CSS
  variables with a dark theme that follows the OS setting.
- **API location.** By default requests go to the same origin: Vite's dev server proxies `/api` to
  `localhost:8000` and nginx proxies it in containers. `VITE_API_URL` (a build argument) points the
  browser at a backend directly instead.

## 7. Deployment

### 7.1 Containers

- **Backend image:** `python:3.12-slim`, dependencies installed in their own layer, `app/`,
  `artifacts/` and `train.py` copied in. Listens on `$PORT` (default 8000). Because artifacts are
  baked in, **retraining requires rebuilding and redeploying the image**.
- **Frontend image:** multi-stage. Node builds the static site; nginx serves it, proxies `/api/` to
  `BACKEND_URL` (substituted into the config template at start) and falls back to `index.html` for
  client-side routes. `client_max_body_size` (50 MB) must stay at least `MAX_UPLOAD_MB`.
- **`docker-compose.yml`** runs both; the frontend waits for the backend health check.

### 7.2 Production (EC2)

One `t3.micro` in `ap-south-1` with an Elastic IP. `deploy/ec2/user-data.sh` installs Docker and a
2 GB swap file (image builds need more than 1 GB of memory). `deploy/ec2/run.sh` creates a
`clickfraud-net` network and starts `clickfraud-backend` (no published port) and
`clickfraud-frontend` (port 80), replacing any running containers. It either pulls images from ECR
(when `BACKEND_IMAGE`/`FRONTEND_IMAGE` are set, as in CI) or builds from source (manual deploys via
`deploy.sh`).

### 7.3 CI/CD (`.github/workflows/ci-cd.yml`)

| Job | Runs on | What it does |
| --- | --- | --- |
| Backend tests | every push / PR | Starts the real app in-process, checks health and `/api/model`, scans the demo click log (**must flag exactly 6 bots**) and checks that a malformed file gives 422. |
| Frontend build | every push / PR | `npm ci` and `npm run build` (type-check + bundle). |
| Deploy | push to `main` only, after the two above, and only once bootstrap has run | Assume the AWS role via OIDC → build and push both images to ECR (tag = commit SHA and `latest`) → run `run.sh` on the instance through SSM → smoke-test the public URL. |

### 7.4 Access model

- **No stored AWS keys.** GitHub issues an OIDC token; AWS exchanges it for short-lived credentials
  on `clickfraud-monitor-github-deploy`. Its trust policy accepts only this repository's
  `production` environment.
- **Least privilege.** The deploy role can push to the two ECR repositories and send shell commands
  to the one instance; the instance role can use Systems Manager and pull from ECR.
- **No inbound SSH.** Rollouts go through SSM Run Command.
- `deploy/ec2/setup-cicd.sh` is the one-time bootstrap (run with admin credentials): it deploys
  `infra/github-deploy-role.yaml`, attaches the instance profile and sets the GitHub environment,
  secret and variables. The template assumes the account already has the GitHub OIDC identity
  provider.

## 8. Design decisions

| Decision | Why | Consequence |
| --- | --- | --- |
| Unsupervised Isolation Forest | Click logs have no fraud labels. | No precision/recall; verdicts are anomaly judgements relative to the training data. `contamination` is an assumption. |
| Device as the unit of analysis | Avazu has no session id and only hour-granular timestamps. | No time dimension: features are counts, ratios and cardinalities. |
| Same feature code for training and scoring | Prevents train/serve drift. | A change to `features.py` means retraining. |
| Feature list saved with the scaler | The detector rebuilds exactly the trained columns in the trained order. | Model files must be regenerated together. |
| Re-score the training table at startup | The baseline always matches the loaded model. | A short startup delay (51,052 rows). |
| Score against the training distribution | Stable, comparable results between uploads. | An upload cannot move the thresholds. |
| Three verdicts (bot / review / human) | Gives analysts a middle band instead of a hard cut at 2 %. | "Human" only means "not flagged". |
| Missing optional columns default to a neutral value + warning | Partial files still scan without false alarms. | Results are less informative; the warning says why. |
| Hand-rolled SVG charts | One chart does not justify a library. | Chart code is maintained in-repo. |
| Browser-side filter, sort and paging | Instant interaction. | The API caps results at 5,000 rows; counts and histogram still cover the whole file. |
| Same-origin `/api` via proxy | No CORS configuration in practice. | Needs the nginx / Vite proxy. |
| Artifacts baked into the image | Immutable, reproducible deploys. | Retraining means a new image. |
| SSM instead of SSH for deploys | No open port 22, no key in CI. | Requires the instance role and the SSM agent. |

## 9. Limitations and risks

1. **Shared placeholder ids.** Avazu gives many anonymous users one `device_id`. Those rows collapse
   into a few very large "devices" that the model flags (about 91 % of training clicks sit in the 2 %
   flagged sessions). The flags are anomaly judgements, not proof of fraud.
2. **Chunked training.** `build_features_from_clicks` aggregates each chunk separately, so a device
   spanning chunks appears as several partial rows.
3. **No labels.** There is no accuracy figure. The demo click log is synthetic, with six scripted bot
   devices; CI and the post-deploy check assert that exactly those six are flagged, which ties
   `make_samples.py`, the model and the CI assertions together.
4. **Risk mapping assumption.** `REVIEW_QUANTILE` must stay above `contamination` (see §4.3).
5. **Unauthenticated API, permissive CORS.** `ALLOWED_ORIGINS` defaults to `*`, and the live site
   is served over plain HTTP on port 80. Uploads are processed in memory and not stored.
6. **Single instance.** Deploys replace the containers, so there is a brief gap in service on each
   rollout, and there is no automated rollback (images tagged by commit SHA make a manual one
   possible).
7. **Version coupling.** The `scikit-learn` pin in `backend/requirements.txt` must match
   `sklearn_version` in `model_meta.json`, because the joblib files are version-sensitive.
8. **Hand-maintained contracts.** `frontend/src/types.ts` mirrors the backend response and the sample
   descriptions in `ScanPage.tsx` quote `make_samples.py`'s output; nothing checks either
   automatically.

## 10. Extending it

- **Retrain:** `cd backend && python train.py [--clicks file.csv]`, then rebuild the image. Update the
  CI assertion on the demo log if the number of flagged bots changes.
- **Add a feature:** add it to `SESSION_FEATURES` and `aggregate_click_log`, add a label in
  `FEATURE_LABELS`, retrain, and update the README's feature description.
- **Real-time or sessionised detection:** needs timestamps finer than Avazu's hourly ones, a
  streaming ingestion path and a state store; the current design is batch and per-upload.
