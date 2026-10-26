# Click Fraud Detection

Unsupervised bot-traffic detection for ad click logs. An Isolation Forest is trained on device-level behavioural
features built from the [Avazu CTR dataset](https://www.kaggle.com/c/avazu-ctr-prediction). A FastAPI service scores
uploaded CSVs against it, and a React dashboard shows the trained baseline and the scan results.

```
frontend/  React + TypeScript (Vite), served by nginx in production
backend/   FastAPI scoring API, feature engineering, training script, model artifacts
```

## How it works

1. **Feature engineering** (`backend/app/features.py`). Impression-level rows are grouped by `device_id` and reduced
   to 15 behavioural features: impressions, clicks, click-through rate, distinct sites and apps, dominant device and
   connection type, and the number of distinct values of the anonymised `C14`–`C21` columns. Training and scoring share
   this code.
2. **Training** (`backend/train.py`). Constant columns are dropped, features are standardised, and an
   `IsolationForest` (200 trees, 2% contamination) is fit on 51,052 device sessions built from 500,000 Avazu
   impressions. 1,021 sessions (2.00%) are flagged.
3. **Scoring** (`backend/app/detector.py`). Each uploaded device or session gets:
   - a **verdict**. *Bot* means the model's decision score is below 0. *Review* means it isn't flagged but falls in the
     riskiest 5% of the training distribution. Everything else is *Human*.
   - a **risk score** from 0 to 100, mapped from the decision score so that Bot ≥ 70 and Review ≥ 40.
   - up to three **signals**: the features furthest from the training mean, in standard deviations.

## API

| Method | Path          | Description                                                                 |
| ------ | ------------- | --------------------------------------------------------------------------- |
| GET    | `/api/health` | Liveness and model status                                                   |
| GET    | `/api/model`  | Model parameters, training-set counts, score histogram, feature profile, top anomalies |
| POST   | `/api/scan`   | Multipart `file` upload (CSV, ≤ 50 MB). Returns counts, histogram and per-row results |

`/api/scan` detects the format from the header row:

- **Raw click log**: needs `click` and `device_id` (or `device_ip`). `site_id`, `app_id`, `device_type`,
  `device_conn_type` and `C14`–`C21` are used when present.
- **Session features**: the 15 model feature columns. A `session_id`, `user_id` or `device_id` column labels the rows.

Interactive docs are at `/docs` on the backend.

## Running locally

```bash
# Backend (http://localhost:8000)
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (http://localhost:5173, proxies /api to :8000)
cd frontend
npm install
npm run dev
```

Or run both in containers:

```bash
docker compose up --build     # http://localhost:8080
```

## Retraining

```bash
cd backend
python train.py                              # refit on artifacts/session_features.csv
python train.py --clicks avazu_train.csv     # rebuild features from a raw Avazu file, then fit
python make_samples.py                       # regenerate the sample CSVs in frontend/public/samples
```

## Deploying

Both services ship as Docker images and deploy independently to any container host (Railway, Render, Fly.io, ECS).

- **backend**: build from `backend/`. It listens on `$PORT` (default 8000). `ALLOWED_ORIGINS` restricts CORS
  (default `*`).
- **frontend**: build from `frontend/`. nginx serves the app and proxies `/api` to `BACKEND_URL`, for example
  `https://your-backend.up.railway.app`. To have the browser call the API directly instead, pass
  `--build-arg VITE_API_URL=https://your-backend...`.

### EC2 (current deployment)

Live at **http://3.6.185.10**, on the EC2 instance `clickfraud-monitor` (t3.micro, ap-south-1, Elastic IP). The
containers `clickfraud-backend` and `clickfraud-frontend` run on the `clickfraud-net` Docker network. Every AWS
resource is tagged `Project=clickfraud-monitor`.

| File                         | Purpose                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| `deploy/ec2/user-data.sh`    | Instance bootstrap: Docker and 2 GB swap                                     |
| `deploy/ec2/run.sh`          | Starts the stack on the host. Pulls ECR images if `BACKEND_IMAGE`/`FRONTEND_IMAGE` are set, otherwise builds from source |
| `deploy/ec2/deploy.sh`       | Manual deploy from your machine over SSH (`EC2_HOST=3.6.185.10 deploy/ec2/deploy.sh`) |
| `deploy/ec2/setup-cicd.sh`   | One-time AWS and GitHub setup for continuous deployment (see below)            |

### CI/CD

`.github/workflows/ci-cd.yml` runs on every push and pull request:

1. **Backend tests**: install dependencies and smoke-test `/api/health`, `/api/model` and `/api/scan`.
2. **Frontend build**: `npm ci` and type-check and build.
3. **Deploy** (pushes to `main` only): assume an AWS role via GitHub OIDC (no stored AWS keys), build both images and
   push them to ECR tagged with the commit SHA, roll them out on the instance with SSM Run Command (no inbound SSH
   needed), then health-check the live URL.

The deploy job stays skipped until `deploy/ec2/setup-cicd.sh` has been run once. That script creates the ECR repos,
the instance role and the GitHub deploy role, and sets the repo variables (`AWS_ROLE_ARN`, `AWS_REGION`,
`EC2_INSTANCE_ID`, `APP_URL`).

## Notes on the data

- Avazu assigns one shared placeholder `device_id` to many anonymous users. Those buckets turn into very large
  "sessions" that the model flags. That is why flagged sessions account for about 91% of training clicks despite
  being only 2% of sessions.
- Avazu has no fraud labels, so there is no precision/recall figure. Verdicts are anomaly judgements relative to the
  training distribution. `demo_click_log.csv` is synthetic, with six scripted bot devices: all six are flagged, and no
  human device is.
