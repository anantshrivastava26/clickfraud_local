"""HTTP API for the click-fraud detector.

GET  /api/health    liveness + whether the model is loaded
GET  /api/model     trained model metadata and training-set profile
POST /api/scan      score an uploaded CSV (raw click log or session features)

This module is a thin web layer. All modelling lives in detector.py (scoring) and
features.py (feature building); the handlers only validate the upload, translate problems
into HTTP status codes, and hand the parsed DataFrame to the Detector. Nothing from an
upload is stored: the service is stateless per request.

Run from backend/:  uvicorn app.main:app --reload      (interactive docs at /docs)

Configuration (environment variables)
    MAX_UPLOAD_MB    largest accepted upload, default 50. Keep nginx's client_max_body_size
                     (frontend/nginx.conf.template) at least this large, otherwise nginx
                     rejects big files before they reach this service.
    ALLOWED_ORIGINS  comma-separated CORS origins, default "*".

Status codes returned by POST /api/scan
    415  file name is not .csv / .txt         413  larger than MAX_UPLOAD_MB
    400  empty file, or not parseable as CSV   422  readable CSV in an unrecognised format
                                                    (the message says which columns are missing)
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detector import Detector

# Upper bound on an upload. The file is read fully into memory and then parsed by pandas, so this
# also bounds the worst-case memory used by one request.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# The shared, read-only scorer. It is built once at startup (see lifespan) because constructing it
# re-scores the whole training set, which is too slow to repeat per request. It is None until
# startup has finished, which is what /api/health reports as model_loaded.
detector: Detector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the Detector before the first request is served. A missing or incompatible model file
    therefore fails the deployment at boot instead of on the first upload, and CI's `with
    TestClient(app)` exercises this path.
    """
    global detector
    detector = Detector()
    yield


app = FastAPI(title="Click Fraud Detection API", version="2.0.0", lifespan=lifespan)

# CORS only matters when a browser calls this API from a different origin. In the shipped setups the
# frontend proxies /api (Vite in development, nginx in Docker and on EC2), so requests are
# same-origin. The permissive '*' default keeps local setups simple; set ALLOWED_ORIGINS to the real
# site in production. Only GET and POST are allowed because those are the only methods the API uses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# The handlers below are plain `def`, not `async def`, on purpose: pandas and scikit-learn are
# CPU-bound and blocking, so FastAPI runs them in its thread pool and a slow scan cannot stall
# health checks or other requests.
@app.get("/api/health")
def health():
    """Liveness probe (used by docker-compose's healthcheck and the CI smoke test). model_loaded is
    False only while startup is still running.
    """
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/api/model")
def model_summary():
    """The training-set baseline the UI shows. It is precomputed in Detector.__init__, so this is a
    lookup, not a computation.
    """
    return detector.summary


@app.post("/api/scan")
def scan(file: UploadFile = File(...)):
    """Score an uploaded CSV. Whether it is a raw click log or an already-aggregated session file is
    detected by the Detector from the header row.
    """
    name = file.filename or "upload.csv"
    # Cheap early rejection by file extension only; the real validation is the CSV parse below.
    if not name.lower().endswith((".csv", ".txt")):
        raise HTTPException(415, "Upload a .csv file.")

    # Read at most one byte past the limit: that is enough to tell the file is too big without ever
    # loading an arbitrarily large upload into memory.
    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content.strip():
        raise HTTPException(400, "The file is empty.")

    try:
        # low_memory=False parses the whole file in one pass so each column gets a single consistent
        # dtype.
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:  # pandas raises several parser error types
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    try:
        # The Detector raises ValueError for problems the user can fix (unknown format, no rows,
        # missing columns). Report those as 422 with the message, which the UI shows verbatim.
        # Anything else is a bug and surfaces as a 500.
        return detector.scan(df, name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
