"""HTTP API for the click-fraud detector.

GET  /api/health    liveness + whether the model is loaded
GET  /api/model     trained model metadata and training-set profile
POST /api/scan      score an uploaded CSV (raw click log or session features)
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detector import Detector

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

detector: Detector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = Detector()
    yield


app = FastAPI(title="Click Fraud Detection API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/api/model")
def model_summary():
    return detector.summary


@app.post("/api/scan")
def scan(file: UploadFile = File(...)):
    name = file.filename or "upload.csv"
    if not name.lower().endswith((".csv", ".txt")):
        raise HTTPException(415, "Upload a .csv file.")

    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content.strip():
        raise HTTPException(400, "The file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:  # pandas raises several parser error types
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    try:
        return detector.scan(df, name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
