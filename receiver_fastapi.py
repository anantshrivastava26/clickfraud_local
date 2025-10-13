# receiver_fastapi.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime, timedelta, timezone
import joblib
import pandas as pd
import threading
import time
import uvicorn
import uuid

# Config
SESSION_INACTIVITY_SECONDS = 30    # if no event for user within this, flush session
MAX_EVENTS_PER_SESSION = 200       # flush if too many events
FLUSH_INTERVAL_SECONDS = 5         # background flush loop interval

# Load model+scaler
scaler_blob = joblib.load("scaler.joblib")

if isinstance(scaler_blob, dict):
    scaler = scaler_blob["scaler"]
    FEATURE_NAMES = scaler_blob["features"]
else:
    scaler = scaler_blob
    # fallback: infer features from model or csv
    try:
        import pandas as pd
        df = pd.read_csv("session_features.csv", nrows=1)
        FEATURE_NAMES = list(df.columns)
    except Exception:
        FEATURE_NAMES = getattr(scaler, "feature_names_in_", [])

model = joblib.load("isoforest_model.joblib")

app = FastAPI(title="Click Receiver (Demo)")

# In-memory sessions: user_id -> {events: [...], last_ts: datetime}
sessions: Dict[str, Dict] = {}
lock = threading.Lock()

class ClickEvent(BaseModel):
    user_id: str      # device_id or device_ip (string)
    ad_id: str = ""
    timestamp: str = None  # ISO format recommended
    user_agent: str = ""
    is_bot_simulated: bool = False
    # optional additional fields we can use to compute features
    impressions: int = 1
    clicks: int = 1
    device_type: int = 0
    device_conn_type: int = 0
    device_model: str = ""

def now_utc():
    return datetime.now(timezone.utc)

def add_event(e: ClickEvent):
    ts = None
    if e.timestamp:
        try:
            ts = datetime.fromisoformat(e.timestamp)
        except:
            ts = now_utc()
    else:
        ts = now_utc()

    ev = {
        "ts": ts,
        "ad_id": e.ad_id,
        "ua": e.user_agent,
        "is_bot_simulated": e.is_bot_simulated,
        "impressions": e.impressions,
        "clicks": e.clicks,
        "device_type": e.device_type,
        "device_conn_type": e.device_conn_type,
        "device_model": e.device_model
    }
    with lock:
        s = sessions.setdefault(e.user_id, {"events": [], "last_ts": ts})
        s["events"].append(ev)
        s["last_ts"] = ts
        # If session too big, flush immediately
        if len(s["events"]) >= MAX_EVENTS_PER_SESSION:
            sess = sessions.pop(e.user_id, None)
            if sess:
                _ = score_and_emit_session(e.user_id, sess)

def build_features_from_events(events: List[Dict]):
    df = pd.DataFrame(events)
    if df.empty:
        return None
    # ensure ts
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    impressions = len(df)
    clicks = int(pd.to_numeric(df.get("clicks", pd.Series([0]*len(df))), errors="coerce").fillna(0).sum())
    click_rate = clicks / impressions if impressions>0 else 0.0
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    session_length_secs = (ts_max - ts_min).total_seconds() if pd.notna(ts_min) and pd.notna(ts_max) else 0.0
    unique_sites = df["ad_id"].nunique() if "ad_id" in df else 0
    unique_apps = 0
    device_type = int(df.get("device_type", pd.Series([-1])).mode()[0]) if "device_type" in df else -1
    device_conn_type = int(df.get("device_conn_type", pd.Series([-1])).mode()[0]) if "device_conn_type" in df else -1
    ua_len = df.get("device_model", pd.Series([""])).astype(str).apply(len).mean() if "device_model" in df else 0
    hours = df["ts"].dt.hour.dropna()
    hour_mean = float(hours.mean()) if not hours.empty else float(now_utc().hour)
    hour_std = float(hours.std()) if not hours.empty else 0.0
    inter_clicks = df["ts"].sort_values().diff().dt.total_seconds().dropna()
    ict_mean = float(inter_clicks.mean()) if not inter_clicks.empty else session_length_secs
    ict_std = float(inter_clicks.std()) if not inter_clicks.empty else 0.0

    feat = {
        "impressions": impressions,
        "clicks": clicks,
        "click_rate": click_rate,
        "session_length_secs": session_length_secs,
        "unique_sites": unique_sites,
        "unique_apps": unique_apps,
        "device_type": device_type,
        "device_conn_type": device_conn_type,
        "ua_len": float(ua_len),
        "hour_mean": float(hour_mean),
        "hour_std": float(hour_std),
        "interclick_mean": float(ict_mean),
        "interclick_std": float(ict_std)
    }

    # add c14..c21 zeros if model expects them
    for c in ["c14","c15","c16","c17","c18","c19","c20","c21"]:
        feat[f"{c}_nunique"] = 0

    return feat

import json

def score_and_emit_session(user_id: str, session: Dict):
    feats = build_features_from_events(session["events"])
    if feats is None:
        return None
    x = pd.DataFrame([feats])
    x = x.reindex(columns=FEATURE_NAMES, fill_value=0)
    x_scaled = scaler.transform(x)
    score = model.decision_function(x_scaled)[0]
    pred = model.predict(x_scaled)[0]

    result = {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": float(score),
        "is_outlier": int(pred == -1),
        "event_count": len(session["events"])
    }

    # ✅ Append each detection to file
    with open("scored_sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print("SCORED:", result)
    return result


# Background thread: flush inactive sessions periodically
def flush_worker():
    while True:
        time.sleep(FLUSH_INTERVAL_SECONDS)
        cutoff = now_utc() - timedelta(seconds=SESSION_INACTIVITY_SECONDS)
        to_flush = []
        with lock:
            for uid, s in list(sessions.items()):
                if s["last_ts"] < cutoff:
                    to_flush.append(uid)
            for uid in to_flush:
                sess = sessions.pop(uid, None)
                if sess:
                    _ = score_and_emit_session(uid, sess)

bg_thread = threading.Thread(target=flush_worker, daemon=True)
bg_thread.start()

@app.post("/register_click")
async def register_click(event: ClickEvent):
    """Receive single click event. Returns quick ack with current session count."""
    add_event(event)
    with lock:
        pending = len(sessions.get(event.user_id, {"events": []})["events"])
    return {"status": "ok", "pending_events_for_user": pending}

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(sessions)}

if __name__ == "__main__":
    uvicorn.run("receiver_fastapi:app", host="0.0.0.0", port=8000, reload=True)
