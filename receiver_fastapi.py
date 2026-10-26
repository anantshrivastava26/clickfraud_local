# receiver_fastapi.py
"""
Real-time click receiver + fraud scorer (the backend of the demo).

What it does
------------
1. Accepts individual click events over HTTP         (POST /register_click)
2. Buffers them per user_id into in-memory "sessions"
3. When a session ends, collapses it into ONE feature vector, scales it, and asks the
   pre-trained IsolationForest whether the session looks anomalous (bot-like)
4. Appends the verdict as one JSON line to scored_sessions.jsonl, which the Streamlit
   dashboard (app_streamlit.py) tails for its "Live Fraud Detection Feed"

Why sessions instead of scoring every click
-------------------------------------------
The model was trained on SESSION-level aggregates (click rate, distinct sites, ...),
see train_isolation_forest.py. A single click carries almost no signal; fraud shows up
as a pattern over many clicks, so we have to buffer clicks and score the pattern.

A session ends (is "flushed") when either
  * the user has been idle for SESSION_INACTIVITY_SECONDS   -> flush_worker thread, or
  * the session reaches MAX_EVENTS_PER_SESSION events        -> add_event, immediately.

Running it
----------
    python receiver_fastapi.py                       # dev: port 8000, auto-reload
    uvicorn receiver_fastapi:app --port 8000         # without reload

All file paths below (*.joblib, scored_sessions.jsonl, session_features.csv) are
relative to the CURRENT WORKING DIRECTORY, so start the server from the repo root.
scaler.joblib and isoforest_model.joblib must already exist (run
train_isolation_forest.py first).

Deployment constraints (by design, this is a demo)
--------------------------------------------------
* Run exactly ONE process/worker. Session state lives in this process's memory; with
  several workers one user's clicks would be split across processes and never form a
  complete session. To scale out, move `sessions` to a shared store (e.g. Redis).
* State is lost on restart/reload: any unflushed sessions are dropped, not scored.
* No authentication and it binds to 0.0.0.0 - fine for a local demo, not for exposure.
"""
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Inactivity gap that closes a session. Deliberately SHORT so the live demo shows
# results within seconds. (Training uses a 30-MINUTE gap - SESSION_GAP_MINUTES in
# train_isolation_forest.py - because Avazu logs are hour-granular; the two values are
# not meant to be equal, but a real deployment should keep them comparable.)
SESSION_INACTIVITY_SECONDS = 30    # if no event for user within this, flush session
# Safety valve. A bot that clicks continuously never goes idle, so the inactivity
# rule alone would let its session grow forever (memory) and never get scored.
# This cap forces a verdict after N events. Bots hit it within seconds in the demo.
MAX_EVENTS_PER_SESSION = 200       # flush if too many events
# How often the janitor thread wakes up. Worst-case delay between "user went quiet"
# and "verdict written" is therefore SESSION_INACTIVITY_SECONDS + FLUSH_INTERVAL_SECONDS.
FLUSH_INTERVAL_SECONDS = 5         # background flush loop interval

# ---------------------------------------------------------------------------
# Load model + scaler ONCE at import time, so requests never pay disk/unpickle cost.
# ---------------------------------------------------------------------------
scaler_blob = joblib.load("scaler.joblib")

# scaler.joblib has two historical formats:
#   * current : {"scaler": StandardScaler, "features": [ordered column names]}
#   * legacy  : a bare StandardScaler (no feature list stored)
# FEATURE_NAMES is the train/serve CONTRACT: the exact columns, in the exact order,
# that the scaler and model were fitted on. Every vector we score is forced into this
# shape (see score_and_emit_session) so a feature added/removed on either side cannot
# silently shift columns.
if isinstance(scaler_blob, dict):
    scaler = scaler_blob["scaler"]
    FEATURE_NAMES = scaler_blob["features"]
else:
    scaler = scaler_blob
    # fallback: infer features from model or csv
    # NOTE: best-effort only. The CSV header lists ALL features, including columns the
    # training step may have dropped as constant, so this can disagree with what the
    # scaler was really fitted on. Re-run training to get the dict format instead.
    try:
        import pandas as pd
        df = pd.read_csv("session_features.csv", nrows=1)
        FEATURE_NAMES = list(df.columns)
    except Exception:
        FEATURE_NAMES = getattr(scaler, "feature_names_in_", [])

model = joblib.load("isoforest_model.joblib")

app = FastAPI(title="Click Receiver (Demo)")

# ---------------------------------------------------------------------------
# Shared mutable state
# ---------------------------------------------------------------------------
# In-memory sessions: user_id -> {events: [...], last_ts: datetime}
#   events  : list of per-click dicts (built in add_event)
#   last_ts : timestamp of the newest event; the janitor compares it to "now"
sessions: Dict[str, Dict] = {}
# `sessions` is touched by two threads: FastAPI request handlers and flush_worker.
# A plain threading.Lock is the right tool because flush_worker is a real OS thread
# (an asyncio.Lock would not protect against it). It is NOT re-entrant, so code that
# runs while holding it (e.g. score_and_emit_session) must never try to take it again.
lock = threading.Lock()

class ClickEvent(BaseModel):
    """Request body of POST /register_click - one click/impression from one user."""
    user_id: str      # device_id or device_ip (string)
    # user_id is the SESSION KEY: all events with the same value are grouped together.
    ad_id: str = ""
    # Event time. Send timezone-aware ISO-8601 (e.g. "2025-10-13T09:18:03+00:00").
    # If omitted or unparseable, the server's receive time is used (see add_event).
    timestamp: str = None  # ISO format recommended
    # Accepted and stored but NOT a model input - the model has no user-agent feature.
    user_agent: str = ""
    # Ground-truth tag set by click_generator.py, kept only for debugging/demo.
    # It is deliberately never used for scoring: the model is unsupervised and must
    # reach its verdict from behaviour alone.
    is_bot_simulated: bool = False
    # optional additional fields we can use to compute features
    # `impressions` is accepted for API symmetry but ignored: the impressions feature
    # is the NUMBER OF EVENTS in the session (as in training, where each row is one
    # impression).
    impressions: int = 1
    clicks: int = 1          # 1 = the impression was clicked, 0 = not clicked
    device_type: int = 0
    device_conn_type: int = 0
    # Stand-in for a user-agent string: its LENGTH feeds the `ua_len` feature,
    # mirroring training, where Avazu's device_model plays the same role.
    device_model: str = ""

def now_utc():
    """Timezone-aware 'now'. Always use aware datetimes so comparisons never mix kinds."""
    return datetime.now(timezone.utc)

def add_event(e: ClickEvent):
    """Append one click to its user's session; flush the session if it got too big."""
    ts = None
    if e.timestamp:
        try:
            ts = datetime.fromisoformat(e.timestamp)
        except:
            # Never reject a click over a malformed timestamp - ingest it using
            # server time instead. Availability beats strictness for a click stream.
            ts = now_utc()
    else:
        ts = now_utc()
    # NOTE: a timestamp WITHOUT a UTC offset parses to a *naive* datetime. flush_worker
    # compares last_ts with a timezone-aware cutoff, which raises TypeError for naive
    # values and would kill that thread. Clients must send aware timestamps (the
    # bundled generator does).
    # Also note that inactivity is judged by EVENT time vs. the server's wall clock, so
    # replaying old timestamps makes sessions look idle and flush at the next sweep.

    # Flatten the pydantic model into a plain dict; this is what feature building reads.
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
            # pop() removes the session, so the user's NEXT click starts a fresh one
            # (a very long burst is therefore reported as several back-to-back sessions).
            sess = sessions.pop(e.user_id, None)
            if sess:
                # Scored while still holding `lock`: simple and race-free (nothing can
                # append to this session meanwhile), at the cost of pausing ingestion
                # for the few ms a single-row predict + file append takes. For higher
                # throughput, pop under the lock and score after releasing it.
                _ = score_and_emit_session(e.user_id, sess)

def build_features_from_events(events: List[Dict]):
    """Collapse a session's raw events into the feature dict the model understands.

    This MUST stay in step with `aggregate_session` in train_isolation_forest.py:
    same feature names, same meaning. Where the live API cannot supply the training
    field, a neutral stand-in is used and called out below.
    Returns None for an empty session, otherwise a dict of 21 features (the model
    itself only consumes the subset listed in FEATURE_NAMES).
    """
    df = pd.DataFrame(events)
    if df.empty:
        return None
    # ensure ts
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    # One event == one impression (same convention as training rows).
    impressions = len(df)
    clicks = int(pd.to_numeric(df.get("clicks", pd.Series([0]*len(df))), errors="coerce").fillna(0).sum())
    click_rate = clicks / impressions if impressions>0 else 0.0
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    session_length_secs = (ts_max - ts_min).total_seconds() if pd.notna(ts_min) and pd.notna(ts_max) else 0.0
    # Training counts distinct site_id; the click API has no site, so ad_id is the proxy.
    # A bot hopping across many ads inflates this - a classic fraud signal.
    unique_sites = df["ad_id"].nunique() if "ad_id" in df else 0
    # No app concept in this API -> constant 0 (training had real app_id counts).
    unique_apps = 0
    # Most frequent value in the session, matching training. mode()[0] picks the first
    # on ties.
    device_type = int(df.get("device_type", pd.Series([-1])).mode()[0]) if "device_type" in df else -1
    device_conn_type = int(df.get("device_conn_type", pd.Series([-1])).mode()[0]) if "device_conn_type" in df else -1
    ua_len = df.get("device_model", pd.Series([""])).astype(str).apply(len).mean() if "device_model" in df else 0
    # Time-of-day features. UTC hours, as event times are UTC-aware.
    hours = df["ts"].dt.hour.dropna()
    hour_mean = float(hours.mean()) if not hours.empty else float(now_utc().hour)
    hour_std = float(hours.std()) if not hours.empty else 0.0
    # Gaps between consecutive clicks: bots click at machine-regular, tiny intervals.
    # With a single event there is no gap, so fall back to session length (0.0), the
    # same fallback training uses.
    inter_clicks = df["ts"].sort_values().diff().dt.total_seconds().dropna()
    ict_mean = float(inter_clicks.mean()) if not inter_clicks.empty else session_length_secs
    ict_std = float(inter_clicks.std()) if not inter_clicks.empty else 0.0
    # NOTE: the committed model was trained on data where session_length_secs, hour_*,
    # interclick_* and ua_len were all constant, so training dropped them. They are
    # still computed here for parity (and in case the model is retrained on data with
    # real timing), but score_and_emit_session's reindex() discards them today.

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
    # Avazu's anonymised C14-C21 columns do not exist in a live click event. Zero is a
    # safe filler ONLY because IsolationForest splits on thresholds: a 0 falls on the
    # same side of every split as the training minimum (1), so it behaves like a
    # "typical" session. A distance- or gradient-based model would be thrown off by it.
    for c in ["c14","c15","c16","c17","c18","c19","c20","c21"]:
        feat[f"{c}_nunique"] = 0

    return feat

import json

def score_and_emit_session(user_id: str, session: Dict):
    """Score one finished session and persist the verdict. Returns the result dict.

    Callers hold `lock` (see add_event / flush_worker), which also guarantees that only
    one thread appends to the output file at a time, so JSON lines never interleave.
    """
    feats = build_features_from_events(session["events"])
    if feats is None:
        return None
    x = pd.DataFrame([feats])
    # Enforce the training-time schema: keep only the trained columns, in trained order,
    # and zero-fill any the model expects but we did not compute. This is also what
    # drops the 6 features the model never learned (see build_features_from_events).
    x = x.reindex(columns=FEATURE_NAMES, fill_value=0)
    x_scaled = scaler.transform(x)
    # decision_function: higher = more normal, NEGATIVE = anomalous. It is the raw
    # score shifted so that 0 is the decision boundary; that boundary was fixed at
    # training time by `contamination=0.02`, i.e. "the most extreme ~2% of TRAINING
    # sessions are outliers". predict() just returns -1 when the score is below 0.
    score = model.decision_function(x_scaled)[0]
    pred = model.predict(x_scaled)[0]

    result = {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),   # time of VERDICT, not of the clicks
        "score": float(score),
        "is_outlier": int(pred == -1),                         # 1 = suspicious, 0 = normal
        "event_count": len(session["events"])
    }

    # ✅ Append each detection to file
    # JSON Lines file = the hand-off to the dashboard. A flat append-only file needs no
    # broker/DB for a local demo, and writers/readers stay decoupled (the dashboard
    # works with the receiver stopped). Opened per write: slower than keeping a handle,
    # but crash-safe and trivially simple. Grows without bound - rotate it if long-running.
    with open("scored_sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print("SCORED:", result)
    return result


# Background thread: flush inactive sessions periodically
def flush_worker():
    """Janitor loop: score and evict sessions whose user has gone quiet.

    A user who simply stops clicking never triggers another request, so nothing on the
    request path would ever close their session. Polling from a thread is the simplest
    way to catch that.
    """
    while True:
        time.sleep(FLUSH_INTERVAL_SECONDS)
        cutoff = now_utc() - timedelta(seconds=SESSION_INACTIVITY_SECONDS)
        to_flush = []
        with lock:
            # Two passes (collect, then pop) so we never mutate `sessions` while
            # iterating over it.
            for uid, s in list(sessions.items()):
                if s["last_ts"] < cutoff:
                    to_flush.append(uid)
            for uid in to_flush:
                sess = sessions.pop(uid, None)
                if sess:
                    _ = score_and_emit_session(uid, sess)

# daemon=True: this thread must not keep the process alive after the server stops.
# It starts at import time, so every process that imports this module gets one. With
# `python receiver_fastapi.py` + reload=True that is two: the launcher (this file as
# __main__, which also loads the models) and the server child that actually serves
# requests. The launcher's copy idles on an empty `sessions` dict - harmless, just wasteful.
bg_thread = threading.Thread(target=flush_worker, daemon=True)
bg_thread.start()

@app.post("/register_click")
async def register_click(event: ClickEvent):
    """Receive single click event. Returns quick ack with current session count."""
    # NOTE: declared `async` but the work below is blocking (lock + occasional model
    # call + file write), so it briefly occupies the event loop. Acceptable at demo
    # volume; switch to a plain `def` (FastAPI then uses a thread pool) if it grows.
    add_event(event)
    with lock:
        pending = len(sessions.get(event.user_id, {"events": []})["events"])
    # `pending` is 0 right after a size-triggered flush (the session was just closed).
    return {"status": "ok", "pending_events_for_user": pending}

@app.get("/health")
async def health():
    """Liveness probe + how many sessions are currently buffered."""
    # len() on a dict is atomic under the GIL, so no lock is needed for this read-only gauge.
    return {"status": "ok", "sessions_active": len(sessions)}

if __name__ == "__main__":
    # The "module:app" import-string form is REQUIRED for reload=True. reload is a
    # dev convenience only: it restarts the process on every code change, which also
    # wipes the in-memory sessions. host 0.0.0.0 listens on all interfaces.
    uvicorn.run("receiver_fastapi:app", host="0.0.0.0", port=8000, reload=True)
