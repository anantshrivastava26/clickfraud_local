# Architecture

How the click-fraud detection demo is put together, why it is built that way, and where it is
rough. Read this before changing the feature set, the session logic, or the model artifacts.

## 1. Overview

The system flags **bot-like click behaviour** in ad traffic without any fraud labels. It does
this in two separate phases that share one contract (a list of feature names):

| Phase | When | What happens |
| --- | --- | --- |
| **Offline training** | once, or when retraining | A sample of the Avazu ad-click log is split into *sessions*; each session becomes one row of behavioural features; an unsupervised **IsolationForest** learns what "typical" looks like. |
| **Online scoring** | continuously | Click events arrive over HTTP, are buffered into live sessions, and each finished session is turned into the same features and scored by the saved model. Verdicts are appended to a file and shown on a dashboard. |

The unit of detection is the **session**, not the click: a single click carries almost no signal,
while a burst of 200 clicks with a 100 % click rate across many ads is obviously unusual.

### What this repository contains (and does not)

Everything runs locally. The only AWS dependency in the code is **S3, used by the training script
to download the input CSV** (`boto3`). Glue, Athena, SageMaker, Lambda, CloudWatch and API Gateway
are part of the project's intended cloud design, but **no code for them lives in this repo**.

## 2. System diagram

```text
 OFFLINE  (train_isolation_forest.py)
 ─────────────────────────────────────────────────────────────────────────────
  S3: Avazu CSV sample
        │  boto3 download
        ▼
  sessionize per user ─► aggregate to features ─► drop constant columns
        │                        │                       │
        │                        ▼                       ▼
        │             session_features.csv      StandardScaler ─► IsolationForest
        │                                                │               │
        │                                                ▼               ▼
        │                                         scaler.joblib   isoforest_model.joblib
        │                                         {scaler,features}
 ONLINE ─┼─────────────────────────────────────────────────────────────────────
        │
  click_generator.py                receiver_fastapi.py                  app_streamlit.py
  (simulated humans + bots)         (FastAPI, single process)            (dashboard)
        │                                                                    ▲
        │  POST /register_click     ┌────────────────────────────┐           │ reads
        └──────────────────────────►│ sessions{} in memory       │           │
                                    │   │ idle ≥30 s  (sweeper)  │           │
                                    │   │ or ≥200 events         │           │
                                    │   ▼                        │   append  │
                                    │ features → scale → model ──┼──► scored_sessions.jsonl
                                    └────────────────────────────┘
                                         ▲ loads at startup
                                         └── scaler.joblib, isoforest_model.joblib

  eval_and_visualize.py  ──reads──►  session_features.csv + both joblib files
                         ──writes─►  anomaly_score_hist.png, pca_outliers.png, top_anomalies.csv
```

## 3. Components

| File | Role | Reads | Writes |
| --- | --- | --- | --- |
| [train_isolation_forest.py](train_isolation_forest.py) | Offline pipeline: download → sessionize → features → scale → train | S3 object (`S3_BUCKET`/`S3_KEY`) | `session_features.csv`, `scaler.joblib`, `isoforest_model.joblib` |
| [receiver_fastapi.py](receiver_fastapi.py) | **Backend.** HTTP ingest, live sessionization, feature building, scoring | both `.joblib` files (at startup) | `scored_sessions.jsonl` (append) |
| [click_generator.py](click_generator.py) | Traffic simulator: humans and bots posting to the receiver | – | HTTP requests |
| [app_streamlit.py](app_streamlit.py) | Dashboard: metrics, simulator buttons, batch upload, charts, live feed | `.joblib` files, `session_features.csv`, `scored_sessions.jsonl` | – |
| [eval_and_visualize.py](eval_and_visualize.py) | Offline check: score distribution, PCA plot, top-50 anomalies | `session_features.csv`, `.joblib` files | `anomaly_score_hist.png`, `pca_outliers.png`, `top_anomalies.csv` |

The programs never call each other. They are coupled **only through files**:
`scaler.joblib` / `isoforest_model.joblib` (training → everyone) and `scored_sessions.jsonl`
(receiver → dashboard). That keeps each piece independently runnable, at the price of no real
queue or database.

## 4. Offline path: training

[train_isolation_forest.py](train_isolation_forest.py), in order:

1. **Download** the CSV from S3 to `LOCAL_TMP` (`download_from_s3`).
2. **Sessionize** (`process_chunks`): read in 200 000-row chunks, derive a `user_id`
   (`device_id`, else `device_ip`, else `c1`, else the row index), sort by time, and start a new
   session whenever the gap to the previous row exceeds `SESSION_GAP_MINUTES` (30).
3. **Aggregate** (`aggregate_session`): one dict of features per session (table in §6).
4. **Filter and scale**: fill NaNs, **drop zero-variance columns**, fit a `StandardScaler`.
5. **Train**: `IsolationForest(n_estimators=200, contamination=0.02, random_state=42)`.
6. **Persist**: `scaler.joblib` holds `{"scaler": ..., "features": [...]}`; the model goes to
   `isoforest_model.joblib`.

How the model turns a vector into a verdict: `decision_function` returns a score where **higher is
more normal and negative is anomalous**; `predict` returns `-1` for negative scores. Zero is the
threshold, and it was placed by `contamination=0.02` so that the lowest-scoring ~2 % of *training*
sessions fall below it.

## 5. Online path: scoring

### 5.1 Request flow

1. A client `POST`s a click to `/register_click` (see §7).
2. `add_event` parses the timestamp (falling back to server time), flattens the event to a dict,
   and appends it to `sessions[user_id]` under a lock.
3. If that session now holds `MAX_EVENTS_PER_SESSION` (200) events it is removed and scored
   immediately.
4. Otherwise it waits. A daemon thread (`flush_worker`) wakes every `FLUSH_INTERVAL_SECONDS` (5),
   and scores and evicts every session whose last event is older than `SESSION_INACTIVITY_SECONDS`
   (30).
5. **Scoring** (`score_and_emit_session`): `build_features_from_events` → `reindex` to
   `FEATURE_NAMES` → `scaler.transform` → `decision_function` / `predict`.
6. The verdict is appended as one JSON line to `scored_sessions.jsonl`; the dashboard picks it up
   on its next refresh.

### 5.2 Session lifecycle

```text
 first click ──► OPEN ◄──── more clicks (same user_id)
                  │
     ≥200 events  │  or  no click for ≥30 s (checked every 5 s)
                  ▼
               SCORED ──► removed from memory; the next click from that user opens a NEW session
```

Time from "user stops clicking" to "verdict written" is therefore **30 s to 35 s**. A bot that never
pauses is cut off by the size cap instead and is scored within seconds.

### 5.3 Concurrency model

* State is one dict, `sessions`, in the memory of **one process**.
* Two threads touch it: the request handlers and `flush_worker`. A `threading.Lock` guards every
  access. It is a plain, non-re-entrant lock, so nothing that runs while holding it may try to take
  it again.
* Scoring happens **while the lock is held**. That makes "pop the session, then score it" atomic
  and also serialises writes to the JSONL file, so lines cannot interleave. The cost is that
  ingestion pauses for the few milliseconds a one-row prediction and a file append take.
* `register_click` is declared `async` but does blocking work, so it briefly occupies the event
  loop. Fine at demo volume.

## 6. The feature contract

`aggregate_session` (training) and `build_features_from_events` (serving) must stay in step. This
is the table to check whenever either changes.

| Feature | Training source (Avazu) | Live source (click API) | Used by model |
| --- | --- | --- | :---: |
| `impressions` | number of rows in session | number of events | ✔ |
| `clicks` | sum of `click` | sum of `clicks` | ✔ |
| `click_rate` | clicks / impressions | clicks / impressions | ✔ |
| `unique_sites` | distinct `site_id` | distinct `ad_id` *(proxy)* | ✔ |
| `unique_apps` | distinct `app_id` | constant `0` *(no equivalent)* | ✔ |
| `device_type`, `device_conn_type` | most frequent value | most frequent value | ✔ |
| `c14_nunique` … `c21_nunique` (8) | distinct values of anonymised `C14`–`C21` | constant `0` *(no equivalent)* | ✔ |
| `session_length_secs` | last − first timestamp | last − first timestamp | ✘ constant in training |
| `hour_mean`, `hour_std` | hour-of-day stats | hour-of-day stats (UTC) | ✘ constant in training |
| `interclick_mean`, `interclick_std` | gaps between rows | gaps between events | ✘ constant in training |
| `ua_len` | length of `device_model` | length of `device_model` | ✘ constant in training |

**The model sees 15 of the 21 features.** The six marked ✘ were constant in the committed training
data, so the constant-column filter dropped them, and the serving side discards them again with
`reindex(columns=FEATURE_NAMES)`. They are still computed, so a retrain on data with real timing
would pick them up with no serving change.

`scaler.joblib` stores the ordered feature list next to the scaler precisely so that every consumer
rebuilds an identical input matrix. `scaler.transform` is positional: a shifted column would
silently produce wrong scores, not an error.

## 7. Interfaces

### HTTP (receiver, default `http://localhost:8000`)

`POST /register_click` – body (`ClickEvent`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `user_id` | string | **required** | Session key. All events with the same value are one visitor. |
| `ad_id` | string | `""` | Counted into `unique_sites`. |
| `timestamp` | string | server time | ISO-8601. **Include a UTC offset** (e.g. `+00:00`); see §9. |
| `clicks` | int | `1` | `1` = clicked, `0` = impression only. |
| `device_type`, `device_conn_type` | int | `0` | Categorical device info. |
| `device_model` | string | `""` | Its length becomes `ua_len`. |
| `impressions` | int | `1` | Accepted but **ignored**: impressions = number of events. |
| `user_agent` | string | `""` | Stored, not used by the model. |
| `is_bot_simulated` | bool | `false` | Ground-truth tag from the generator; **never used for scoring**. |

Response: `{"status": "ok", "pending_events_for_user": <int>}`.

`GET /health` → `{"status": "ok", "sessions_active": <int>}`.

### Files

| File | Format |
| --- | --- |
| `scored_sessions.jsonl` | One JSON object per line: `user_id`, `timestamp` (time of the *verdict*), `score` (float; <0 = outlier), `is_outlier` (0/1), `event_count`. Append-only. |
| `scaler.joblib` | `{"scaler": StandardScaler, "features": [str, ...]}`. A bare `StandardScaler` (older format) is still accepted by every reader. |
| `isoforest_model.joblib` | A fitted `sklearn.ensemble.IsolationForest`. |
| `session_features.csv` | One row per training session, all 21 features, no filtering. |

## 8. Design decisions

| Decision | Why | Consequence |
| --- | --- | --- |
| **Score sessions, not clicks** | The model was trained on session aggregates; fraud is a pattern across clicks. | Verdicts are delayed until a session ends. |
| **Unsupervised IsolationForest** | There are no fraud labels in click logs. | No precision/recall can be computed; `contamination` is an assumption, not a measurement. |
| **Two flush triggers** (idle timeout + size cap) | Idle alone never fires for a bot that clicks continuously, so its session would grow forever. | Long bursts are reported as several back-to-back sessions of 200 events. |
| **Short live inactivity gap (30 s) vs 30 min in training** | A demo needs results in seconds. | Live and training sessions are not equivalent; a real deployment should align them. |
| **Feature list saved with the scaler** | Guarantees training and serving use the same columns in the same order. | A model change means regenerating *both* joblib files together. |
| **Zero-fill for features the API cannot supply** (`unique_apps`, `c14`–`c21`) | Keeps one model for both worlds. Safe *because* trees split on thresholds: a `0` lands on the same side as the training minimum (`1`). | Those 9 features carry no live signal, and the trick would break with a distance- or gradient-based model. |
| **Drop constant columns before training** | A constant column carries no information: the forest can never split on it. | The model does not use timing today (§10.1). |
| **`scored_sessions.jsonl` as the hand-off to the dashboard** | No broker or database needed; receiver and dashboard are independent. | Unbounded growth; the dashboard re-reads the whole file on each refresh. |
| **Never reject a click for a bad timestamp** | Availability beats strictness on an ingest path. | Garbage timestamps are silently replaced with server time. |
| **Ground-truth flag carried but unused** | Lets the generator label its own traffic for debugging while keeping detection honest. | – |
| **In-memory session store** | Zero infrastructure. | Single process only; state lost on restart (§9). |

## 9. Operational constraints

* **Run one receiver process.** With several workers, one user's clicks would be split across
  processes and never form a complete session. Scaling out needs a shared store (e.g. Redis).
* **Restarts drop open sessions** unscored. `reload=True` (used by `python receiver_fastapi.py`)
  also restarts on every code change.
* **Send timezone-aware timestamps.** A timestamp without an offset parses as a naive `datetime`;
  the sweeper compares against an aware cutoff, which raises `TypeError` and **stops the sweeper
  thread**, so idle sessions would no longer be flushed. The bundled generator sends aware times.
* **Inactivity is judged by event time against the server clock.** Replaying old timestamps makes
  sessions look idle and flush at the next sweep.
* **Start every program from the repo root.** All artifact paths are relative to the working
  directory.
* **No authentication, and it binds to `0.0.0.0`.** Local demo only.

## 10. Known limitations

1. **Timing features are inert.** Avazu stores `hour` as `YYMMDDHH` (e.g. `14102100`) but the
   training script parses it as `YYYYMMDDHH`, producing nonsense dates (year 1410). Under the
   pinned pandas 2.x these become `NaT`, which is why the time features are constant and get
   dropped. Even with the right format (`%y%m%d%H`) the data is hour-granular, so with a 30-minute
   session gap each session sits inside one hour and `session_length_secs` / `interclick_*` would
   still be ~0. Real timing signals need finer-grained timestamps. As things stand, detection is
   driven by volume and breadth (`impressions`, `clicks`, `click_rate`, `unique_sites`).
2. **Train/serve skew.** Besides the zero-filled features and the session gap, the training
   "users" are Avazu `device_id`s, and in the public Avazu data most rows share one placeholder
   id. The largest training outliers (100 000+ impressions) are therefore probably aggregates of
   many real people rather than one bot. Live users are synthetic ids.
3. **No ground truth.** In the sample log committed with the repo, 6 of 7 simulated-bot sessions
   and 0 of 30 simulated-human sessions were flagged, which is encouraging but only shows the
   model reacts to the generator's crude bot profile, not that it generalises.
4. **`fraud_probability` in the dashboard is a relative rank** (`1 − min-max(score)`), not a
   calibrated probability, and the batch-upload version is rescaled per upload.
5. **Training script structure.** Steps 4–5 (scale and train) are indented at module level, not
   inside `if __name__ == "__main__"`; running the file works, importing it raises `NameError`.
   `LOCAL_TMP` is a POSIX path (`/tmp/...`) that needs adjusting on Windows. Sessions are built
   per 200 000-row chunk, so a user straddling a chunk boundary is split.
6. **Artifact/library mismatch.** The committed `.joblib` files were pickled with scikit-learn
   1.7.0 while `requirements.txt` pins 1.5.1; loading prints `InconsistentVersionWarning`. Retrain
   or align the version.
7. **Dashboard cost.** Streamlit re-runs the whole script on each auto-refresh, so it re-scores
   every training session and refits the PCA each time; only model loading is cached.
8. **Unbounded `scored_sessions.jsonl`.**

## 11. Running it locally

The repository ships trained artifacts, so you can try the live demo without AWS access.

```bash
pip install -r requirements.txt            # Python 3.11+

python receiver_fastapi.py                 # terminal 1: backend on :8000
python click_generator.py                  # terminal 2: simulated humans + bots
streamlit run app_streamlit.py             # terminal 3: dashboard, see "Live Fraud Detection Feed"
```

Optional:

```bash
python eval_and_visualize.py               # regenerate the plots and top_anomalies.csv

# retrain from your own S3 sample (needs AWS credentials; see the note on LOCAL_TMP for Windows)
S3_BUCKET=my-bucket S3_KEY=train_sample.csv python train_isolation_forest.py
# PowerShell:  $env:S3_BUCKET="my-bucket"; $env:S3_KEY="train_sample.csv"; python train_isolation_forest.py
```

After retraining, restart the receiver and dashboard so they reload the new artifacts.

## 12. Extension points

* **More than one receiver / durability:** move `sessions` to Redis, or put a stream (Kinesis,
  Kafka) in front and make scoring a consumer.
* **Higher throughput:** pop sessions under the lock but score outside it; make the handler a
  plain `def`; keep the model in a separate scoring service.
* **Better detection:** fix the `hour` parsing, use sub-hour timestamps so the timing features
  become real, and add labelled data (even a small set) to evaluate precision/recall.
* **Cloud deployment:** the original design (Glue → Athena → SageMaker → Lambda → CloudWatch)
  could map onto this code as: Glue/Athena for the sessionization in `process_chunks`, SageMaker
  for the training step, Lambda or an endpoint for `score_and_emit_session`, CloudWatch for the
  verdict log.
