# train_isolation_forest.py
"""
Offline training pipeline: raw ad-impression log  ->  session features  ->  model.

Pipeline
--------
  1. download   a CSV sample of the Avazu CTR log from S3
  2. sessionize group rows per user, split into sessions at long idle gaps
  3. aggregate  turn each session into ONE row of behavioural features
  4. scale      standardise the features (StandardScaler)
  5. train      fit an unsupervised IsolationForest on the scaled matrix

Outputs (written to the current directory; consumed by the other scripts)
-------------------------------------------------------------------------
  session_features.csv  one row per session, ALL computed features (eval + dashboard)
  scaler.joblib         {"scaler": StandardScaler, "features": [ordered column names]}
  isoforest_model.joblib  the fitted IsolationForest

The FEATURE LIST saved inside scaler.joblib is the contract with the serving side:
receiver_fastapi.py, eval_and_visualize.py and app_streamlit.py all reindex their input
to exactly these columns before scaling/scoring.

Why IsolationForest / why unsupervised
--------------------------------------
Click logs have no fraud labels. IsolationForest needs none: it isolates points with
random splits, and rare/extreme points (huge click counts, one "user" hitting hundreds of
sites, ...) get isolated in fewer splits, i.e. a lower score.

Configuration (environment variables)
-------------------------------------
  S3_BUCKET  bucket holding the sample   (default "my-avazu-bucket-12345")
  S3_KEY     object key of the CSV        (default "train_sample.csv")
AWS credentials come from the standard boto3 chain (env vars, ~/.aws, role, ...).

Usage:  python train_isolation_forest.py
"""
import os
import io
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json

# CONFIG
S3_BUCKET = os.getenv("S3_BUCKET", "my-avazu-bucket-12345")
S3_KEY = os.getenv("S3_KEY", "train_sample.csv")  # path in bucket
# NOTE: POSIX-style path. On Windows "/tmp" resolves to <current drive>:\tmp, which must
# already exist or the download fails - point this at a writable directory there.
LOCAL_TMP = "/tmp/avazu_sample.csv"
CHUNK_SIZE = 200_000   # adjust to fit local memory; file has 1M rows
# A user idle longer than this starts a new session. Avazu is hour-granular (see the
# note in process_chunks), so in practice every session ends up inside a single hour.
# The live receiver uses a much shorter gap on purpose (see receiver_fastapi.py).
SESSION_GAP_MINUTES = 30
MODEL_OUT = "isoforest_model.joblib"
SCALER_OUT = "scaler.joblib"
FEATURES_OUT = "session_features.csv"

# Helper: read CSV from S3 into a local tmp file (streamed)
def download_from_s3(bucket, key, local_path):
    """Copy s3://bucket/key to local_path. boto3 streams it to disk in parts, so the
    file never has to fit in memory."""
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded s3://{bucket}/{key} -> {local_path}")

# Sessionization + per-session aggregator
def process_chunks(csv_path, chunk_size=CHUNK_SIZE):
    """Read the CSV in chunks, split each user's rows into sessions, return one feature
    dict per session (see aggregate_session).

    Chunked reading keeps memory bounded on a 1M+ row file. Trade-off: users are
    sessionized WITHIN each chunk only, so a user whose rows straddle a chunk boundary
    is cut into two sessions. Sorting the file by user first would avoid that; it was
    judged not worth the cost for an anomaly-detection demo.
    """
    # We'll build a list of session-level dictionaries
    sessions = []
    # Use these columns if available; if not present, code will still work
    parse_dates = False   # (unused leftover)
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=str,           # ✅ Force all columns to string type
        low_memory=False     # ✅ Prevent chunk-wise type guessing
    ):
        # (Everything is read as str so IDs such as hashed device_id are never coerced
        # to numbers; numeric columns are converted explicitly where needed below.)

        # Standardize column names (lowercase)
        chunk.columns = [c.lower() for c in chunk.columns]
        # Ensure a timestamp column. Avazu has 'hour' as yyyymmddHH format integer/string.
        # Convert 'hour' to datetime if present
        if 'hour' in chunk.columns:
            # hour looks like '20130521 14' or 2013052114? Accept both:
            # If numeric like 2013052100 convert to string and parse first 10 digits.
            # KNOWN ISSUE: the real Kaggle Avazu file stores hour as YYMMDDHH (8 digits,
            # e.g. 14102100), NOT YYYYMMDDHH. strptime("%Y%m%d%H") then "succeeds" with
            # garbage - '14102100' becomes year 1410, 10 Feb. With the pinned pandas 2.x
            # (nanosecond timestamps, valid range 1677-2262) such dates are out of range,
            # so aggregate_session's pd.to_datetime(errors="coerce") turns them into NaT.
            # (pandas 3.x can represent them, so the outcome depends on the pandas version.)
            # Net effect in the committed data: session_length_secs, hour_*, interclick_*
            # are constant, so they are dropped before training (see the constant-column
            # filter below). Fix: parse with "%y%m%d%H".
            def parse_hour(h):
                try:
                    s = str(int(h))
                except:
                    s = str(h)
                # try parse as YYYYMMDDHH
                try:
                    return datetime.strptime(s[:10], "%Y%m%d%H")
                except:
                    # fallback: try ISO parse
                    try:
                        return pd.to_datetime(s)
                    except:
                        return pd.NaT
            chunk['ts'] = chunk['hour'].apply(parse_hour)
        else:
            # If there is no hour, create synthetic ts as now plus row idx delta
            # NOTE: despite the sentence above, no per-row delta is added: every row gets
            # the SAME "now" timestamp, so all of a user's rows fall into one session
            # with zero length.
            chunk['ts'] = pd.Timestamp.utcnow()

        # user identifier: prefer device_id else device_ip else C1
        # Priority = how well the column identifies one real user. Caveat: in Avazu a
        # single placeholder device_id is shared by a huge share of rows, so those rows
        # collapse into a few giant "users"/sessions (the 100k+ impression outliers in
        # top_anomalies.csv are very likely exactly that), which is why the model flags
        # volume extremes.
        if 'device_id' in chunk.columns:
            chunk['user_id'] = chunk['device_id'].astype(str)
        elif 'device_ip' in chunk.columns:
            chunk['user_id'] = chunk['device_ip'].astype(str)
        elif 'c1' in chunk.columns:
            chunk['user_id'] = chunk['c1'].astype(str)
        else:
            chunk['user_id'] = chunk.index.astype(str)

        # group per user and sessionize by gap
        # Sorting by (user, time) lets one linear pass per user find the gaps.
        chunk = chunk.sort_values(['user_id','ts'])
        for user, g in chunk.groupby('user_id'):
            prev_ts = None
            session_idx = 0      # (unused bookkeeping)
            acc = []             # rows of the session currently being built
            for _, row in g.iterrows():
                ts = row['ts']
                if pd.isna(ts):
                    continue
                if prev_ts is None:
                    # start session
                    session_idx += 1
                    session_start = ts
                    acc = [row]
                else:
                    gap = (ts - prev_ts).total_seconds() / 60.0
                    if gap > SESSION_GAP_MINUTES:
                        # flush old session
                        sessions.append(aggregate_session(acc))
                        # start new
                        session_idx += 1
                        acc = [row]
                    else:
                        acc.append(row)
                prev_ts = ts
            # the last session of this user never sees a "big gap", so flush it here
            if acc:
                sessions.append(aggregate_session(acc))
    return sessions

def safe_get(row, col, default=np.nan):
    """(Unused helper) row[col] if present, else default."""
    return row[col] if col in row else default

def aggregate_session(rows):
    """Reduce one session (a list of pandas rows) to a flat dict of numeric features.

    These are the model's candidate inputs. receiver_fastapi.build_features_from_events
    reproduces the same names/meanings for live traffic, so keep the two in sync.
    Fraud intuition behind each group:
      volume / ratio : impressions, clicks, click_rate   - bots click too much, too often
      breadth        : unique_sites, unique_apps, c14..c21_nunique - bots hop around
      device         : device_type, device_conn_type, ua_len
      timing         : session_length_secs, hour_*, interclick_* - bots are fast / regular
    """
    # rows: list of pandas Series (rows) belonging to same session
    df = pd.DataFrame(rows)

    # ✅ Ensure timestamps are datetime
    # errors="coerce": unparseable / out-of-range values become NaT rather than raising.
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

    # Basic counts
    # In the Avazu log every row is one ad impression, so impressions = number of rows.
    impressions = len(df)
    clicks = pd.to_numeric(df['click'], errors='coerce').fillna(0).sum() if 'click' in df else 0

    click_rate = clicks / impressions if impressions > 0 else 0.0

    # session time
    ts_min = df['ts'].min()
    ts_max = df['ts'].max()
    session_length_secs = (
        (ts_max - ts_min).total_seconds()
        if pd.notna(ts_min) and pd.notna(ts_max)
        else 0.0
    )

    # unique sites/apps
    unique_sites = df['site_id'].nunique() if 'site_id' in df else 0
    unique_apps = df['app_id'].nunique() if 'app_id' in df else 0

    # device info
    # Dominant (most frequent) value in the session; -1 means "unknown".
    device_type = (
        df['device_type'].mode()[0]
        if 'device_type' in df and not df['device_type'].mode().empty
        else -1
    )
    device_conn_type = (
        df['device_conn_type'].mode()[0]
        if 'device_conn_type' in df and not df['device_conn_type'].mode().empty
        else -1
    )

    # UA-like proxy: device_model length
    # Avazu has no raw user-agent string. device_model is a hashed id (always 8 chars in
    # the committed data), so this feature is constant here and gets dropped later.
    ua_len = df['device_model'].astype(str).apply(len).mean() if 'device_model' in df else 0

    # hour features
    # Sentinel -1 / 0 when no valid timestamp exists (see KNOWN ISSUE in process_chunks).
    hours = df['ts'].dt.hour.dropna()
    hour_mean = hours.mean() if not hours.empty else -1
    hour_std = hours.std() if not hours.empty else 0

    # C14-C21 stats if present
    # Avazu's anonymised categorical columns: we cannot interpret them, but "how many
    # distinct values did this session touch" is a useful, meaning-agnostic breadth signal.
    c_stats = {}
    for c in ['c14','c15','c16','c17','c18','c19','c20','c21']:
        if c in df.columns:
            try:
                c_stats[f'{c}_nunique'] = df[c].nunique()
            except Exception:
                c_stats[f'{c}_nunique'] = 0
        else:
            c_stats[f'{c}_nunique'] = 0

    # inter-click time mean/std
    # With one event there is no gap; fall back to session length (same as the receiver).
    inter_clicks = df['ts'].sort_values().diff().dt.total_seconds().dropna()
    ict_mean = inter_clicks.mean() if not inter_clicks.empty else session_length_secs
    ict_std = inter_clicks.std() if not inter_clicks.empty else 0.0

    return {
        'impressions': impressions,
        'clicks': clicks,
        'click_rate': click_rate,
        'session_length_secs': session_length_secs,
        'unique_sites': unique_sites,
        'unique_apps': unique_apps,
        'device_type': int(device_type) if device_type is not None else -1,
        'device_conn_type': int(device_conn_type) if device_conn_type is not None else -1,
        'ua_len': float(ua_len),
        'hour_mean': float(hour_mean) if pd.notna(hour_mean) else -1,
        'hour_std': float(hour_std) if pd.notna(hour_std) else 0,
        'interclick_mean': float(ict_mean),
        'interclick_std': float(ict_std),
        **c_stats
    }


if __name__ == "__main__":
    # 1) download CSV from S3
    print("Downloading CSV from S3...")
    download_from_s3(S3_BUCKET, S3_KEY, LOCAL_TMP)

    # 2) process and aggregate sessions
    print("Processing chunks and aggregating sessions. This may take a while...")
    sessions = process_chunks(LOCAL_TMP)

    # Convert to DataFrame
    feat_df = pd.DataFrame(sessions)
    print(f"Built {len(feat_df)} session records.")
    # Saved BEFORE any column filtering, so this file keeps every computed feature.
    feat_df.to_csv(FEATURES_OUT, index=False)
    print(f"Wrote session features to {FEATURES_OUT}")

    # 3) Prepare training data (drop suspicious or infinite values)
    X = feat_df.fillna(0).astype(float)
    # remove constant columns if any
    # A zero-variance column carries no information (the forest can never split on it),
    # and scaling would just turn it into all zeros. The surviving columns become the
    # model's feature list.
    X = X.loc[:, X.std() > 0.0]

   # 4) Scale
# NOTE: steps 4-5 below are indented at module level, NOT inside the `__main__` block
# above. Running this file as a script works (X is defined by then), but importing it
# would execute them and fail with NameError. Behaviour kept as-is on purpose.
scaler = StandardScaler()
# Zero mean / unit variance per feature. IsolationForest itself is insensitive to
# feature scale (it splits uniformly between each feature's min and max), so this is not
# needed for the forest; it is kept because the dashboard's PCA plots and the shared
# scaler artifact use the same scaled space.
X_scaled = scaler.fit_transform(X)

# ✅ Save both scaler and feature names
# Persisting the ORDERED column list beside the scaler is what lets the serving code
# rebuild an identical input matrix (see FEATURE_NAMES in receiver_fastapi.py).
feature_names = list(X.columns)
joblib.dump({"scaler": scaler, "features": feature_names}, SCALER_OUT)
print(f"✅ Saved scaler + {len(feature_names)} features -> {SCALER_OUT}")
print("Feature names preview:", feature_names[:10])

# 5) Train IsolationForest
print("Training IsolationForest...")
model = IsolationForest(
    n_estimators=200,       # more trees = more stable scores (diminishing returns past ~100-200)
    # Prior belief that ~2% of sessions are fraudulent. NOT a measured value: it only
    # positions the decision threshold (predict() flags the lowest-scoring 2% of the
    # training sessions). It does not change the trees or the raw scores.
    contamination=0.02,
    random_state=42,        # reproducible runs
    n_jobs=-1               # use all CPU cores
)
model.fit(X_scaled)
joblib.dump(model, MODEL_OUT)
print(f"✅ Saved model -> {MODEL_OUT}")
print("Done training successfully.")

