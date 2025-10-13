# train_isolation_forest.py
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
LOCAL_TMP = "/tmp/avazu_sample.csv" 
CHUNK_SIZE = 200_000   # adjust to fit local memory; file has 1M rows
SESSION_GAP_MINUTES = 30
MODEL_OUT = "isoforest_model.joblib"
SCALER_OUT = "scaler.joblib"
FEATURES_OUT = "session_features.csv"

# Helper: read CSV from S3 into a local tmp file (streamed)
def download_from_s3(bucket, key, local_path):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded s3://{bucket}/{key} -> {local_path}")

# Sessionization + per-session aggregator
def process_chunks(csv_path, chunk_size=CHUNK_SIZE):
    # We'll build a list of session-level dictionaries
    sessions = []
    # Use these columns if available; if not present, code will still work
    parse_dates = False
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=str,           # ✅ Force all columns to string type
        low_memory=False     # ✅ Prevent chunk-wise type guessing
    ):

        # Standardize column names (lowercase)
        chunk.columns = [c.lower() for c in chunk.columns]
        # Ensure a timestamp column. Avazu has 'hour' as yyyymmddHH format integer/string.
        # Convert 'hour' to datetime if present
        if 'hour' in chunk.columns:
            # hour looks like '20130521 14' or 2013052114? Accept both:
            # If numeric like 2013052100 convert to string and parse first 10 digits.
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
            chunk['ts'] = pd.Timestamp.utcnow()
        
        # user identifier: prefer device_id else device_ip else C1
        if 'device_id' in chunk.columns:
            chunk['user_id'] = chunk['device_id'].astype(str)
        elif 'device_ip' in chunk.columns:
            chunk['user_id'] = chunk['device_ip'].astype(str)
        elif 'c1' in chunk.columns:
            chunk['user_id'] = chunk['c1'].astype(str)
        else:
            chunk['user_id'] = chunk.index.astype(str)

        # group per user and sessionize by gap
        chunk = chunk.sort_values(['user_id','ts'])
        for user, g in chunk.groupby('user_id'):
            prev_ts = None
            session_idx = 0
            acc = []
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
            if acc:
                sessions.append(aggregate_session(acc))
    return sessions

def safe_get(row, col, default=np.nan):
    return row[col] if col in row else default

def aggregate_session(rows):
    # rows: list of pandas Series (rows) belonging to same session
    df = pd.DataFrame(rows)

    # ✅ Ensure timestamps are datetime
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

    # Basic counts
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
    ua_len = df['device_model'].astype(str).apply(len).mean() if 'device_model' in df else 0

    # hour features
    hours = df['ts'].dt.hour.dropna()
    hour_mean = hours.mean() if not hours.empty else -1
    hour_std = hours.std() if not hours.empty else 0

    # C14-C21 stats if present
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
    feat_df.to_csv(FEATURES_OUT, index=False)
    print(f"Wrote session features to {FEATURES_OUT}")

    # 3) Prepare training data (drop suspicious or infinite values)
    X = feat_df.fillna(0).astype(float)
    # remove constant columns if any
    X = X.loc[:, X.std() > 0.0]

   # 4) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save both scaler and feature names
feature_names = list(X.columns)
joblib.dump({"scaler": scaler, "features": feature_names}, SCALER_OUT)
print(f"✅ Saved scaler + {len(feature_names)} features -> {SCALER_OUT}")
print("Feature names preview:", feature_names[:10])

# 5) Train IsolationForest
print("Training IsolationForest...")
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)
joblib.dump(model, MODEL_OUT)
print(f"✅ Saved model -> {MODEL_OUT}")
print("Done training successfully.")

