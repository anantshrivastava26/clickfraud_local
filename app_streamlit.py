# app_streamlit.py
"""
Streamlit dashboard: the front-end of the demo.

Run:  streamlit run app_streamlit.py     (from the repo root; paths are relative)

Sections, top to bottom
-----------------------
  1. Headline metrics        re-scores every training session with the saved model
  2. Human vs Bot simulator  hand-built feature vectors pushed through the model
  3. Session explorer        filterable table of the scored training sessions
  4. Batch upload            score your own CSV of session features
  5. Score histogram + PCA   the same charts as eval_and_visualize.py
  6. Live feed               tails scored_sessions.jsonl, written by receiver_fastapi.py

Data flow: this app only READS files (scaler.joblib, isoforest_model.joblib,
session_features.csv, scored_sessions.jsonl). It never talks to the receiver over HTTP,
so the dashboard still works with the receiver stopped; the live feed just goes quiet.

Streamlit re-executes this whole script on every interaction AND on every auto-refresh
tick (section 6), so anything expensive at module level is repeated. Only the model/data
loading is cached (st.cache_resource); the scoring and PCA below are recomputed each time.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
SCALER_OUT = "scaler.joblib"
MODEL_OUT = "isoforest_model.joblib"
FEATURES_OUT = "session_features.csv"

st.set_page_config(page_title="Click Fraud Detection Dashboard", page_icon="🚨", layout="wide")

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
# cache_resource keeps the unpickled model/scaler/DataFrame in memory across reruns
# instead of reading ~50k CSV rows and two joblib files on every click.
@st.cache_resource
def load_model_and_data():
    """Load scaler, model, and dataset"""
    scaler_data = joblib.load(SCALER_OUT)

    # Same two scaler.joblib formats as the other scripts: the current dict (scaler +
    # trained feature list) or a legacy bare scaler, whose features we have to guess.
    if isinstance(scaler_data, dict):
        scaler = scaler_data["scaler"]
        feature_names = scaler_data["features"]
    else:
        scaler = scaler_data
        # fallback: read from CSV if available
        try:
            df_tmp = pd.read_csv(FEATURES_OUT, nrows=1)
            feature_names = list(df_tmp.select_dtypes(include=[np.number]).columns)
        except Exception:
            feature_names = getattr(scaler, "feature_names_in_", [])

    model = joblib.load(MODEL_OUT)
    df = pd.read_csv(FEATURES_OUT)
    return scaler, feature_names, model, df


scaler, feature_names, model, df = load_model_and_data()

st.title("🧠 Click Fraud Detection Dashboard")
st.write("Unsupervised anomaly detection using IsolationForest on Avazu-style session features.")

# -----------------------------
# PREPROCESS DATA
# -----------------------------
# Same recipe as eval_and_visualize.py: keep the trained columns, in trained order, then
# scale and score. `df` is the cached object, so these columns are re-assigned each rerun.
X = df.select_dtypes(include=[np.number]).fillna(0)
if feature_names is not None:
    X = X.reindex(columns=feature_names, fill_value=0)

X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)                           # -1 = outlier, 1 = normal
df["is_outlier"] = (pred == -1).astype(int)
df["anomaly_score"] = model.decision_function(X_scaled)  # higher = more normal

# -----------------------------
# METRICS
# -----------------------------
colA, colB, colC = st.columns(3)
colA.metric("Total sessions", len(df))
colB.metric("Suspicious sessions", int(df["is_outlier"].sum()))

# "Fraud probability" = 1 - min-max-scaled anomaly score: the most anomalous session in
# the table gets 1.0, the most normal 0.0. It is a RELATIVE ranking inside this table,
# not a calibrated probability, and it shifts whenever the table changes.
score_scaler = MinMaxScaler(feature_range=(0, 1))
df["fraud_probability"] = 1 - score_scaler.fit_transform(df[["anomaly_score"]])
colC.metric("Avg Fraud Probability", f"{df['fraud_probability'].mean():.2f}")


# -----------------------------
# SIMULATOR
# -----------------------------
st.subheader("🧍 Human vs 🤖 Bot Click Simulation")
col1, col2 = st.columns(2)


def preprocess_input(sample_dict):
    """Ensure columns match scaler training features"""
    x = pd.DataFrame([sample_dict])
    if feature_names is not None:
        # Drops any key the model was not trained on and zero-fills any it expects.
        x = x.reindex(columns=feature_names, fill_value=0)
    return x


# The two buttons below score hard-coded example sessions: a "typical" one and an
# obviously automated one. They show what the model reacts to, independent of the live
# receiver. (Only the volume / ratio / breadth fields matter; the timing fields are
# discarded by preprocess_input because the model was not trained on them.)
with col1:
    st.markdown("### Human-like Session")
    if st.button("👆 Simulate Human Click"):
        human = {
            "impressions": 5,
            "clicks": 1,
            "click_rate": 1 / 5,
            "session_length_secs": 180,
            "unique_sites": 1,
            "unique_apps": 1,
            "device_type": 1,
            "device_conn_type": 1,
            "ua_len": 80,
            "hour_mean": datetime.datetime.now(datetime.UTC).hour,
            "hour_std": 0.5,
            "interclick_mean": 45,
            "interclick_std": 10,
            "c14_nunique": 0,
            "c15_nunique": 0,
            "c16_nunique": 0,
            "c17_nunique": 0,
            "c18_nunique": 0,
            "c19_nunique": 0,
            "c20_nunique": 0,
            "c21_nunique": 0,
        }

        x = preprocess_input(human)
        x_scaled = scaler.transform(x)
        score = model.decision_function(x_scaled)[0]
        is_out = model.predict(x_scaled)[0] == -1
        st.success(f"Prediction: {'Suspicious' if is_out else 'Normal'} (score {score:.3f})")


with col2:
    st.markdown("### Bot-like Session")
    if st.button("🤖 Simulate Bot Click"):
        # 90% click rate, 20 sites / 10 apps in one minute, 1 s between clicks, 3 a.m.
        bot = {
            "impressions": 50,
            "clicks": 45,
            "click_rate": 45 / 50,
            "session_length_secs": 60,
            "unique_sites": 20,
            "unique_apps": 10,
            "device_type": 0,
            "device_conn_type": 0,
            "ua_len": 12,
            "hour_mean": 3,
            "hour_std": 0.1,
            "interclick_mean": 1,
            "interclick_std": 0.5,
            "c14_nunique": 5,
            "c15_nunique": 5,
            "c16_nunique": 5,
            "c17_nunique": 5,
            "c18_nunique": 5,
            "c19_nunique": 5,
            "c20_nunique": 5,
            "c21_nunique": 5,
        }

        x = preprocess_input(bot)
        x_scaled = scaler.transform(x)
        score = model.decision_function(x_scaled)[0]
        is_out = model.predict(x_scaled)[0] == -1
        st.error(f"Prediction: {'Suspicious' if is_out else 'Normal'} (score {score:.3f})")

# -----------------------------
# INTERACTIVE FILTER + TABLE
# -----------------------------
st.subheader("🔍 Explore Session Data")

filter_choice = st.radio("Show sessions:", ["All", "Suspicious only", "Normal only"], horizontal=True)
if filter_choice == "Suspicious only":
    filtered = df[df["is_outlier"] == 1]
elif filter_choice == "Normal only":
    filtered = df[df["is_outlier"] == 0]
else:
    filtered = df

# Only the first 50 rows are rendered: drawing all ~50k would make the page sluggish.
st.dataframe(filtered.head(50))

# -----------------------------
# BATCH UPLOAD TESTING
# -----------------------------
st.subheader("📤 Upload Session Features for Batch Scoring")
uploaded = st.file_uploader("Choose CSV file", type="csv")

if uploaded:
    udf = pd.read_csv(uploaded)
    # Columns the model was not trained on are dropped, missing ones become 0, so the
    # upload only needs to contain (a superset of) the trained features.
    udf = udf.reindex(columns=feature_names, fill_value=0)
    u_scaled = scaler.transform(udf)
    preds = model.predict(u_scaled)
    udf["is_outlier"] = (preds == -1).astype(int)
    udf["anomaly_score"] = model.decision_function(u_scaled)
    # NOTE: this RE-FITS the min-max scaler on the uploaded batch, so these values are
    # relative to this upload only and not comparable with the main table above.
    udf["fraud_probability"] = 1 - score_scaler.fit_transform(udf[["anomaly_score"]])
    st.write(udf.head())
    st.metric("Suspicious count", int(udf["is_outlier"].sum()))

    # Plot upload distribution
    fig3, ax3 = plt.subplots()
    ax3.hist(udf["anomaly_score"], bins=40, color='orange', edgecolor='white', alpha=0.7)
    ax3.axvline(udf["anomaly_score"].mean(), color='red', linestyle='--', label='Mean')
    ax3.legend()
    ax3.set_title("Uploaded Dataset Anomaly Score Distribution")
    st.pyplot(fig3)


# -----------------------------
# VISUALIZATION: ANOMALY DISTRIBUTION
# -----------------------------
st.subheader("📉 Anomaly Score Distribution")

fig, ax = plt.subplots(figsize=(6, 3))  # width=6 inches, height=3 inches

ax.hist(df["anomaly_score"], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(df["anomaly_score"].mean(), color='red', linestyle='--', label='Mean Score')
ax.legend()
ax.set_xlabel("Anomaly Score (higher = normal)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Session Anomaly Scores", fontsize=10)
st.pyplot(fig, use_container_width=False)


# -----------------------------
# VISUALIZATION: PCA SCATTER
# -----------------------------
st.subheader("🧠 PCA Visualization of Sessions")

# 2-D projection for display only (the model itself does not use PCA). It is refitted on
# ~50k rows on every rerun, including each auto-refresh tick below; wrap it in
# st.cache_data if the page ever feels slow.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig2, ax2 =plt.subplots(figsize=(6, 3))
scatter = ax2.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["is_outlier"],
    cmap="coolwarm",
    alpha=0.6,
    s=15
)
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_title("Red = Suspicious | Blue = Normal", fontsize=10)
st.pyplot(fig2, use_container_width=False)
# -----------------------------
# LIVE FEED: REAL-TIME DETECTIONS
# -----------------------------
# NOTE: `from datetime import datetime` below rebinds the name `datetime` from the
# module (imported at the top, used by the simulator above) to the class. That is safe
# only because the simulator code runs earlier in the script; keep it that way or alias one.
import json, os, streamlit_autorefresh
from datetime import datetime
from streamlit_autorefresh import st_autorefresh  # ✅ add this at top of file too (pip install streamlit-autorefresh)

st.subheader("🚨 Live Fraud Detection Feed")

# Refresh every N seconds (non-blocking)
# st_autorefresh is a tiny browser-side timer that triggers a normal Streamlit rerun
# (so the whole script runs again - see the module docstring). A sleep loop would block
# the script and freeze the UI instead.
refresh_interval = st.slider("Auto-refresh interval (seconds)", 2, 15, 5)
st_autorefresh(interval=refresh_interval * 1000, key="fraud_feed_refresh")

def read_latest_detections(limit=30):
    """Read and parse the latest scored sessions"""
    # scored_sessions.jsonl is appended to by receiver_fastapi.py (one JSON object per
    # line) and is the only coupling between the two programs.
    if not os.path.exists("scored_sessions.jsonl"):
        return pd.DataFrame()

    rows = []
    with open("scored_sessions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                # The receiver may be mid-write, leaving a partial last line; skip it
                # rather than crash the whole dashboard.
                continue

    if not rows:
        return pd.DataFrame()

    df_live = pd.DataFrame(rows)
    df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], errors="coerce")
    # Newest first, capped at `limit`. The whole file is re-read every refresh, which is
    # fine for a demo log but would need tailing/rotation for a long-running one.
    df_live = df_live.sort_values("timestamp", ascending=False).head(limit)
    return df_live

# Load recent detections
live_df = read_latest_detections()

if not live_df.empty:
    suspicious_count = int((live_df["is_outlier"] == 1).sum())
    normal_count = int((live_df["is_outlier"] == 0).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Recent Suspicious", suspicious_count)
    col2.metric("Recent Normal", normal_count)
    col3.metric("Total Recent", len(live_df))

    # Color code suspicious rows
    def highlight_rows(row):
        color = "#ff0000" if row["is_outlier"] == 1 else "#00a100"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        live_df[["user_id", "timestamp", "score", "is_outlier", "event_count"]]
        .style.apply(highlight_rows, axis=1),
        use_container_width=True,
        height=300
    )

    # Mini trend chart
    # Bar chart of how many of the recent verdicts were Normal vs Suspicious.
    trend = live_df.groupby("is_outlier")["score"].count().rename({0: "Normal", 1: "Suspicious"})
    st.bar_chart(trend)

else:
    st.info("🕒 Waiting for detections from receiver_fastapi.py ...")
