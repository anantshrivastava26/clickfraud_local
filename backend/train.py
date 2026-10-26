"""Train the IsolationForest click-fraud detector locally.

Usage:
    python train.py                                  # retrain from artifacts/session_features.csv
    python train.py --clicks path/to/avazu_train.csv # build features from a raw click log first

Pipeline: (optionally) aggregate a raw click log into per-device features -> drop constant
features -> standardise -> fit the IsolationForest -> write three files to artifacts/:

    scaler.joblib           {"scaler": StandardScaler, "features": [ordered names]}
    isoforest_model.joblib  the fitted model
    model_meta.json         provenance shown in the UI (never used for scoring)

app/detector.py loads exactly these files, so after retraining restart the API (or rebuild
the Docker image: the artifacts are baked into it). Features are built by app/features.py,
the same code the API runs on uploads, so training and scoring cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.features import SESSION_FEATURES, aggregate_click_log

ARTIFACTS = Path(__file__).parent / "artifacts"
FEATURES_CSV = ARTIFACTS / "session_features.csv"
MODEL_OUT = ARTIFACTS / "isoforest_model.joblib"
SCALER_OUT = ARTIFACTS / "scaler.joblib"
META_OUT = ARTIFACTS / "model_meta.json"

# Model settings. CONTAMINATION is the assumed share of fraudulent sessions, not a measured one: it
# only positions the decision boundary (the lowest-scoring 2% of training sessions are flagged) and
# does not change the trees. RANDOM_STATE makes retraining reproducible.
N_ESTIMATORS = 200
CONTAMINATION = 0.02
RANDOM_STATE = 42


def build_features_from_clicks(path: Path, chunk_size: int) -> pd.DataFrame:
    """Aggregate a large click log into one row per device without loading it all at once. Each
    chunk is aggregated on its own (dtype=str so ids are never mangled into numbers) and the
    results are concatenated. Limitation: a device whose rows span several chunks yields one
    partial row per chunk instead of one merged row, which understates very large devices.
    """
    frames = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, dtype=str, low_memory=False)):
        feats, _ = aggregate_click_log(chunk)
        frames.append(feats.reset_index(drop=True))
        print(f"  chunk {i + 1}: {len(chunk):,} impressions -> {len(feats):,} device sessions")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", type=Path, help="Raw Avazu-format click log to build features from")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    if args.clicks:
        print(f"Building session features from {args.clicks} ...")
        feat_df = build_features_from_clicks(args.clicks, args.chunk_size)
        feat_df.to_csv(FEATURES_CSV, index=False)
        print(f"Wrote {len(feat_df):,} sessions -> {FEATURES_CSV}")
    else:
        feat_df = pd.read_csv(FEATURES_CSV)
        print(f"Loaded {len(feat_df):,} sessions from {FEATURES_CSV}")

    # Train only on the features the API can rebuild (SESSION_FEATURES). This also ignores extra
    # columns in an older session_features.csv, such as the timing features.
    X = feat_df[[c for c in SESSION_FEATURES if c in feat_df.columns]].fillna(0).astype(float)
    # A feature with zero variance carries no information (the forest can never split on it).
    # Dropping it here, and recording it in the metadata, means the saved feature list is exactly
    # what the model uses.
    dropped = [c for c in X.columns if X[c].std() == 0]
    X = X.drop(columns=dropped)
    feature_names = list(X.columns)
    if dropped:
        print(f"Dropped constant features: {', '.join(dropped)}")

    # Standardise so every feature is on one scale. IsolationForest itself does not need this, but
    # the same scaler is applied at scoring time, and its output doubles as the z-scores behind each
    # row's 'signals' (detector._signals).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training IsolationForest on {len(X):,} sessions x {len(feature_names)} features ...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    # Sanity check: with contamination=0.02 about 2% of the training sessions should come out
    # flagged.
    flagged = int((model.predict(X_scaled) == -1).sum())

    # The scaler and its feature list are saved together. That list is the contract telling the
    # detector which columns, in which order, the scaler and model expect.
    joblib.dump({"scaler": scaler, "features": feature_names}, SCALER_OUT)
    joblib.dump(model, MODEL_OUT)
    # scikit-learn is recorded because joblib files are sensitive to the version that wrote them;
    # the UI shows it. Only the file name of --clicks is stored, not its full path.
    META_OUT.write_text(
        json.dumps(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sklearn_version": sklearn.__version__,
                "training_rows": len(X),
                "features": feature_names,
                "dropped_constant_features": dropped,
                "n_estimators": N_ESTIMATORS,
                "contamination": CONTAMINATION,
                "random_state": RANDOM_STATE,
                "source": str(args.clicks.name) if args.clicks else FEATURES_CSV.name,
            },
            indent=2,
        )
    )
    print(f"Flagged {flagged:,} of {len(X):,} training sessions ({flagged / len(X):.2%}).")
    print(f"Saved {MODEL_OUT.name}, {SCALER_OUT.name}, {META_OUT.name}")


if __name__ == "__main__":
    main()
