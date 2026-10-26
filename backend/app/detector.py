"""Loads the trained IsolationForest and scores session feature frames.

A Detector is created once at startup (app/main.py) and afterwards only read. It turns the
model's raw output into the three things the UI shows:

  verdict  "bot"     the model's own decision (decision score < 0)
           "review"  not flagged, but among the riskiest REVIEW_QUANTILE of the training data
           "human"   everything else - it only means "not flagged"
  risk     0-100 rescaling of the decision score, chosen so the verdict cut-offs land on
           round numbers (bot >= 70, review >= 40); see Detector.__init__
  signals  up to MAX_SIGNALS features furthest from the training mean, so a reviewer can
           see why a row scored the way it did

About the score: IsolationForest.decision_function is higher = more normal and negative =
anomalous; 0 is the model's decision boundary, placed at training time by `contamination`.

Everything is calibrated against the training distribution, never against the uploaded
file, so a given session gets the same score no matter what else is in the upload.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import aggregate_click_log, is_click_log, normalise_columns, prepare_session_features

# Resolved relative to this file, not the working directory, so the API finds its model files
# wherever uvicorn is started from. The Dockerfile copies artifacts/ next to app/.
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Sessions that are not flagged by the model but sit in the riskiest 5% of the
# training distribution are surfaced for manual review.
REVIEW_QUANTILE = 0.05
# A feature counts as a signal when it is this many standard deviations from the training mean.
SIGNAL_Z = 2.5
MAX_SIGNALS = 3
# Cap on rows returned per scan, to keep responses and the browser table manageable. Rows are sorted
# most-anomalous first, so truncation only drops the least interesting ones; counts and the
# histogram always cover the whole file (the `truncated` flag tells the UI).
MAX_RESULT_ROWS = 5000
# Risk bands: risk >= RISK_BOT is a bot, >= RISK_REVIEW needs review. They are the fixed anchor
# values of the score-to-risk mapping built in Detector.__init__.
RISK_BOT = 70.0
RISK_REVIEW = 40.0
# Number of equal-width score buckets in the histogram sent to the UI.
HIST_BINS = 40

# Display names for the UI. A feature missing here falls back to its column name.
FEATURE_LABELS = {
    "impressions": "Impressions",
    "clicks": "Clicks",
    "click_rate": "Click-through rate",
    "unique_sites": "Distinct sites",
    "unique_apps": "Distinct apps",
    "device_type": "Device type",
    "device_conn_type": "Connection type",
    **{f"c{i}_nunique": f"Distinct C{i} values" for i in range(14, 22)},
}


@dataclass
class Scored:
    """Scored rows plus the standardised matrix they were scored on. The standardised values are
    kept because the z-scores behind each row's 'signals' are read from them, which avoids
    transforming twice.
    """

    frame: pd.DataFrame  # original features + score, risk, verdict
    scaled: np.ndarray


class Detector:
    """Scoring service wrapped around the saved scaler, the model and the training feature table."""

    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        """Load the model files and pre-compute everything that does not depend on an upload.

        artifacts/ must contain scaler.joblib ({"scaler": ..., "features": [...]}),
        isoforest_model.joblib and session_features.csv (the training table); model_meta.json is
        optional. A missing required file raises here, so a bad deployment fails at startup
        rather than on the first request.
        """
        blob = joblib.load(artifacts / "scaler.joblib")
        self.scaler = blob["scaler"]
        # The ordered column list stored with the scaler is the contract between training and
        # scoring: every frame is reduced to exactly these columns, in this order, before scaling.
        self.features: list[str] = list(blob["features"])
        self.model = joblib.load(artifacts / "isoforest_model.joblib")
        # Optional provenance (training date, scikit-learn version, source file) shown in the UI.
        # Scoring never depends on it.
        meta_path = artifacts / "model_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Re-score the training table at startup instead of storing precomputed numbers: the
        # thresholds, histogram and feature profile below then always match the loaded model and
        # scaler exactly, at the cost of a moment of startup time. The CSV may hold extra columns
        # (older versions had timing features); only the model's features are used.
        train = pd.read_csv(artifacts / "session_features.csv")
        train_x = train[self.features].fillna(0).astype(float)
        train_scores = self.model.decision_function(self.scaler.transform(train_x))

        # The 'review' band starts at the REVIEW_QUANTILE of the training scores. The model itself
        # flags only about `contamination` (2%) of sessions as bots; this surfaces the next most
        # suspicious slice instead of calling it clean.
        self.review_threshold = float(np.quantile(train_scores, REVIEW_QUANTILE))
        # Piecewise-linear map from decision score to a 0-100 risk score, anchored
        # so the model's own cut-offs land on round numbers: <0 is bot (risk >= 70),
        # below the review quantile is review (40-69), everything else human (< 40).
        # NOTE: np.interp needs increasing x values, i.e. min < 0 < review_threshold < max. That
        # holds only while REVIEW_QUANTILE is larger than the model's contamination (score 0 sits at
        # the `contamination` percentile of the training scores), so keep that in mind when changing
        # either. Scores outside the training range clamp to risk 100 or 0.
        self._risk_anchors = (
            [float(train_scores.min()), 0.0, self.review_threshold, float(train_scores.max())],
            [100.0, RISK_BOT, RISK_REVIEW, 0.0],
        )
        # Training median per feature, shown beside a row's value as the 'typical' level in its
        # signals.
        self.feature_medians = train_x.median()
        # Histogram edges are fixed from the training score range, so every scan is bucketed on the
        # same axis and charts are comparable between files. Scores beyond the range are clipped
        # into the first or last bucket.
        lo, hi = float(train_scores.min()), float(train_scores.max())
        self.bin_edges = np.linspace(lo, hi, HIST_BINS + 1)

        # Score the training table once; its summary (counts, histogram, profile, top anomalies) is
        # served as-is by GET /api/model.
        self.training = self._score_frame(train_x)
        self.summary = self._training_summary()

    # ------------------------------------------------------------------ scoring

    def _risk(self, scores: np.ndarray) -> np.ndarray:
        """Map decision scores to a 0-100 risk, rounded to one decimal."""
        xs, ys = self._risk_anchors
        return np.round(np.interp(scores, xs, ys), 1)

    def _score_frame(self, x: pd.DataFrame) -> Scored:
        """Standardise, score and label every row of `x`. Selecting x[self.features] both drops
        extra columns and fixes the column order the scaler expects.
        """
        scaled = self.scaler.transform(x[self.features].astype(float))
        scores = self.model.decision_function(scaled)
        # predict() is the model's own verdict: -1 (outlier) exactly when the decision score is
        # below 0.
        is_bot = self.model.predict(scaled) == -1
        # Precedence: bot first, then review (only for rows not flagged), otherwise human.
        verdict = np.where(is_bot, "bot", np.where(scores < self.review_threshold, "review", "human"))
        frame = x.copy()
        frame["score"] = scores
        frame["risk"] = self._risk(scores)
        frame["verdict"] = verdict
        return Scored(frame=frame, scaled=scaled)

    def _signals(self, scaled_row: np.ndarray, raw_row: pd.Series) -> list[dict]:
        """Explain a row by its most extreme features. `scaled_row` is StandardScaler output, i.e.
        standard deviations from the training mean, so z reads directly as 'how many sigmas off'
        (negative = unusually low). Features are visited by descending |z|, so the loop can stop
        at the first one below SIGNAL_Z. Values shown to the user are the raw ones; 'typical' is
        the training median.
        """
        order = np.argsort(-np.abs(scaled_row))
        out = []
        for idx in order[:MAX_SIGNALS]:
            z = float(scaled_row[idx])
            if abs(z) < SIGNAL_Z:
                break
            name = self.features[idx]
            out.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": _num(raw_row[name]),
                    "typical": _num(self.feature_medians[name]),
                    "z": round(z, 1),
                }
            )
        return out

    def _rows(self, scored: Scored, limit: int) -> list[dict]:
        """Per-row result objects, most anomalous first. A stable sort keeps input order among equal
        scores, so output is deterministic.
        """
        frame = scored.frame
        order = np.argsort(frame["score"].values, kind="stable")[:limit]
        rows = []
        for i in order:
            r = frame.iloc[i]
            rows.append(
                {
                    # The index is the entity id: a device id, a session id, or row-N (see
                    # features.py).
                    "id": str(frame.index[i]),
                    "verdict": r["verdict"],
                    "risk": float(r["risk"]),
                    "score": round(float(r["score"]), 4),
                    "impressions": _num(r.get("impressions", 0)),
                    "clicks": _num(r.get("clicks", 0)),
                    "click_rate": _num(r.get("click_rate", 0)),
                    "signals": self._signals(scored.scaled[i], r),
                    # Every model input for this row, used by the UI's detail drawer.
                    "features": {f: _num(r[f]) for f in self.features},
                }
            )
        return rows

    def _histogram(self, frame: pd.DataFrame) -> list[dict]:
        """Counts per score bucket, split by verdict so the UI can stack them. Out-of-range scores
        are clipped into the end buckets (see bin_edges).
        """
        scores = np.clip(frame["score"].values, self.bin_edges[0], self.bin_edges[-1])
        # digitize is 1-based, hence the -1. The upper clip keeps a score exactly on the last edge
        # in the final bucket.
        idx = np.clip(np.digitize(scores, self.bin_edges) - 1, 0, HIST_BINS - 1)
        bins = []
        for b in range(HIST_BINS):
            in_bin = idx == b
            verdicts = frame["verdict"].values[in_bin]
            bins.append(
                {
                    "x0": round(float(self.bin_edges[b]), 4),
                    "x1": round(float(self.bin_edges[b + 1]), 4),
                    "bot": int((verdicts == "bot").sum()),
                    "review": int((verdicts == "review").sum()),
                    "human": int((verdicts == "human").sum()),
                }
            )
        return bins

    def _counts(self, frame: pd.DataFrame) -> dict:
        """Headline numbers. 'sessions' is the number of scored rows (devices, for a raw click log).
        bot_impressions and bot_clicks measure how much traffic comes from flagged rows - what
        the UI calls invalid traffic. A missing impressions or clicks column counts as 0.
        """
        v = frame["verdict"].value_counts()
        clicks = frame["clicks"] if "clicks" in frame else pd.Series(0, index=frame.index)
        imps = frame["impressions"] if "impressions" in frame else pd.Series(0, index=frame.index)
        bot = frame["verdict"] == "bot"
        return {
            "sessions": int(len(frame)),
            "bot": int(v.get("bot", 0)),
            "review": int(v.get("review", 0)),
            "human": int(v.get("human", 0)),
            "impressions": int(imps.sum()),
            "clicks": int(clicks.sum()),
            "bot_impressions": int(imps[bot].sum()),
            "bot_clicks": int(clicks[bot].sum()),
        }

    # ----------------------------------------------------------------- outputs

    def _training_summary(self) -> dict:
        """Everything GET /api/model returns. In the feature profile, 'human' means 'not flagged as
        a bot', so it includes the review sessions.
        """
        frame = self.training.frame
        bot = frame["verdict"] == "bot"
        profile = []
        for f in self.features:
            profile.append(
                {
                    "feature": f,
                    "label": FEATURE_LABELS.get(f, f),
                    "human_median": _num(frame.loc[~bot, f].median()),
                    "human_p95": _num(frame.loc[~bot, f].quantile(0.95)),
                    "bot_median": _num(frame.loc[bot, f].median()),
                    "bot_p95": _num(frame.loc[bot, f].quantile(0.95)),
                }
            )
        return {
            "model": {
                "algorithm": "IsolationForest",
                "n_estimators": int(self.model.n_estimators),
                "contamination": float(self.model.contamination),
                "max_samples": int(self.model.max_samples_),
                # IsolationForest's decision boundary is 0 by construction.
                "decision_threshold": 0.0,
                "review_threshold": round(self.review_threshold, 4),
                "risk_bands": {"bot": RISK_BOT, "review": RISK_REVIEW},
                "features": [{"name": f, "label": FEATURE_LABELS.get(f, f)} for f in self.features],
                "trained_at": self.meta.get("trained_at"),
                "sklearn_version": self.meta.get("sklearn_version"),
                "source": self.meta.get("source"),
            },
            "counts": self._counts(frame),
            "histogram": self._histogram(frame),
            "profile": profile,
            "top_anomalies": self._rows(self.training, 25),
        }

    def scan(self, raw: pd.DataFrame, filename: str) -> dict:
        """Score an uploaded file and return the whole API response. Raises ValueError for input the
        user can fix; main.py maps that to HTTP 422.
        """
        df = normalise_columns(raw)
        if df.empty:
            raise ValueError("The file has a header but no rows.")
        # Two accepted shapes: an impression-level raw log (aggregated per device here) or an
        # already-aggregated session file (validated, then scored as it is).
        if is_click_log(df):
            mode = "click_log"
            feats, warnings = aggregate_click_log(df)
        else:
            mode = "sessions"
            feats, warnings = prepare_session_features(df, self.features)

        scored = self._score_frame(feats)
        rows = self._rows(scored, MAX_RESULT_ROWS)
        return {
            "file": filename,
            "mode": mode,
            "input_rows": int(len(df)),
            "warnings": warnings,
            "counts": self._counts(scored.frame),
            "histogram": self._histogram(scored.frame),
            "results": rows,
            "truncated": len(scored.frame) > len(rows),
        }


def _num(v) -> float | int:
    """Make a value JSON-safe: numpy scalars become plain numbers, NaN and infinity (which JSON
    cannot represent) become 0, whole numbers stay ints, and everything else is rounded to 4
    decimals.
    """
    f = float(v)
    if not np.isfinite(f):
        return 0
    return int(f) if f.is_integer() else round(f, 4)
