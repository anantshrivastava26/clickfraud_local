"""Loads the trained IsolationForest and scores session feature frames."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import aggregate_click_log, is_click_log, normalise_columns, prepare_session_features

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Sessions that are not flagged by the model but sit in the riskiest 5% of the
# training distribution are surfaced for manual review.
REVIEW_QUANTILE = 0.05
# A feature counts as a signal when it is this many standard deviations from the training mean.
SIGNAL_Z = 2.5
MAX_SIGNALS = 3
MAX_RESULT_ROWS = 5000
RISK_BOT = 70.0
RISK_REVIEW = 40.0
HIST_BINS = 40

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
    frame: pd.DataFrame  # original features + score, risk, verdict
    scaled: np.ndarray


class Detector:
    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        blob = joblib.load(artifacts / "scaler.joblib")
        self.scaler = blob["scaler"]
        self.features: list[str] = list(blob["features"])
        self.model = joblib.load(artifacts / "isoforest_model.joblib")
        meta_path = artifacts / "model_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        train = pd.read_csv(artifacts / "session_features.csv")
        train_x = train[self.features].fillna(0).astype(float)
        train_scores = self.model.decision_function(self.scaler.transform(train_x))

        self.review_threshold = float(np.quantile(train_scores, REVIEW_QUANTILE))
        # Piecewise-linear map from decision score to a 0-100 risk score, anchored
        # so the model's own cut-offs land on round numbers: <0 is bot (risk >= 70),
        # below the review quantile is review (40-69), everything else human (< 40).
        self._risk_anchors = (
            [float(train_scores.min()), 0.0, self.review_threshold, float(train_scores.max())],
            [100.0, RISK_BOT, RISK_REVIEW, 0.0],
        )
        self.feature_medians = train_x.median()
        lo, hi = float(train_scores.min()), float(train_scores.max())
        self.bin_edges = np.linspace(lo, hi, HIST_BINS + 1)

        self.training = self._score_frame(train_x)
        self.summary = self._training_summary()

    # ------------------------------------------------------------------ scoring

    def _risk(self, scores: np.ndarray) -> np.ndarray:
        xs, ys = self._risk_anchors
        return np.round(np.interp(scores, xs, ys), 1)

    def _score_frame(self, x: pd.DataFrame) -> Scored:
        scaled = self.scaler.transform(x[self.features].astype(float))
        scores = self.model.decision_function(scaled)
        is_bot = self.model.predict(scaled) == -1
        verdict = np.where(is_bot, "bot", np.where(scores < self.review_threshold, "review", "human"))
        frame = x.copy()
        frame["score"] = scores
        frame["risk"] = self._risk(scores)
        frame["verdict"] = verdict
        return Scored(frame=frame, scaled=scaled)

    def _signals(self, scaled_row: np.ndarray, raw_row: pd.Series) -> list[dict]:
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
        frame = scored.frame
        order = np.argsort(frame["score"].values, kind="stable")[:limit]
        rows = []
        for i in order:
            r = frame.iloc[i]
            rows.append(
                {
                    "id": str(frame.index[i]),
                    "verdict": r["verdict"],
                    "risk": float(r["risk"]),
                    "score": round(float(r["score"]), 4),
                    "impressions": _num(r.get("impressions", 0)),
                    "clicks": _num(r.get("clicks", 0)),
                    "click_rate": _num(r.get("click_rate", 0)),
                    "signals": self._signals(scored.scaled[i], r),
                    "features": {f: _num(r[f]) for f in self.features},
                }
            )
        return rows

    def _histogram(self, frame: pd.DataFrame) -> list[dict]:
        scores = np.clip(frame["score"].values, self.bin_edges[0], self.bin_edges[-1])
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
        df = normalise_columns(raw)
        if df.empty:
            raise ValueError("The file has a header but no rows.")
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
    f = float(v)
    if not np.isfinite(f):
        return 0
    return int(f) if f.is_integer() else round(f, 4)
