"""Feature engineering shared by training and scoring.

Raw Avazu-style click logs (one row per impression) are aggregated into one
behavioural profile per device. The same function runs at training time and
when a user uploads a raw log, so the model always sees identically built
features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

C_COLUMNS = [f"c{i}" for i in range(14, 22)]

# Every column the aggregation can produce. Training drops the ones that turn
# out constant; the fitted model's feature list is the source of truth.
SESSION_FEATURES = [
    "impressions",
    "clicks",
    "click_rate",
    "unique_sites",
    "unique_apps",
    "device_type",
    "device_conn_type",
    *[f"{c}_nunique" for c in C_COLUMNS],
]

# Columns that identify a raw click log, in order of preference for the device key.
DEVICE_KEYS = ["device_id", "device_ip"]
ID_COLUMNS = ["session_id", "user_id", "device_id", "device_ip", "id"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def is_click_log(df: pd.DataFrame) -> bool:
    """A raw log has a device key and a per-row click flag."""
    return "click" in df.columns and any(k in df.columns for k in DEVICE_KEYS)


def _mode_per_group(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    counts = df.groupby([key, col], sort=False).size().reset_index(name="n")
    counts = counts.sort_values([key, "n"], ascending=[True, False])
    top = counts.drop_duplicates(key).set_index(key)[col]
    return pd.to_numeric(top, errors="coerce").fillna(-1).astype(int)


def aggregate_click_log(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate an impression-level log into one row per device.

    Returns the per-device feature frame (indexed by device key) and a list of
    human-readable warnings about columns that were missing and defaulted.
    """
    df = normalise_columns(df)
    key = next(k for k in DEVICE_KEYS if k in df.columns)
    df[key] = df[key].astype(str)
    df["click"] = pd.to_numeric(df["click"], errors="coerce").fillna(0)

    warnings: list[str] = []
    groups = df.groupby(key, sort=False)
    out = pd.DataFrame(index=groups.size().index)
    out["impressions"] = groups.size()
    out["clicks"] = groups["click"].sum()
    out["click_rate"] = out["clicks"] / out["impressions"]

    for col, feat in [("site_id", "unique_sites"), ("app_id", "unique_apps")]:
        if col in df.columns:
            out[feat] = groups[col].nunique()
        else:
            out[feat] = 1
            warnings.append(f"Column '{col}' not found; assumed a single value per device.")

    for col in ["device_type", "device_conn_type"]:
        if col in df.columns:
            out[col] = _mode_per_group(df, key, col)
        else:
            out[col] = -1
            warnings.append(f"Column '{col}' not found; treated as unknown.")

    missing_c = [c for c in C_COLUMNS if c not in df.columns]
    for c in C_COLUMNS:
        out[f"{c}_nunique"] = groups[c].nunique() if c in df.columns else 1
    if missing_c:
        warnings.append(
            f"Anonymised columns {', '.join(missing_c).upper()} not found; assumed a single value per device."
        )

    out.index.name = "entity_id"
    return out, warnings


def prepare_session_features(df: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Validate an already-aggregated session feature file.

    Derives click_rate when it can, and raises ValueError listing what is
    missing otherwise.
    """
    df = normalise_columns(df)
    warnings: list[str] = []

    if "click_rate" in required and "click_rate" not in df.columns and {"clicks", "impressions"} <= set(df.columns):
        imps = pd.to_numeric(df["impressions"], errors="coerce").replace(0, np.nan)
        df["click_rate"] = (pd.to_numeric(df["clicks"], errors="coerce") / imps).fillna(0)
        warnings.append("Derived click_rate from clicks / impressions.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "File is neither a raw click log (needs 'click' and 'device_id' or 'device_ip') "
            f"nor a session feature file. Missing feature columns: {', '.join(missing)}."
        )

    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    if id_col:
        df.index = df[id_col].astype(str)
    else:
        df.index = [f"row-{i + 1}" for i in range(len(df))]
    df.index.name = "entity_id"

    feats = df[required].apply(pd.to_numeric, errors="coerce")
    bad = int(feats.isna().any(axis=1).sum())
    if bad:
        warnings.append(f"{bad:,} row(s) had non-numeric values; those cells were treated as 0.")
    return feats.fillna(0), warnings
