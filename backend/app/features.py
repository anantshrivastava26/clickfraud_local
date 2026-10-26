"""Feature engineering shared by training and scoring.

Raw Avazu-style click logs (one row per impression) are aggregated into one
behavioural profile per device. The same function runs at training time and
when a user uploads a raw log, so the model always sees identically built
features.

Design notes
  * A "session" here is one device's whole activity in the file. Avazu has no session id
    and only hour-granular timestamps, so there is no time dimension: the features are
    counts, ratios and cardinalities. (An earlier version also computed timing features;
    they were constant on this data, so training dropped them.)
  * Missing optional columns never fail a scan. They get a neutral default (one distinct
    value, unknown device type) plus a warning, so absent data does not look anomalous.
  * Input that is neither a click log nor a session file raises ValueError; main.py turns
    that into HTTP 422 and the message is shown to the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Avazu's anonymised categorical columns C14..C21. Their meaning is unknown, but how many distinct
# values a device touches is a useful, meaning-agnostic breadth signal.
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
# Caveat: Avazu gives many anonymous users one shared placeholder device_id, which aggregates into a
# few huge 'devices' (see the README, 'Notes on the data').
DEVICE_KEYS = ["device_id", "device_ip"]
# For a session-feature file: the first of these columns found becomes the row label shown in the
# results.
ID_COLUMNS = ["session_id", "user_id", "device_id", "device_ip", "id"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lower-cased column names, so 'C14', 'c14' and ' Click ' all
    match. Copying leaves the caller's frame untouched.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def is_click_log(df: pd.DataFrame) -> bool:
    """A raw log has a device key and a per-row click flag."""
    return "click" in df.columns and any(k in df.columns for k in DEVICE_KEYS)


def _mode_per_group(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Most frequent value of `col` within each `key` group, as ints; ties go to the value seen
    first, and non-numeric values become -1 ('unknown'). Counting (key, value) pairs and keeping
    the top one per key is vectorised, avoiding a Python-level loop over every device.
    """
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
    # Group by the first device key present. Callers must check is_click_log() first: with no key at
    # all this raises StopIteration.
    key = next(k for k in DEVICE_KEYS if k in df.columns)
    # Ids can be parsed as numbers; keep them as text so they group consistently and display
    # unchanged.
    df[key] = df[key].astype(str)
    # Non-numeric click flags count as 'not clicked'.
    df["click"] = pd.to_numeric(df["click"], errors="coerce").fillna(0)

    warnings: list[str] = []
    # sort=False keeps devices in order of first appearance.
    groups = df.groupby(key, sort=False)
    out = pd.DataFrame(index=groups.size().index)
    # One row of the log is one impression, so a device's impressions are its row count.
    out["impressions"] = groups.size()
    out["clicks"] = groups["click"].sum()
    out["click_rate"] = out["clicks"] / out["impressions"]

    # Breadth features. If the column is absent, assume a single distinct value: that is the typical
    # value in the training data, so missing data is not mistaken for suspicious behaviour (the
    # warning tells the user it happened).
    for col, feat in [("site_id", "unique_sites"), ("app_id", "unique_apps")]:
        if col in df.columns:
            out[feat] = groups[col].nunique()
        else:
            out[feat] = 1
            warnings.append(f"Column '{col}' not found; assumed a single value per device.")

    # Dominant device type per device. -1 marks 'unknown' when the column is absent.
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

    # 'entity_id' is the label the UI shows: a device id for click logs, a session id or row number
    # for session files.
    out.index.name = "entity_id"
    return out, warnings


def prepare_session_features(df: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Validate an already-aggregated session feature file.

    Derives click_rate when it can, and raises ValueError listing what is
    missing otherwise.
    """
    df = normalise_columns(df)
    warnings: list[str] = []

    # Convenience: accept files that have clicks and impressions but no ratio. Zero impressions
    # become NaN first, so the division cannot produce infinity.
    if "click_rate" in required and "click_rate" not in df.columns and {"clicks", "impressions"} <= set(df.columns):
        imps = pd.to_numeric(df["impressions"], errors="coerce").replace(0, np.nan)
        df["click_rate"] = (pd.to_numeric(df["clicks"], errors="coerce") / imps).fillna(0)
        warnings.append("Derived click_rate from clicks / impressions.")

    # `required` is the model's own feature list, so a session file must contain exactly the columns
    # the model was trained on (extra columns are ignored).
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "File is neither a raw click log (needs 'click' and 'device_id' or 'device_ip') "
            f"nor a session feature file. Missing feature columns: {', '.join(missing)}."
        )

    # Label rows by the first id-like column found, otherwise by row number.
    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    if id_col:
        df.index = df[id_col].astype(str)
    else:
        df.index = [f"row-{i + 1}" for i in range(len(df))]
    df.index.name = "entity_id"

    # Select the model's columns in model order; non-numeric cells become NaN, then 0 (and are
    # counted in the warning below).
    feats = df[required].apply(pd.to_numeric, errors="coerce")
    bad = int(feats.isna().any(axis=1).sum())
    if bad:
        warnings.append(f"{bad:,} row(s) had non-numeric values; those cells were treated as 0.")
    return feats.fillna(0), warnings
