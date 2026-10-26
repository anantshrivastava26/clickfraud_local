"""Generate the downloadable sample CSVs served by the frontend.

    python make_samples.py

Writes to ../frontend/public/samples:
  avazu_sessions_sample.csv  1,000 sessions drawn from the training set
  demo_sessions.csv          12 hand-built sessions (8 human-like, 4 bot-like)
  demo_click_log.csv         synthetic Avazu-format raw click log with a few bot devices

Every random draw uses a fixed seed, so rerunning produces identical files.

Coupling to watch: demo_click_log.csv is built to contain exactly 6 bot-like devices, and
both the CI smoke test and the post-deploy check (.github/workflows/ci-cd.yml) assert that
a scan flags exactly 6. If you change the generator or retrain the model, rerun the scan
and update those assertions. The sample descriptions in frontend/src/pages/ScanPage.tsx
also quote this script's row and device counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "frontend" / "public" / "samples"
# Fixed seed: the sample files are committed, so regenerating them should not create a diff.
RNG = np.random.default_rng(7)


def training_sample() -> pd.DataFrame:
    """1,000 random sessions from the training table, given ids S00000.. so the UI has a label per
    row. All stored columns are kept; the API reads only the model's features.
    """
    df = pd.read_csv(ROOT / "artifacts" / "session_features.csv")
    sample = df.sample(1000, random_state=7).reset_index(drop=True)
    sample.insert(0, "session_id", [f"S{i:05d}" for i in sample.index])
    return sample


def demo_sessions() -> pd.DataFrame:
    """The 12 hand-written sessions in data/synthetic_sessions.csv, labelled demo-01 .. demo-12."""
    df = pd.read_csv(ROOT / "data" / "synthetic_sessions.csv")
    df.insert(0, "session_id", [f"demo-{i + 1:02d}" for i in range(len(df))])
    return df


def click_log() -> pd.DataFrame:
    """Synthetic impression-level log in Avazu's column layout: many ordinary devices plus a few
    scripted bots (see the two loops at the end).
    """
    sites = [f"{h:08x}" for h in RNG.integers(0, 2**32, 400)]
    apps = [f"{h:08x}" for h in RNG.integers(0, 2**32, 300)]
    rows = []

    def add(device: str, ip: str, n: int, ctr: float, site_pool, app_pool, c_vary: int, dtype: int, conn: int):
        """Append n impressions for one device. ctr is the click probability; site_pool and app_pool
        are the ids it draws from; c_vary is how many different values the anonymised C14-C21
        columns can take (1 = always the same); dtype and conn are device_type and
        device_conn_type.
        """
        base_c = RNG.integers(15000, 24000)
        for _ in range(n):
            rows.append(
                {
                    "click": int(RNG.random() < ctr),
                    # YYMMDDHH, like the real Avazu `hour` column (the features do not use it).
                    "hour": 14102100 + int(RNG.integers(0, 24)),
                    "C1": 1005,
                    "banner_pos": int(RNG.integers(0, 2)),
                    "site_id": RNG.choice(site_pool),
                    "app_id": RNG.choice(app_pool),
                    "device_id": device,
                    "device_ip": ip,
                    "device_model": "8a4875bd",
                    "device_type": dtype,
                    "device_conn_type": conn,
                    **{f"C{i}": int(base_c + RNG.integers(0, c_vary) * 7 + i) for i in range(14, 22)},
                }
            )

    # 400 ordinary devices: 1-3 impressions each, 15% click rate, only two sites and apps, constant
    # anonymised values, connection type 0.
    for _ in range(400):
        dev = f"{RNG.integers(0, 2**32):08x}"
        ip = f"{RNG.integers(0, 2**32):08x}"
        add(dev, ip, int(RNG.integers(1, 4)), 0.15, sites[:2], apps[:2], 1, 1, 0)
    # 6 scripted bots: hundreds of impressions each, a 55% click rate, spread over the whole site
    # and app pool, many different anonymised values, connection type 2. These are what the detector
    # is expected to flag.
    for _ in range(6):
        dev = f"{RNG.integers(0, 2**32):08x}"
        ip = f"{RNG.integers(0, 2**32):08x}"
        add(dev, ip, int(RNG.integers(250, 600)), 0.55, sites, apps, 40, 1, 2)

    # Shuffle so the bots are not all grouped at the end of the file.
    df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    df.insert(0, "id", [str(10**18 + i * 7919) for i in range(len(df))])
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("avazu_sessions_sample.csv", training_sample()),
        ("demo_sessions.csv", demo_sessions()),
        ("demo_click_log.csv", click_log()),
    ]:
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df):,} rows")
