// Shapes of the JSON returned by the backend (backend/app/detector.py). Keep them in sync with it:
// nothing validates the responses at run time.
export type Verdict = "bot" | "review" | "human";

// One reason a row looks unusual: a feature far from the training mean. `z` is in standard
// deviations (negative = unusually low); `typical` is the training median.
export interface Signal {
  feature: string;
  label: string;
  value: number;
  typical: number;
  z: number;
}

// One scored device or session. `risk` is 0-100; `score` is the raw IsolationForest decision score
// (lower = more anomalous, below 0 = bot). `features` holds every model input, for the detail
// drawer.
export interface ResultRow {
  id: string;
  verdict: Verdict;
  risk: number;
  score: number;
  impressions: number;
  clicks: number;
  click_rate: number;
  signals: Signal[];
  features: Record<string, number>;
}

// Headline totals. `sessions` counts scored rows, which are devices for a raw click log. The bot_*
// fields sum the traffic of rows flagged as bots.
export interface Counts {
  sessions: number;
  bot: number;
  review: number;
  human: number;
  impressions: number;
  clicks: number;
  bot_impressions: number;
  bot_clicks: number;
}

// One histogram bucket covering decision scores x0 to x1, split by verdict.
export interface HistBin {
  x0: number;
  x1: number;
  bot: number;
  review: number;
  human: number;
}

// Training-set median and 95th percentile of one feature, for non-bot ('human') and bot sessions.
export interface FeatureProfile {
  feature: string;
  label: string;
  human_median: number;
  human_p95: number;
  bot_median: number;
  bot_p95: number;
}

// Response of GET /api/model: the model's settings plus the training-set baseline.
export interface ModelSummary {
  model: {
    algorithm: string;
    n_estimators: number;
    contamination: number;
    max_samples: number;
    decision_threshold: number;
    review_threshold: number;
    risk_bands: { bot: number; review: number };
    features: { name: string; label: string }[];
    trained_at: string | null;
    sklearn_version: string | null;
    source: string | null;
  };
  counts: Counts;
  histogram: HistBin[];
  profile: FeatureProfile[];
  top_anomalies: ResultRow[];
}

// Response of POST /api/scan. `mode` says how the file was read; `truncated` is true when `results`
// was capped (counts and histogram still cover the whole file).
export interface ScanResult {
  file: string;
  mode: "click_log" | "sessions";
  input_rows: number;
  warnings: string[];
  counts: Counts;
  histogram: HistBin[];
  results: ResultRow[];
  truncated: boolean;
}
