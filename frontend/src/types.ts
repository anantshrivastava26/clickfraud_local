export type Verdict = "bot" | "review" | "human";

export interface Signal {
  feature: string;
  label: string;
  value: number;
  typical: number;
  z: number;
}

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

export interface HistBin {
  x0: number;
  x1: number;
  bot: number;
  review: number;
  human: number;
}

export interface FeatureProfile {
  feature: string;
  label: string;
  human_median: number;
  human_p95: number;
  bot_median: number;
  bot_p95: number;
}

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
