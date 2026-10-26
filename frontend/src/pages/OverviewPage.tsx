import { useState } from "react";
import { Link } from "react-router-dom";
import { useApp } from "../App";
import { Scan } from "../components/Icons";
import { ResultsTable } from "../components/ResultsTable";
import { HistogramLegend, ScoreHistogram } from "../components/ScoreHistogram";
import { SessionDrawer } from "../components/SessionDrawer";
import { VerdictMeter } from "../components/VerdictMeter";
import { fmtCompact, fmtDate, fmtInt, fmtNum, fmtPct } from "../format";
import type { ResultRow } from "../types";

// 'Model baseline' page: what the model learned from the training data, read from GET /api/model
// through the shared app state. Uploaded files are scored against this baseline on the Scan page.
export function OverviewPage() {
  const { model, modelError } = useApp();
  const [selected, setSelected] = useState<ResultRow | null>(null);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="crumbs">Detection / Model baseline</div>
          <h1>Model baseline</h1>
          <p>
            How the IsolationForest scored the device sessions it was trained on. Uploaded traffic is scored against
            this baseline.
          </p>
        </div>
        <Link to="/scan" className="btn primary" style={{ textDecoration: "none" }}>
          <Scan size={14} />
          Scan a file
        </Link>
      </div>

      {modelError && <div className="callout error">{modelError}</div>}
      {!model && !modelError && <Loading />}

      {model && (
        <div className="stack">
          <div className="kpis">
            <Kpi label="Sessions profiled" value={fmtInt(model.counts.sessions)} foot={`${fmtCompact(model.counts.impressions)} impressions`} />
            <Kpi
              label="Flagged as bot"
              value={fmtInt(model.counts.bot)}
              foot={`${fmtPct(model.counts.bot, model.counts.sessions, 2)} of sessions`}
            />
            <Kpi
              label="Sent to review"
              value={fmtInt(model.counts.review)}
              foot={`${fmtPct(model.counts.review, model.counts.sessions, 2)} of sessions`}
            />
            <Kpi
              label="Clicks from flagged sessions"
              value={fmtPct(model.counts.bot_clicks, model.counts.clicks)}
              foot={`${fmtInt(model.counts.bot_clicks)} of ${fmtInt(model.counts.clicks)}`}
            />
            <Kpi label="Model features" value={String(model.model.features.length)} foot={`${model.model.n_estimators} trees`} />
          </div>

          <div className="card">
            <div className="card-head">
              <div>
                <h2>Decision score distribution</h2>
                <div className="sub">Sessions per score bucket, log scale. Scores below 0 are classified as bot.</div>
              </div>
              <HistogramLegend />
            </div>
            <div className="card-body">
              <ScoreHistogram bins={model.histogram} reviewThreshold={model.model.review_threshold} />
              <div style={{ marginTop: 14 }}>
                <VerdictMeter counts={model.counts} unit="sessions" />
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <div className="card-head">
                <div>
                  <h2>Feature profile</h2>
                  <div className="sub">Medians and 95th percentiles for each verdict group in the training data.</div>
                </div>
              </div>
              <div className="card-body" style={{ padding: "12px 0 4px" }}>
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th className="r">Human p50</th>
                        <th className="r">Human p95</th>
                        <th className="r">Bot p50</th>
                        <th className="r">Bot p95</th>
                      </tr>
                    </thead>
                    <tbody>
                      {model.profile.map((p) => (
                        <tr key={p.feature}>
                          <td>{p.label}</td>
                          <td className="r">{fmtNum(p.human_median)}</td>
                          <td className="r">{fmtNum(p.human_p95)}</td>
                          <td className="r" style={{ fontWeight: p.bot_median > p.human_p95 ? 600 : 400 }}>
                            {fmtNum(p.bot_median)}
                          </td>
                          <td className="r">{fmtNum(p.bot_p95)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="card" style={{ alignSelf: "start" }}>
              <div className="card-head">
                <h2>Model</h2>
                <span className="tag">{model.model.algorithm}</span>
              </div>
              <div className="card-body">
                <dl className="dl">
                  <dt>Estimators</dt>
                  <dd>{model.model.n_estimators}</dd>
                  <dt>Samples per tree</dt>
                  <dd>{model.model.max_samples}</dd>
                  <dt>Contamination</dt>
                  <dd>{(model.model.contamination * 100).toFixed(1)}%</dd>
                  <dt>Bot cut-off</dt>
                  <dd>score &lt; {model.model.decision_threshold.toFixed(2)}</dd>
                  <dt>Review cut-off</dt>
                  <dd>score &lt; {model.model.review_threshold.toFixed(3)}</dd>
                  <dt>Risk bands</dt>
                  <dd>
                    Bot ≥ {model.model.risk_bands.bot} · Review ≥ {model.model.risk_bands.review}
                  </dd>
                  <dt>Training source</dt>
                  <dd className="mono">{model.model.source ?? "—"}</dd>
                  <dt>Trained</dt>
                  <dd>{fmtDate(model.model.trained_at)}</dd>
                  <dt>scikit-learn</dt>
                  <dd>{model.model.sklearn_version ?? "—"}</dd>
                </dl>
                <div style={{ borderTop: "1px solid var(--border)", marginTop: 14, paddingTop: 12 }}>
                  <div className="secondary" style={{ marginBottom: 8, fontSize: 12 }}>
                    Input features
                  </div>
                  <div className="cols">
                    {model.model.features.map((f) => (
                      <code key={f.name}>{f.name}</code>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head" style={{ paddingBottom: 12 }}>
              <div>
                <h2>Most anomalous training sessions</h2>
                <div className="sub">The 25 lowest decision scores. Select a row to see what drove the score.</div>
              </div>
            </div>
            <ResultsTable
              rows={model.top_anomalies}
              idLabel="Session"
              controls={false}
              selectedId={selected?.id}
              onSelect={setSelected}
            />
          </div>
        </div>
      )}

      {selected && model && (
        <SessionDrawer row={selected} model={model} idLabel="Session" onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

// Summary tile: label, large value, footnote. Exported because the Scan page reuses it.
export function Kpi({ label, value, foot, icon }: { label: string; value: string; foot?: string; icon?: React.ReactNode }) {
  return (
    <div className="kpi">
      <div className="kpi-label">
        {icon}
        {label}
      </div>
      <div className="kpi-value">{value}</div>
      {foot && <div className="kpi-foot">{foot}</div>}
    </div>
  );
}

// Placeholder blocks shown while the model summary is loading.
function Loading() {
  return (
    <div className="stack">
      <div className="skeleton" style={{ height: 86 }} />
      <div className="skeleton" style={{ height: 320 }} />
      <div className="skeleton" style={{ height: 240 }} />
    </div>
  );
}
