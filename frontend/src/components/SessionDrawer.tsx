import { useEffect } from "react";
import { fmtNum, fmtZ } from "../format";
import type { ModelSummary, ResultRow } from "../types";
import { Close } from "./Icons";
import { VerdictBadge } from "./VerdictBadge";

interface Props {
  row: ResultRow;
  model: ModelSummary;
  idLabel: string;
  onClose: () => void;
}

// Slide-in detail panel for one row: its risk on the 0-100 scale, the signals behind the score, and
// every feature value beside the human and bot training medians. Closes on Escape, on a click on
// the backdrop, or with the close button.
export function SessionDrawer({ row, model, idLabel, onClose }: Props) {
  // Escape closes the drawer. The listener is removed again when the drawer unmounts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // The risk bar is drawn with the same bands the backend uses, so its colours always line up with
  // the verdicts: 0 to review = human, review to bot = review, bot to 100 = bot.
  const { bot, review } = model.model.risk_bands;
  // Training medians by feature name, for the comparison columns. Features that triggered a signal
  // are shown in bold in the table.
  const typical = Object.fromEntries(model.profile.map((p) => [p.feature, p]));
  const signalFeatures = new Set(row.signals.map((s) => s.feature));

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label={`${idLabel} ${row.id}`}>
        <div className="drawer-head">
          <div style={{ minWidth: 0 }}>
            <div className="muted" style={{ fontSize: 12 }}>
              {idLabel}
            </div>
            <h2>{row.id}</h2>
            <VerdictBadge verdict={row.verdict} />
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <Close />
          </button>
        </div>

        <div className="drawer-body">
          <section>
            <h3>Risk score</h3>
            <div className="risk-big">
              <strong>{Math.round(row.risk)}</strong>
              <span className="muted">/ 100</span>
              <span className="muted num" style={{ marginLeft: "auto", fontSize: 12 }}>
                decision score {row.score.toFixed(4)}
              </span>
            </div>
            <div className="scale">
              <div style={{ flexGrow: review, background: "var(--chart-human)" }} />
              <div style={{ flexGrow: bot - review, background: "var(--review)" }} />
              <div style={{ flexGrow: 100 - bot, background: "var(--bot)" }} />
              <span className="scale-pin" style={{ left: `calc(${row.risk}% - 1px)` }} />
            </div>
            <div className="scale-labels">
              <span>0 Human</span>
              <span>{review} Review</span>
              <span>{bot} Bot</span>
              <span>100</span>
            </div>
          </section>

          <section>
            <h3>Why it was scored this way</h3>
            {row.signals.length ? (
              row.signals.map((s) => (
                <div className="signal" key={s.feature}>
                  <span className="signal-name">{s.label}</span>
                  <span className="signal-z">{fmtZ(s.z)}</span>
                  <span className="signal-detail">
                    {fmtNum(s.value)} here vs. a training median of {fmtNum(s.typical)}
                  </span>
                </div>
              ))
            ) : (
              <p className="secondary" style={{ margin: 0 }}>
                No feature sits more than 2.5σ from the training mean. The score comes from the combination of
                values rather than one outlier.
              </p>
            )}
          </section>

          <section>
            <h3>Feature values</h3>
            <table className="data">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th className="r">Value</th>
                  <th className="r">Human median</th>
                  <th className="r">Bot median</th>
                </tr>
              </thead>
              <tbody>
                {model.model.features.map((f) => (
                  <tr key={f.name}>
                    <td style={{ fontWeight: signalFeatures.has(f.name) ? 600 : 400 }}>{f.label}</td>
                    <td className="r">{fmtNum(row.features[f.name] ?? 0)}</td>
                    <td className="r muted">{fmtNum(typical[f.name]?.human_median ?? 0)}</td>
                    <td className="r muted">{fmtNum(typical[f.name]?.bot_median ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </aside>
    </>
  );
}
