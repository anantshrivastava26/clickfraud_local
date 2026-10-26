import { fmtInt, fmtPct, VERDICT_LABEL } from "../format";
import type { Counts, Verdict } from "../types";
import { VerdictGlyph } from "./VerdictBadge";

const ORDER: Verdict[] = ["bot", "review", "human"];
const FILL: Record<Verdict, string> = { bot: "var(--bot)", review: "var(--review)", human: "var(--chart-human)" };

// One stacked bar showing the bot / review / human split, with counts and percentages in the
// legend. Segments are sized with flex-grow = count, so no percentage maths is needed, and verdicts
// with no rows are left out.
export function VerdictMeter({ counts, unit }: { counts: Counts; unit: string }) {
  const total = counts.sessions;
  return (
    <div>
      <div className="meter" role="img" aria-label={ORDER.map((v) => `${VERDICT_LABEL[v]} ${counts[v]}`).join(", ")}>
        {ORDER.map((v) =>
          counts[v] ? <div key={v} style={{ flexGrow: counts[v], flexBasis: 0, background: FILL[v] }} /> : null,
        )}
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        {ORDER.map((v) => (
          <span key={v}>
            <VerdictGlyph verdict={v} />
            {VERDICT_LABEL[v]}
            <b className="num" style={{ fontWeight: 500, color: "var(--text)" }}>
              {fmtInt(counts[v])}
            </b>
            <span className="muted num">{fmtPct(counts[v], total)}</span>
          </span>
        ))}
        <span className="muted" style={{ marginLeft: "auto" }}>
          {fmtInt(total)} {unit}
        </span>
      </div>
    </div>
  );
}
