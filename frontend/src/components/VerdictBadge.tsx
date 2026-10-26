import { VERDICT_COLOR, VERDICT_LABEL } from "../format";
import type { Verdict } from "../types";
import { BotGlyph, HumanGlyph, ReviewGlyph } from "./Icons";

// A verdict is always shown as icon + colour + text, never colour alone, so it stays readable for
// colour-blind users and in grayscale.
const GLYPH = { bot: BotGlyph, review: ReviewGlyph, human: HumanGlyph };

export function VerdictGlyph({ verdict, size }: { verdict: Verdict; size?: number }) {
  const G = GLYPH[verdict];
  return <G size={size} style={{ color: VERDICT_COLOR[verdict] }} />;
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span className={`badge ${verdict}`}>
      <VerdictGlyph verdict={verdict} />
      {VERDICT_LABEL[verdict]}
    </span>
  );
}

// Risk as a small bar plus the rounded number. The bar's width is the 0-100 risk and its colour is
// the verdict's.
export function RiskCell({ risk, verdict }: { risk: number; verdict: Verdict }) {
  return (
    <span className="risk">
      <span className="risk-track">
        <span className="risk-fill" style={{ display: "block", width: `${risk}%`, background: VERDICT_COLOR[verdict] }} />
      </span>
      <span className="num" style={{ minWidth: 26, textAlign: "right" }}>
        {Math.round(risk)}
      </span>
    </span>
  );
}
