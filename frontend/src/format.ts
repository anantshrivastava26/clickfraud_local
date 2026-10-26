import type { ResultRow, Verdict } from "./types";

// Number and date formatting shared by the pages. The Intl formatters are created once because
// constructing them is comparatively slow.
const int = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmtInt = (n: number) => int.format(Math.round(n));
export const fmtCompact = (n: number) => compact.format(n);

// part / whole as a percentage string. An empty whole gives '0%' so we never print NaN.
export function fmtPct(part: number, whole: number, digits = 1): string {
  if (!whole) return "0%";
  return `${((part / whole) * 100).toFixed(digits)}%`;
}

// Integers with thousands separators; fractions below 1 with 3 decimals so values like click rates
// stay readable; other numbers with 2.
export function fmtNum(n: number): string {
  if (Number.isInteger(n)) return int.format(n);
  if (Math.abs(n) < 1) return n.toFixed(3);
  return n.toFixed(2);
}

export function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

// Verdict text and colour in one place. The colours are CSS variables, so they follow the
// light/dark theme.
export const VERDICT_LABEL: Record<Verdict, string> = { bot: "Bot", review: "Review", human: "Human" };
export const VERDICT_COLOR: Record<Verdict, string> = {
  bot: "var(--bot)",
  review: "var(--review)",
  human: "var(--human)",
};

// z-score as '+3.2σ' or '−1.5σ' (a true minus sign). Extreme values are capped at '100σ+' to keep
// table cells narrow.
export function fmtZ(z: number): string {
  const sign = z > 0 ? "+" : "−";
  const a = Math.abs(z);
  return a >= 100 ? `${sign}100σ+` : `${sign}${a.toFixed(1)}σ`;
}

// Serialise scan results for the Export button: id, verdict, risk, decision_score, one column per
// model feature, then the top signals as text. Fields containing a comma, quote or newline are
// quoted, with quotes doubled (RFC 4180).
export function toCsv(rows: ResultRow[], features: string[]): string {
  const head = ["id", "verdict", "risk", "decision_score", ...features, "top_signals"];
  const esc = (v: string | number) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = rows.map((r) =>
    [
      r.id,
      r.verdict,
      r.risk,
      r.score,
      ...features.map((f) => r.features[f] ?? ""),
      r.signals.map((s) => `${s.label} ${fmtZ(s.z)}`).join("; "),
    ]
      .map(esc)
      .join(","),
  );
  return [head.join(","), ...lines].join("\n");
}

// Trigger a browser download of `text` without a server round trip: wrap it in a Blob, click a
// temporary link, then release the object URL.
export function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
