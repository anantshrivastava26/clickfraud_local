import type { ResultRow, Verdict } from "./types";

const int = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmtInt = (n: number) => int.format(Math.round(n));
export const fmtCompact = (n: number) => compact.format(n);

export function fmtPct(part: number, whole: number, digits = 1): string {
  if (!whole) return "0%";
  return `${((part / whole) * 100).toFixed(digits)}%`;
}

export function fmtNum(n: number): string {
  if (Number.isInteger(n)) return int.format(n);
  if (Math.abs(n) < 1) return n.toFixed(3);
  return n.toFixed(2);
}

export function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

export const VERDICT_LABEL: Record<Verdict, string> = { bot: "Bot", review: "Review", human: "Human" };
export const VERDICT_COLOR: Record<Verdict, string> = {
  bot: "var(--bot)",
  review: "var(--review)",
  human: "var(--human)",
};

export function fmtZ(z: number): string {
  const sign = z > 0 ? "+" : "−";
  const a = Math.abs(z);
  return a >= 100 ? `${sign}100σ+` : `${sign}${a.toFixed(1)}σ`;
}

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

export function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
