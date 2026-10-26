import { useEffect, useRef, useState } from "react";
import { fmtInt } from "../format";
import type { HistBin, Verdict } from "../types";

interface Props {
  bins: HistBin[];
  reviewThreshold: number;
  height?: number;
}

const FILL: Record<Verdict, string> = {
  bot: "var(--bot)",
  review: "var(--review)",
  human: "var(--chart-human)",
};
const STACK: Verdict[] = ["human", "review", "bot"];
const M = { top: 22, right: 8, bottom: 40, left: 44 };

/** Session count by decision score, log-scaled so the small bot tail stays visible. */
export function ScoreHistogram({ bins, reviewThreshold, height = 240 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const totals = bins.map((b) => b.bot + b.review + b.human);
  const maxTotal = Math.max(1, ...totals);
  const decades = Math.max(1, Math.ceil(Math.log10(maxTotal + 1)));
  const innerW = width - M.left - M.right;
  const innerH = height - M.top - M.bottom;
  const x0 = bins[0]?.x0 ?? 0;
  const x1 = bins[bins.length - 1]?.x1 ?? 1;
  const sx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * innerW;
  const sy = (count: number) => M.top + innerH - (Math.log10(count + 1) / decades) * innerH;
  const bw = innerW / bins.length;
  const gap = bw > 6 ? 2 : 1;

  const ticks = Array.from({ length: decades + 1 }, (_, i) => 10 ** i);
  const xTicks = niceTicks(x0, x1, width < 520 ? 4 : 7);
  const markers = [
    { x: 0, label: "Bot cut-off" },
    { x: reviewThreshold, label: "Review" },
  ].filter((m) => m.x >= x0 && m.x <= x1);

  const hb = hover !== null ? bins[hover] : null;
  const tipLeft = hover !== null ? Math.min(Math.max(sx(bins[hover].x0) + bw / 2 - 90, 0), width - 190) : 0;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Histogram of decision scores by verdict">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={width - M.right} y1={sy(t - 1)} y2={sy(t - 1)} stroke="var(--grid)" />
            <text x={M.left - 8} y={sy(t - 1) + 4} textAnchor="end">
              {t === 1 ? "0" : t >= 1000 ? `${t / 1000}k` : t}
            </text>
          </g>
        ))}

        {bins.map((b, i) => {
          const total = totals[i];
          if (!total) return null;
          const top = sy(total);
          const barH = M.top + innerH - top;
          let y = M.top + innerH;
          const x = M.left + i * bw + gap / 2;
          const w = Math.max(1, bw - gap);
          return (
            <g key={i} opacity={hover === null || hover === i ? 1 : 0.55}>
              {STACK.map((v) => {
                if (!b[v]) return null;
                const h = Math.max(1, (b[v] / total) * barH);
                y -= h;
                return <rect key={v} x={x} y={y} width={w} height={h} fill={FILL[v]} rx={y <= top + 0.5 ? Math.min(2, w / 2) : 0} />;
              })}
            </g>
          );
        })}

        <line x1={M.left} x2={width - M.right} y1={M.top + innerH} y2={M.top + innerH} stroke="var(--border-strong)" />
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={M.top + innerH + 16} textAnchor="middle">
            {t.toFixed(2)}
          </text>
        ))}
        <text className="axis-title" x={M.left} y={height - 4}>
          ← more anomalous
        </text>
        <text className="axis-title" x={M.left + innerW / 2} y={height - 4} textAnchor="middle">
          Decision score
        </text>
        <text className="axis-title" x={width - M.right} y={height - 4} textAnchor="end">
          more normal →
        </text>

        {markers.map((m) => (
          <g key={m.label}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={M.top - 6} y2={M.top + innerH} stroke="var(--text-2)" strokeDasharray="3 3" />
            <text x={sx(m.x) + 4} y={M.top - 10} className="axis-title">
              {m.label}
            </text>
          </g>
        ))}

        {bins.map((_, i) => (
          <rect
            key={i}
            x={M.left + i * bw}
            y={M.top}
            width={bw}
            height={innerH}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>

      {hb && (
        <div className="tooltip" style={{ left: tipLeft, top: 0 }}>
          <div className="tooltip-title num">
            Score {hb.x0.toFixed(3)} to {hb.x1.toFixed(3)}
          </div>
          {(["bot", "review", "human"] as Verdict[]).map((v) => (
            <div className="tooltip-row" key={v}>
              <span>
                <i className="swatch" style={{ background: FILL[v] }} />
                {v === "bot" ? "Bot" : v === "review" ? "Review" : "Human"}
              </span>
              <span>{fmtInt(hb[v])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function HistogramLegend() {
  return (
    <div className="legend">
      <span>
        <i className="swatch" style={{ background: FILL.bot }} />
        Bot
      </span>
      <span>
        <i className="swatch" style={{ background: FILL.review }} />
        Review
      </span>
      <span>
        <i className="swatch" style={{ background: FILL.human }} />
        Human
      </span>
    </div>
  );
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 1e6) / 1e6);
  return out;
}
