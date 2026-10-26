import { useMemo, useState } from "react";
import { fmtInt, fmtNum, fmtZ } from "../format";
import type { ResultRow, Verdict } from "../types";
import { Search, Sort } from "./Icons";
import { RiskCell, VerdictBadge } from "./VerdictBadge";

type SortKey = "risk" | "impressions" | "clicks" | "click_rate";
type Filter = "all" | Verdict;

interface Props {
  rows: ResultRow[];
  idLabel: string;
  selectedId?: string | null;
  onSelect: (row: ResultRow) => void;
  pageSize?: number;
  controls?: boolean;
}

export function ResultsTable({ rows, idLabel, selectedId, onSelect, pageSize = 50, controls = true }: Props) {
  const [filter, setFilter] = useState<Filter>(rows.some((r) => r.verdict === "bot") ? "bot" : "all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "risk", dir: "desc" });
  const [page, setPage] = useState(0);

  const counts = useMemo(() => {
    const c = { all: rows.length, bot: 0, review: 0, human: 0 };
    rows.forEach((r) => c[r.verdict]++);
    return c;
  }, [rows]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const out = rows.filter((r) => (filter === "all" || r.verdict === filter) && (!q || r.id.toLowerCase().includes(q)));
    const m = sort.dir === "asc" ? 1 : -1;
    // Risk ties are common (risk is capped at 100); fall back to the raw decision score.
    out.sort((a, b) => (a[sort.key] - b[sort.key]) * m || (b.score - a.score) * m);
    return out;
  }, [rows, filter, query, sort]);

  const pages = Math.max(1, Math.ceil(visible.length / pageSize));
  const current = Math.min(page, pages - 1);
  const slice = visible.slice(current * pageSize, (current + 1) * pageSize);

  const toggleSort = (key: SortKey) => {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "desc" ? "asc" : "desc" } : { key, dir: "desc" }));
    setPage(0);
  };
  const th = (key: SortKey, label: string) => (
    <th className="r" aria-sort={sort.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button onClick={() => toggleSort(key)}>
        {label}
        <Sort dir={sort.key === key ? sort.dir : null} />
      </button>
    </th>
  );

  return (
    <>
      {controls && (
        <div className="card-head" style={{ paddingBottom: 12 }}>
          <div className="segmented" role="tablist">
            {(["all", "bot", "review", "human"] as Filter[]).map((f) => (
              <button
                key={f}
                role="tab"
                aria-selected={filter === f}
                className={filter === f ? "on" : ""}
                onClick={() => {
                  setFilter(f);
                  setPage(0);
                }}
              >
                {f === "all" ? "All" : f === "bot" ? "Bot" : f === "review" ? "Review" : "Human"}
                <span className="count">{fmtInt(counts[f])}</span>
              </button>
            ))}
          </div>
          <div className="input-wrap">
            <Search />
            <input
              className="input"
              placeholder={`Search ${idLabel.toLowerCase()}`}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(0);
              }}
            />
          </div>
        </div>
      )}

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>{idLabel}</th>
              <th>Verdict</th>
              {th("risk", "Risk")}
              {th("impressions", "Impressions")}
              {th("clicks", "Clicks")}
              {th("click_rate", "CTR")}
              <th>Strongest signal</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => (
              <tr
                key={r.id + r.score}
                className={`clickable ${selectedId === r.id ? "selected" : ""}`}
                onClick={() => onSelect(r)}
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && onSelect(r)}
              >
                <td className="mono">{r.id}</td>
                <td>
                  <VerdictBadge verdict={r.verdict} />
                </td>
                <td className="r">
                  <RiskCell risk={r.risk} verdict={r.verdict} />
                </td>
                <td className="r">{fmtInt(r.impressions)}</td>
                <td className="r">{fmtInt(r.clicks)}</td>
                <td className="r">{(r.click_rate * 100).toFixed(1)}%</td>
                <td className="secondary">
                  {r.signals[0] ? (
                    <>
                      {r.signals[0].label} <span className="muted">{fmtNum(r.signals[0].value)}</span>{" "}
                      <span style={{ color: "var(--bot-text)" }}>{fmtZ(r.signals[0].z)}</span>
                    </>
                  ) : (
                    <span className="muted">Within normal range</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!slice.length && <div className="empty">No sessions match this filter.</div>}
      </div>

      {pages > 1 && (
        <div className="table-foot">
          <span className="num">
            {fmtInt(current * pageSize + 1)}–{fmtInt(Math.min(visible.length, (current + 1) * pageSize))} of{" "}
            {fmtInt(visible.length)}
          </span>
          <div className="btn-row">
            <button className="btn" disabled={current === 0} onClick={() => setPage(current - 1)}>
              Previous
            </button>
            <button className="btn" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>
              Next
            </button>
          </div>
        </div>
      )}
    </>
  );
}
