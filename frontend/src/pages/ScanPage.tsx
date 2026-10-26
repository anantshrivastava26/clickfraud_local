import { useRef, useState } from "react";
import { scanFile } from "../api";
import { useApp } from "../App";
import { Download, FileIcon, Info, Upload } from "../components/Icons";
import { ResultsTable } from "../components/ResultsTable";
import { HistogramLegend, ScoreHistogram } from "../components/ScoreHistogram";
import { SessionDrawer } from "../components/SessionDrawer";
import { VerdictGlyph } from "../components/VerdictBadge";
import { VerdictMeter } from "../components/VerdictMeter";
import { download, fmtInt, fmtPct, toCsv } from "../format";
import type { ResultRow, ScanResult } from "../types";
import { Kpi } from "./OverviewPage";

const RAW_COLUMNS = ["click", "device_id", "site_id", "app_id", "device_type", "device_conn_type", "C14 … C21"];
const SAMPLES = [
  { file: "demo_click_log.csv", name: "Raw click log", desc: "3,369 impressions from 406 devices, 6 of them scripted bots" },
  { file: "avazu_sessions_sample.csv", name: "Avazu sessions", desc: "1,000 session rows drawn from the training set" },
  { file: "demo_sessions.csv", name: "Hand-built sessions", desc: "12 sessions: 8 human-like, 4 bot-like" },
];

export function ScanPage() {
  const { model, scan, setScan } = useApp();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ResultRow | null>(null);

  async function run(file: File) {
    setBusy(file.name);
    setError(null);
    try {
      setScan(await scanFile(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runSample(name: string) {
    const res = await fetch(`/samples/${name}`);
    const blob = await res.blob();
    await run(new File([blob], name, { type: "text/csv" }));
  }

  const idLabel = scan?.mode === "click_log" ? "Device" : "Session";

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="crumbs">Detection / Scan traffic</div>
          <h1>Scan traffic</h1>
          <p>Upload a click log or session export. Every device or session is scored against the trained baseline.</p>
        </div>
        {scan && (
          <div className="btn-row">
            <button
              className="btn"
              onClick={() =>
                download(
                  scan.file.replace(/\.csv$/i, "") + "_scored.csv",
                  toCsv(scan.results, model?.model.features.map((f) => f.name) ?? []),
                )
              }
            >
              <Download size={14} />
              Export results
            </button>
            <button className="btn primary" onClick={() => setScan(null)}>
              <Upload size={14} />
              New scan
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="callout error" style={{ marginBottom: 16 }}>
          <Info style={{ flex: "none", marginTop: 2, color: "var(--bot)" }} />
          <div>
            <b>Scan failed.</b> {error}
          </div>
        </div>
      )}

      {!scan ? (
        <UploadPanel busy={busy} onFile={run} onSample={runSample} />
      ) : (
        <Results scan={scan} idLabel={idLabel} selected={selected} onSelect={setSelected} />
      )}

      {selected && model && (
        <SessionDrawer row={selected} model={model} idLabel={idLabel} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

function UploadPanel({
  busy,
  onFile,
  onSample,
}: {
  busy: string | null;
  onFile: (f: File) => void;
  onSample: (name: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const { model } = useApp();

  return (
    <div className="stack">
      <div className="card">
        <div className="card-body">
          <div
            className={`drop ${over ? "over" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => !busy && input.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              const f = e.dataTransfer.files[0];
              if (f && !busy) onFile(f);
            }}
          >
            <span className="drop-icon">
              <Upload />
            </span>
            {busy ? (
              <>
                <h3>Scoring {busy}…</h3>
                <p>Building features and running the model.</p>
                <div className="progress" style={{ maxWidth: 280, margin: "16px auto 0" }}>
                  <div />
                </div>
              </>
            ) : (
              <>
                <h3>Drop a CSV here, or click to browse</h3>
                <p>Up to 50 MB. The format is detected from the header row.</p>
              </>
            )}
            <input
              ref={input}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onFile(f);
                e.target.value = "";
              }}
            />
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head">
            <h2>Accepted formats</h2>
          </div>
          <div className="card-body formats">
            <div className="format">
              <h3>
                Raw click log <span className="tag">one row per impression</span>
              </h3>
              <p>Avazu-style export. Rows are grouped by device and turned into the model's behavioural features.</p>
              <div className="cols">
                {RAW_COLUMNS.map((c) => (
                  <code key={c}>{c}</code>
                ))}
              </div>
            </div>
            <div className="format">
              <h3>
                Session features <span className="tag">one row per session</span>
              </h3>
              <p>Already aggregated. Optional session_id, user_id or device_id column is used as the row label.</p>
              <div className="cols">
                {(model?.model.features ?? []).map((f) => (
                  <code key={f.name}>{f.name}</code>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="card" style={{ alignSelf: "start" }}>
          <div className="card-head">
            <h2>Sample files</h2>
          </div>
          <div className="card-body">
            <div className="samples">
              {SAMPLES.map((s) => (
                <div className="sample" key={s.file}>
                  <div style={{ minWidth: 0 }}>
                    <div className="sample-name">{s.name}</div>
                    <div className="sample-desc">{s.desc}</div>
                  </div>
                  <div className="btn-row" style={{ flexWrap: "nowrap" }}>
                    <a className="btn" href={`/samples/${s.file}`} download aria-label={`Download ${s.name}`}>
                      <Download size={14} />
                    </a>
                    <button className="btn" disabled={!!busy} onClick={() => onSample(s.file)}>
                      Scan
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Results({
  scan,
  idLabel,
  selected,
  onSelect,
}: {
  scan: ScanResult;
  idLabel: string;
  selected: ResultRow | null;
  onSelect: (r: ResultRow) => void;
}) {
  const { model } = useApp();
  const c = scan.counts;
  const unit = scan.mode === "click_log" ? "devices" : "sessions";

  return (
    <div className="stack">
      <div className="card file-bar">
        <div className="file-meta">
          <span className="file-icon">
            <FileIcon />
          </span>
          <div style={{ minWidth: 0 }}>
            <div className="file-name">{scan.file}</div>
            <div className="muted num" style={{ fontSize: 12 }}>
              {fmtInt(scan.input_rows)} rows · {scan.mode === "click_log" ? `grouped into ${fmtInt(c.sessions)} devices` : `${fmtInt(c.sessions)} sessions`}
            </div>
          </div>
          <span className="tag">{scan.mode === "click_log" ? "Raw click log" : "Session features"}</span>
        </div>
        <ScanVerdict counts={c} unit={unit} />
      </div>

      {scan.warnings.length > 0 && (
        <div className="callout warn">
          <Info style={{ flex: "none", marginTop: 2 }} />
          <ul>
            {scan.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="kpis">
        <Kpi label={`${unit[0].toUpperCase()}${unit.slice(1)} scored`} value={fmtInt(c.sessions)} foot={`${fmtInt(c.impressions)} impressions`} />
        <Kpi
          label="Bot"
          icon={<VerdictGlyph verdict="bot" />}
          value={fmtInt(c.bot)}
          foot={`${fmtPct(c.bot, c.sessions)} of ${unit}`}
        />
        <Kpi
          label="Needs review"
          icon={<VerdictGlyph verdict="review" />}
          value={fmtInt(c.review)}
          foot={`${fmtPct(c.review, c.sessions)} of ${unit}`}
        />
        <Kpi label="Invalid click rate" value={fmtPct(c.bot_clicks, c.clicks)} foot={`${fmtInt(c.bot_clicks)} of ${fmtInt(c.clicks)} clicks`} />
        <Kpi
          label="Invalid impressions"
          value={fmtPct(c.bot_impressions, c.impressions)}
          foot={`${fmtInt(c.bot_impressions)} of ${fmtInt(c.impressions)}`}
        />
      </div>

      <div className="card">
        <div className="card-head">
          <div>
            <h2>Score distribution</h2>
            <div className="sub">
              {unit[0].toUpperCase() + unit.slice(1)} per score bucket on the training scale, log scale.
            </div>
          </div>
          <HistogramLegend />
        </div>
        <div className="card-body">
          <ScoreHistogram bins={scan.histogram} reviewThreshold={model?.model.review_threshold ?? 0} height={200} />
          <div style={{ marginTop: 14 }}>
            <VerdictMeter counts={c} unit={unit} />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head" style={{ paddingBottom: 0 }}>
          <div>
            <h2>Scored {unit}</h2>
            <div className="sub">
              {scan.truncated
                ? `Showing the ${fmtInt(scan.results.length)} highest-risk ${unit}. Counts above cover the whole file.`
                : "Select a row to see the signals behind its score."}
            </div>
          </div>
        </div>
        <ResultsTable key={scan.file + scan.input_rows} rows={scan.results} idLabel={idLabel} selectedId={selected?.id} onSelect={onSelect} />
      </div>
    </div>
  );
}

function ScanVerdict({ counts, unit }: { counts: ScanResult["counts"]; unit: string }) {
  const verdict = counts.bot ? "bot" : counts.review ? "review" : "human";
  const text = counts.bot
    ? `Bot traffic detected in ${fmtInt(counts.bot)} ${unit}`
    : counts.review
      ? `No bots. ${fmtInt(counts.review)} ${unit} need review`
      : "No fraudulent traffic detected";
  return (
    <span className={`badge ${verdict}`} style={{ height: 26, padding: "0 10px", fontSize: 12.5 }}>
      <VerdictGlyph verdict={verdict} />
      {text}
    </span>
  );
}
