import { createContext, useContext, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { getHealth, getModel } from "./api";
import { Layers, Pulse, Scan } from "./components/Icons";
import { OverviewPage } from "./pages/OverviewPage";
import { ScanPage } from "./pages/ScanPage";
import type { ModelSummary, ScanResult } from "./types";

interface AppState {
  model: ModelSummary | null;
  modelError: string | null;
  scan: ScanResult | null;
  setScan: (s: ScanResult | null) => void;
}

const Ctx = createContext<AppState | null>(null);
export const useApp = () => useContext(Ctx)!;

export function App() {
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    getModel().then(setModel, (e: Error) => setModelError(e.message));
    const ping = () => getHealth().then(() => setApiUp(true), () => setApiUp(false));
    ping();
    const t = setInterval(ping, 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <Ctx.Provider value={{ model, modelError, scan, setScan }}>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <span className="brand-mark">
              <Pulse size={14} strokeWidth={2} />
            </span>
            <span>Click Fraud Monitor</span>
          </div>
          <div className="nav-label">Detection</div>
          <nav className="nav">
            <NavLink to="/" end>
              <Layers />
              <span>Model baseline</span>
            </NavLink>
            <NavLink to="/scan">
              <Scan />
              <span>Scan traffic</span>
            </NavLink>
          </nav>
          <div className="sidebar-foot">
            <div className="status-line">
              <i className={`dot ${apiUp === null ? "" : apiUp ? "ok" : "down"}`} />
              {apiUp === null ? "Checking API…" : apiUp ? "API operational" : "API unreachable"}
            </div>
          </div>
        </aside>

        <main className="main">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/scan" element={<ScanPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Ctx.Provider>
  );
}
