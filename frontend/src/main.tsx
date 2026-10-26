// Fonts are bundled with the app (no request to a font CDN) and imported before the stylesheet that
// refers to them.
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import "./styles.css";

// BrowserRouter uses real URLs such as /scan, so the web server must fall back to index.html for
// unknown paths: Vite's dev server does this by default and nginx does it with try_files.
// StrictMode re-runs effects once in development only, to expose side-effect bugs.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
