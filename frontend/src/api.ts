import type { ModelSummary, ScanResult } from "./types";

// Empty means same origin: the Vite dev server and the nginx container both proxy /api.
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error("Can't reach the detection API. Check that the backend is running.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => request<{ status: string; model_loaded: boolean }>("/api/health");

export const getModel = () => request<ModelSummary>("/api/model");

export function scanFile(file: File): Promise<ScanResult> {
  const body = new FormData();
  body.append("file", file);
  return request<ScanResult>("/api/scan", { method: "POST", body });
}
