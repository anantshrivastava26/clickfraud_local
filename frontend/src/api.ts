import type { ModelSummary, ScanResult } from "./types";

// Empty means same origin: the Vite dev server and the nginx container both proxy /api.
// A trailing slash is stripped so `${BASE}${path}` never produces '//'.
const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

// Single fetch wrapper that turns every failure into an Error whose message can be shown to the
// user: either a network failure, or the `detail` string FastAPI puts in its error bodies (for
// example 'File is larger than 50 MB.' or the 422 explaining which columns are missing).
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

// Upload as multipart form data. Content-Type is deliberately not set: the browser adds it,
// including the multipart boundary.
export function scanFile(file: File): Promise<ScanResult> {
  const body = new FormData();
  body.append("file", file);
  return request<ScanResult>("/api/scan", { method: "POST", body });
}
