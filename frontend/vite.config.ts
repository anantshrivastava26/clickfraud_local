import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // In development the browser talks only to this server and /api is forwarded to the backend, so
    // no CORS setup is needed. Set API_PROXY_TARGET to point at a different backend.
    proxy: { "/api": process.env.API_PROXY_TARGET ?? "http://localhost:8000" },
  },
});
