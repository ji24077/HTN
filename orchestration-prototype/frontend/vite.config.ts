import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Preserve the browser Host header: backend same-origin checks must see the
// same origin as the Vite page. No admin credentials enter the frontend.
const proxy = {
  target: process.env.ORCHESTRATOR_BACKEND_URL || "http://127.0.0.1:8787",
  changeOrigin: false,
};
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/v1": proxy, "/demo/session": proxy },
  },
  build: { outDir: "../src/orchestrator/server/web", emptyOutDir: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
