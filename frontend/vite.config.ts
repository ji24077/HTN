import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

// Preserve the browser Host header: backend same-origin checks must see the
// same origin as the Vite page. No admin credentials enter the frontend.
export default defineConfig(({ mode }) => {
  const root = fileURLToPath(new URL("../", import.meta.url));
  // These settings stay in Vite's Node process; secrets are never defined in the bundle.
  const env = loadEnv(mode, root, "");
  const origin = env.PUBLIC_ORIGIN ? new URL(env.PUBLIC_ORIGIN) : undefined;
  const proxy = {
    target:
      env.ORCHESTRATOR_BACKEND_URL ||
      `http://127.0.0.1:${env.PUBLIC_PORT || "8080"}`,
    changeOrigin: false,
  };
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: origin?.port ? Number(origin.port) : 5173,
      https:
        env.FRONTEND_TLS_CERT_FILE && env.FRONTEND_TLS_KEY_FILE
          ? {
              cert: readFileSync(resolve(root, env.FRONTEND_TLS_CERT_FILE)),
              key: readFileSync(resolve(root, env.FRONTEND_TLS_KEY_FILE)),
            }
          : undefined,
      strictPort: true,
      proxy: { "/v1": proxy, "/demo/session": proxy, "/auth": proxy },
    },
    build: {
      outDir: "../backend/src/orchestrator/server/web",
      emptyOutDir: true,
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      restoreMocks: true,
    },
  };
});
