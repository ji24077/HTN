import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import * as Sentry from "@sentry/react";
import App from "./App";
import { GpuLab } from "./components/GpuLab";
import { initTelemetry } from "./telemetry";
import "./styles.css";

const localGpuDemo =
  import.meta.env.DEV && window.location.pathname === "/gpu-lab";
if (!localGpuDemo) void initTelemetry();

createRoot(document.getElementById("root")!, {
  onUncaughtError: Sentry.reactErrorHandler(),
  onCaughtError: Sentry.reactErrorHandler(),
  onRecoverableError: Sentry.reactErrorHandler(),
}).render(
  <StrictMode>
    {localGpuDemo ? (
      <main className="gpu-demo">
        <GpuLab active />
      </main>
    ) : (
      <App />
    )}
  </StrictMode>,
);
