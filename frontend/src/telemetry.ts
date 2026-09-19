import * as Sentry from "@sentry/react";

const SENSITIVE =
  /authorization|cookie|token|secret|passw|credential|api[-_]?key|private[-_]?key|^code$/i;
const CREDENTIAL =
  /Bearer\s+[\w.~+/=-]+|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]+|sb_(?:secret|publishable)_[\w-]+/g;

// Supabase puts access tokens and one-time codes in redirect URLs.
function scrubText(value: string): string {
  return value
    .replace(
      /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
      "[redacted]",
    )
    .replace(CREDENTIAL, "[redacted]")
    .replace(
      /([?#&](?:access_token|refresh_token|code|token)=)[^&#\s]+/gi,
      "$1[redacted]",
    );
}

export function scrub<T>(value: T, depth = 0): T {
  if (depth > 12) return "[redacted]" as T;
  if (typeof value === "string") return scrubText(value) as T;
  if (Array.isArray(value))
    return value.map((item) => scrub(item, depth + 1)) as T;
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        SENSITIVE.test(key) && item != null
          ? "[redacted]"
          : scrub(item, depth + 1),
      ]),
    ) as T;
  }
  return value;
}

// The DSN is served at runtime so one built bundle works in every deployment.
export async function initTelemetry(): Promise<void> {
  try {
    const response = await fetch("/telemetry/config", {
      credentials: "same-origin",
    });
    if (!response.ok) return;
    const { dsn, environment, release } = await response.json();
    if (!dsn) return;
    Sentry.init({
      dsn,
      environment,
      release: release || undefined,
      sendDefaultPii: true,
      enableLogs: true,
      integrations: [
        Sentry.browserTracingIntegration(),
        Sentry.replayIntegration({ maskAllInputs: true, blockAllMedia: true }),
        Sentry.consoleLoggingIntegration({ levels: ["warn", "error"] }),
      ],
      tracesSampleRate: 1.0,
      // Same-origin API calls carry trace headers, linking browser and server spans.
      tracePropagationTargets: [/^\//],
      replaysSessionSampleRate: 0,
      replaysOnErrorSampleRate: 1.0,
      beforeSend: (event) => scrub(event),
      beforeSendTransaction: (event) => scrub(event),
      beforeSendLog: (log) => scrub(log),
      beforeBreadcrumb: (breadcrumb) => scrub(breadcrumb),
    });
    Sentry.setTag("component", "dashboard");
  } catch {
    // Telemetry must never block the dashboard.
  }
}
