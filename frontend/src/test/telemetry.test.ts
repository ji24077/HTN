import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as Sentry from "@sentry/react";
import type { Envelope } from "@sentry/core";
import { initTelemetry } from "../telemetry";

const { envelopes } = vi.hoisted(() => ({ envelopes: [] as Envelope[] }));

// Exercise real SDK initialization/serialization; replace only the network transport
// and replay compression/timing so the captured recording can be inspected locally.
vi.mock("@sentry/react", async (importOriginal) => {
  const sdk = await importOriginal<typeof Sentry>();
  return {
    ...sdk,
    init: (options: Parameters<typeof sdk.init>[0]) =>
      sdk.init({
        ...options,
        transport: () => ({
          send: async (envelope: Envelope) => {
            envelopes.push(envelope);
            return { statusCode: 200 };
          },
          flush: async () => true,
        }),
      }),
    replayIntegration: (options: Parameters<typeof sdk.replayIntegration>[0]) =>
      sdk.replayIntegration({
        ...options,
        useCompression: false,
        minReplayDuration: 0,
      }),
  };
});

beforeEach(() => {
  envelopes.length = 0;
  window.history.replaceState(null, "", "/");
  sessionStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        dsn: "https://public@example.test/1",
        environment: "test",
      }),
    }),
  );
});

afterEach(async () => {
  await Sentry.getReplay()?.stop({ flush: false });
  await Sentry.close();
  vi.unstubAllGlobals();
});

it.each([
  "/?code=synthetic-one-time-secret",
  "/#access_token=synthetic-access-secret&refresh_token=synthetic-refresh-secret",
  "/?token_hash=synthetic-token-hash",
])("does not record an authentication callback: %s", async (url) => {
  window.history.replaceState(null, "", url);
  await initTelemetry();
  expect(Sentry.getReplay()).toBeUndefined();
  Sentry.captureException(new Error("callback failure"));
  await Sentry.flush();
  expect(
    envelopes.some((envelope) =>
      envelope[1].some(([header]) => header.type === "event"),
    ),
  ).toBe(true);
  expect(JSON.stringify(envelopes)).not.toContain("synthetic-");
});

it("does not enable replay when auth consumes the URL during config fetch", async () => {
  window.history.replaceState(null, "", "/?code=synthetic-racing-secret");
  vi.mocked(fetch).mockImplementation(async () => {
    window.history.replaceState(null, "", "/");
    return {
      ok: true,
      json: async () => ({ dsn: "https://public@example.test/1" }),
    } as Response;
  });
  await initTelemetry();
  expect(Sentry.getReplay()).toBeUndefined();
});

it("still sends a replay recording on an ordinary dashboard page", async () => {
  // The SDK excludes plain Node; expose jsdom as a browser-capable renderer.
  vi.stubGlobal("process", { ...process, type: "renderer" });
  await initTelemetry();
  const replay = Sentry.getReplay();
  expect(replay).toBeDefined();
  Sentry.captureException(new Error("dashboard failure"));
  await Sentry.flush();
  await replay!.flush({ continueRecording: false });
  const types = envelopes.flatMap((envelope) =>
    envelope[1].map(([header]) => header.type),
  );
  expect(types).toContain("event");
  expect(types).toContain("replay_event");
  expect(types).toContain("replay_recording");
});
