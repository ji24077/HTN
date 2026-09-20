import { describe, expect, it } from "vitest";
import type { AuditEvent, Worker } from "../api/types";
import { eventText, rememberWorkers, shortId, workerName } from "../lib/format";

const event = (overrides: Partial<AuditEvent>): AuditEvent => ({
  id: 1,
  at: "2026-09-19T00:00:00Z",
  entity: "enrollment",
  entity_id: "worker-1a2b",
  previous_state: "pending",
  new_state: "active",
  details: {},
  ...overrides,
});

describe("eventText", () => {
  it("describes enrollment outcomes instead of task states", () => {
    expect(eventText(event({}), [])).toBe("worker-1a2b enrolled");
    expect(eventText(event({ new_state: "failed" }), [])).toBe(
      "worker-1a2b enrollment failed",
    );
    expect(
      eventText(
        event({ new_state: "failed", details: { reason: "expired" } }),
        [],
      ),
    ).toBe("worker-1a2b enrollment expired");
    expect(
      eventText(
        event({ new_state: "failed", details: { reason: "cancelled" } }),
        [],
      ),
    ).toBe("worker-1a2b enrollment withdrawn");
  });

  it("still formats task and worker events by state", () => {
    expect(
      eventText(
        event({ entity: "task", entity_id: "job-1", new_state: "failed" }),
        [],
      ),
    ).toBe("“job-1” failed");
    expect(
      eventText(
        event({ entity: "worker", entity_id: "worker-a", new_state: "alive" }),
        [],
      ),
    ).toBe("Worker A connected");
  });
});

describe("workerName", () => {
  const device = (id: string, name?: string | null) =>
    ({ id, name, session_id: "s", state: "alive", paused: false }) as Worker;

  it("names a machine once a snapshot has introduced it", () => {
    const id = "691cf6cf-6cb3-4a21-86d6-994dc234b31a";
    // Before any snapshot, the id is all there is to go on.
    expect(workerName(id)).toBe(id);
    rememberWorkers([device(id, "Mac Air")]);
    expect(workerName(id)).toBe("Mac Air");
  });

  it("keeps naming a machine that has left the fleet", () => {
    // A finished task still names the laptop that ran it after that laptop goes away,
    // which is the only reason the column is worth reading.
    rememberWorkers([device("gone-1", "old-laptop")]);
    rememberWorkers([device("still-here", "desktop")]);
    expect(workerName("gone-1")).toBe("old-laptop");
  });

  it("ignores a blank name rather than showing an empty card", () => {
    rememberWorkers([device("blank-1", "   ")]);
    expect(workerName("blank-1")).toBe("blank-1");
    rememberWorkers([device("null-1", null)]);
    expect(workerName("null-1")).toBe("null-1");
  });

  it("still says 'Any worker' for an unassigned task", () => {
    expect(workerName(null)).toBe("Any worker");
  });
});

describe("shortId", () => {
  it("trims a uuid but leaves a short name alone", () => {
    expect(shortId("691cf6cf-6cb3-4a21-86d6-994dc234b31a")).toBe("691cf6cf");
    expect(shortId("worker-a")).toBe("worker-a");
  });
});
