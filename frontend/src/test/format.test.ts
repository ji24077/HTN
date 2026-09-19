import { describe, expect, it } from "vitest";
import type { AuditEvent } from "../api/types";
import { eventText } from "../lib/format";

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
