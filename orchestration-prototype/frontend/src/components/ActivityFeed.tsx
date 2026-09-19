import type { AuditEvent, ConnectionStatus, Task } from "../api/types";
import { eventText, time } from "../lib/format";
export function ActivityFeed({
  events,
  tasks,
  status,
}: {
  events: AuditEvent[];
  tasks: Task[];
  status: ConnectionStatus;
}) {
  return (
    <section className="activity">
      <div className="section-heading">
        <h2>Behind the scenes</h2>
        <span className="pill" id="live-pill">
          <i className="pulse" hidden={status !== "live"} />
          <span id="live-status">
            {status === "live"
              ? "Live"
              : status === "connecting"
                ? "Connecting"
                : "Reconnecting"}
          </span>
        </span>
      </div>
      <div className="activity-list" id="events">
        {events.slice(0, 7).map((event) => (
          <div className="event" key={event.id}>
            <div className="event-text">{eventText(event, tasks)}</div>
            <div className="event-time">{time(event.at)}</div>
          </div>
        ))}
        {!events.length && (
          <div className="event">
            <div className="event-text">Waiting for the first connection.</div>
          </div>
        )}
      </div>
    </section>
  );
}
