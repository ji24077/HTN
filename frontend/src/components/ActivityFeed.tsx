import { useState } from "react";
import type { AuditEvent, ConnectionStatus, Task } from "../api/types";
import { eventText, time } from "../lib/format";
import { Icon } from "./Icon";
export function ActivityFeed({
  events,
  tasks,
  status,
}: {
  events: AuditEvent[];
  tasks: Task[];
  status: ConnectionStatus;
}) {
  const [query, setQuery] = useState("");
  const [entity, setEntity] = useState("");
  const filtered = events.filter(
    (event) =>
      (!entity || event.entity === entity) &&
      `${eventText(event, tasks)} ${event.entity_id}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <section className="activity">
      <div className="section-heading">
        <h2>Fleet events</h2>
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
      <div className="activity-filters">
        <label className="search-field">
          <Icon name="search" size={16} />
          <input
            aria-label="Search fleet events"
            placeholder="Search events, tasks or workers…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <select
          aria-label="Filter fleet events"
          value={entity}
          onChange={(event) => setEntity(event.target.value)}
        >
          <option value="">All events</option>
          <option value="task">Tasks</option>
          <option value="worker">Workers</option>
          <option value="enrollment">Enrollments</option>
        </select>
      </div>
      <div className="activity-list" id="events">
        {filtered.slice(0, 100).map((event) => (
          <div className="event" key={event.id}>
            <div className="event-text">{eventText(event, tasks)}</div>
            <time className="event-time" dateTime={event.at}>
              {new Date(event.at).toLocaleDateString([], {
                month: "short",
                day: "numeric",
              })}{" "}
              · {time(event.at)}
            </time>
          </div>
        ))}
        {!filtered.length && (
          <p className="empty-state">
            {events.length
              ? "No events match these filters."
              : "Waiting for the first connection."}
          </p>
        )}
      </div>
      <p className="activity-foot">
        Showing {Math.min(filtered.length, 100)} of {filtered.length} matching
        events in the current feed · times are local.
      </p>
    </section>
  );
}
