CREATE TABLE IF NOT EXISTS workers (
    id text PRIMARY KEY,
    session_id text NOT NULL,
    capabilities jsonb NOT NULL,
    state text NOT NULL CHECK (state IN ('alive', 'unhealthy', 'offline')),
    last_seen timestamptz NOT NULL,
    paused boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS tasks (
    id text PRIMARY KEY,
    spec jsonb NOT NULL,
    state text NOT NULL CHECK (state IN ('queued', 'assigned', 'running', 'succeeded', 'failed', 'cancelled')),
    generation integer NOT NULL DEFAULT 0,
    worker_id text,
    session_id text,
    lease_until timestamptz,
    deadline timestamptz,
    result jsonb,
    failure text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks(created_at, id) WHERE state = 'queued';
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS progress double precision NOT NULL DEFAULT 0;
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS started_at timestamptz;
CREATE INDEX IF NOT EXISTS tasks_leases ON tasks(lease_until) WHERE state IN ('assigned', 'running');
CREATE UNIQUE INDEX IF NOT EXISTS one_task_per_worker ON tasks(worker_id) WHERE state IN ('assigned', 'running');

CREATE TABLE IF NOT EXISTS events (
    id bigserial PRIMARY KEY,
    at timestamptz NOT NULL DEFAULT clock_timestamp(),
    entity text NOT NULL,
    entity_id text NOT NULL,
    previous_state text NOT NULL,
    new_state text NOT NULL,
    details jsonb NOT NULL
);

-- NOTIFY is delivered only after commit. Identical notifications within one
-- transaction coalesce. Row triggers keep no-op reconciliation scans quiet.
CREATE OR REPLACE FUNCTION notify_orchestrator_change() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('orchestrator_changes', '');
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER workers_changed
AFTER INSERT OR UPDATE OR DELETE ON workers
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();
CREATE OR REPLACE TRIGGER tasks_changed
AFTER INSERT OR UPDATE OR DELETE ON tasks
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();
CREATE OR REPLACE TRIGGER events_changed
AFTER INSERT OR UPDATE OR DELETE ON events
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();
