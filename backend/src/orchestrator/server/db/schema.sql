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
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS attestation jsonb;
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
CREATE INDEX IF NOT EXISTS events_task_history ON events(entity, entity_id, id);

CREATE TABLE IF NOT EXISTS execution_events (
    id bigserial PRIMARY KEY,
    task_id text NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt integer NOT NULL,
    worker_id text,
    source text NOT NULL CHECK (source IN ('server', 'worker')),
    sequence bigint NOT NULL,
    kind text NOT NULL,
    occurred_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    data jsonb NOT NULL,
    UNIQUE(task_id, attempt, source, sequence)
);
CREATE INDEX IF NOT EXISTS execution_events_task ON execution_events(task_id, id);

-- Only credential digests are retained. Auth keys are returned once, never stored.
CREATE TABLE IF NOT EXISTS worker_enrollments (
    worker_id text PRIMARY KEY,
    request_id uuid UNIQUE NOT NULL,
    user_id uuid NOT NULL,
    display_name text NOT NULL,
    token_hash text NOT NULL,
    state text NOT NULL CHECK (state IN ('pending', 'active', 'failed')),
    tailscale_key_id text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS worker_enrollments_user_time ON worker_enrollments(user_id, created_at);

-- DWP devices share workers/tasks and the existing scheduler. Pair codes are
-- hashed, expire after ten minutes and are consumed in the device transaction.
CREATE TABLE IF NOT EXISTS dwp_pair_codes (
    code_hash text PRIMARY KEY,
    owner_id uuid,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    used_at timestamptz
);
CREATE INDEX IF NOT EXISTS dwp_pair_codes_owner_time ON dwp_pair_codes(owner_id, created_at);
CREATE INDEX IF NOT EXISTS dwp_pair_codes_expiry ON dwp_pair_codes(expires_at);

CREATE TABLE IF NOT EXISTS dwp_devices (
    worker_id text PRIMARY KEY,
    owner_id uuid,
    public_key text UNIQUE NOT NULL,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS dwp_devices_owner ON dwp_devices(owner_id);

-- How the operator has set this machine's accelerator, and what the machine said back.
--
-- The preference lives here rather than only on the device because the dashboard must be
-- able to set it while the machine is offline: a laptop that is asleep when the switch is
-- flipped has to pick the change up when it reconnects, which it does by the server
-- replaying this value at hello. The device remains free to refuse -- `runtime_applied`
-- records whether it did, so the dashboard shows what is true rather than what was asked.
ALTER TABLE dwp_devices ADD COLUMN IF NOT EXISTS runtime_preference text NOT NULL DEFAULT 'auto';
ALTER TABLE dwp_devices ADD COLUMN IF NOT EXISTS runtime_applied boolean;
ALTER TABLE dwp_devices ADD COLUMN IF NOT EXISTS runtime_detail text NOT NULL DEFAULT '';
ALTER TABLE dwp_devices DROP CONSTRAINT IF EXISTS dwp_devices_runtime_preference_check;
ALTER TABLE dwp_devices ADD CONSTRAINT dwp_devices_runtime_preference_check
    CHECK (runtime_preference IN ('auto', 'cpu'));

-- The key includes the device ID and is durable across gateway processes.
CREATE TABLE IF NOT EXISTS dwp_assertions (
    worker_id text NOT NULL REFERENCES dwp_devices(worker_id) ON DELETE CASCADE,
    jti uuid NOT NULL,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (worker_id, jti)
);
CREATE INDEX IF NOT EXISTS dwp_assertions_expiry ON dwp_assertions(expires_at);

-- Chat history is server-owned and never included in fleet snapshots.
CREATE TABLE IF NOT EXISTS chat_conversations (
    id uuid PRIMARY KEY,
    owner text NOT NULL,
    data jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS chat_conversations_owner ON chat_conversations(owner);

CREATE TABLE IF NOT EXISTS supervised_jobs (
    id text PRIMARY KEY,
    state text NOT NULL DEFAULT 'active'
        CHECK (state IN ('active','paused','succeeded','failed','cancelled')),
    instructions text NOT NULL DEFAULT '',
    memory jsonb NOT NULL DEFAULT '{"findings":[],"questions":[],"followups":[]}',
    event_cursor bigint NOT NULL DEFAULT 0,
    next_check_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    retry_after timestamptz NOT NULL DEFAULT clock_timestamp(),
    finalized boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks((spec->>'job_id'));
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS active_run uuid;
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS revision bigint NOT NULL DEFAULT 0;
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS sentry_cursor text;
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS sentry_start timestamptz;
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS sentry_end timestamptz;
CREATE TABLE IF NOT EXISTS supervisor_events (
    id bigserial PRIMARY KEY,
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    kind text NOT NULL,
    data jsonb NOT NULL,
    dedupe_key text,
    at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(job_id, dedupe_key)
);
CREATE TABLE IF NOT EXISTS job_reservations (
    worker_id text PRIMARY KEY REFERENCES workers(id),
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS supervisor_events_job ON supervisor_events(job_id,id);
CREATE TABLE IF NOT EXISTS supervisor_runs (
    id uuid PRIMARY KEY,
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    event_cursor bigint NOT NULL,
    conversation jsonb NOT NULL,
    status text NOT NULL DEFAULT 'running',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS supervisor_runs_job ON supervisor_runs(job_id,created_at);
CREATE TABLE IF NOT EXISTS supervisor_actions (
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    action_id uuid NOT NULL,
    request jsonb NOT NULL,
    result jsonb NOT NULL,
    at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(job_id,action_id)
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

CREATE OR REPLACE TRIGGER supervisor_events_changed
AFTER INSERT ON supervisor_events
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();

-- Source bundles are immutable and scoped to one simulation job.
CREATE TABLE IF NOT EXISTS simulation_jobs (
    job_id text PRIMARY KEY REFERENCES supervised_jobs(id),
    submission_hash text NOT NULL,
    original_hash text NOT NULL,
    phase text NOT NULL DEFAULT 'submitted',
    data jsonb NOT NULL,
    revision bigint NOT NULL DEFAULT 0,
    retry_after timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    deadline timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS simulation_artifacts (
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    digest text NOT NULL,
    content bytea NOT NULL,
    PRIMARY KEY(job_id,digest)
);
CREATE OR REPLACE TRIGGER simulation_jobs_changed
AFTER INSERT OR UPDATE OR DELETE ON simulation_jobs
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();

-- Persistent uploaded services share tasks, worker leases, logs and supervisors.
CREATE TABLE IF NOT EXISTS hosted_services (
    job_id text PRIMARY KEY REFERENCES supervised_jobs(id),
    submission_hash text NOT NULL,
    bundle_hash text NOT NULL,
    description text NOT NULL,
    config jsonb NOT NULL,
    desired text NOT NULL DEFAULT 'running' CHECK(desired IN ('running','stopped')),
    phase text NOT NULL DEFAULT 'pending',
    message text NOT NULL DEFAULT 'Waiting for a compatible worker.',
    task_id text,
    revision bigint NOT NULL DEFAULT 0,
    attempts integer NOT NULL DEFAULT 0,
    failures jsonb NOT NULL DEFAULT '[]',
    retry_after timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz,
    ready_at timestamptz,
    health_at timestamptz
);
