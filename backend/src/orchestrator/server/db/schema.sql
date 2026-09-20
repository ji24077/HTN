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
-- Keep the agent's service decision available after the upload becomes a service.
ALTER TABLE hosted_services ADD COLUMN IF NOT EXISTS planning jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Recreate derived objects transactionally: prototype return/column layouts may
-- differ. Do not CASCADE; unknown external dependencies should stop the upgrade.
DROP VIEW IF EXISTS usage_record_totals;
DROP FUNCTION IF EXISTS worker_usage_quote(jsonb);

-- Rename the prototype's original monetary columns once. CAD is the sole unit.
DO $$
DECLARE item record;
BEGIN
    FOR item IN SELECT * FROM (VALUES
        ('TABLE','usage_pricing','hourly_rate_usd','hourly_rate_cad'),
        ('TABLE','usage_pricing','core_hour_usd','core_hour_cad'),
        ('TABLE','usage_pricing','ram_gib_hour_usd','ram_gib_hour_cad'),
        ('TABLE','usage_pricing','minimum_hour_usd','minimum_hour_cad'),
        ('TABLE','supervised_jobs','usage_cap_usd','usage_cap_cad'),
        ('TABLE','usage_records','hourly_rate_usd','hourly_rate_cad'),
        ('TABLE','usage_records','cost_usd','cost_cad')
    ) AS columns(kind,relation,old_name,new_name)
    LOOP
        IF EXISTS(SELECT 1 FROM information_schema.columns
            WHERE table_schema=current_schema() AND table_name=item.relation AND column_name=item.old_name) THEN
            EXECUTE format('ALTER %s %I RENAME COLUMN %I TO %I',item.kind,item.relation,item.old_name,item.new_name);
        END IF;
    END LOOP;
END;
$$;

-- Prototype CAD estimates, not provider prices or a payment balance. The legacy
-- flat rate is now an optional override; null selects machine-spec pricing.
CREATE TABLE IF NOT EXISTS usage_pricing (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    hourly_rate_cad numeric(16,6) NOT NULL
        CHECK (hourly_rate_cad >= 0 AND hourly_rate_cad <= 1000000000)
);
INSERT INTO usage_pricing(id,hourly_rate_cad) VALUES(true,1) ON CONFLICT DO NOTHING;
ALTER TABLE usage_pricing ALTER COLUMN hourly_rate_cad DROP NOT NULL;
ALTER TABLE usage_pricing ADD COLUMN IF NOT EXISTS spec_pricing_version integer;
ALTER TABLE usage_pricing ADD COLUMN IF NOT EXISTS core_hour_cad numeric(16,6) NOT NULL DEFAULT 0.002
    CHECK (core_hour_cad >= 0 AND core_hour_cad <= 1000000000);
ALTER TABLE usage_pricing ADD COLUMN IF NOT EXISTS ram_gib_hour_cad numeric(16,6) NOT NULL DEFAULT 0.001
    CHECK (ram_gib_hour_cad >= 0 AND ram_gib_hour_cad <= 1000000000);
ALTER TABLE usage_pricing ADD COLUMN IF NOT EXISTS minimum_hour_cad numeric(16,6) NOT NULL DEFAULT 0.01
    CHECK (minimum_hour_cad >= 0 AND minimum_hour_cad <= 0.5);
-- Switch the previous default once; preserve explicit non-default flat overrides.
UPDATE usage_pricing SET hourly_rate_cad=CASE WHEN hourly_rate_cad=1 THEN NULL ELSE hourly_rate_cad END,
    spec_pricing_version=1 WHERE spec_pricing_version IS NULL;

ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS usage_cap_cad numeric(16,6)
    CHECK (usage_cap_cad >= 0 AND usage_cap_cad <= 1000000000);
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS usage_cap_reached_at timestamptz;
ALTER TABLE supervised_jobs ADD COLUMN IF NOT EXISTS billing_account_id uuid;
CREATE INDEX IF NOT EXISTS jobs_billing_account ON supervised_jobs(billing_account_id)
    WHERE billing_account_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS usage_records (
    id bigserial PRIMARY KEY,
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    task_id text NOT NULL REFERENCES tasks(id),
    attempt integer NOT NULL CHECK (attempt > 0),
    worker_id text NOT NULL,
    hourly_rate_cad numeric(16,6) NOT NULL CHECK (hourly_rate_cad >= 0),
    started_at timestamptz NOT NULL,
    metering_until timestamptz NOT NULL,
    ended_at timestamptz,
    outcome text,
    duration_seconds numeric(20,6),
    cost_cad numeric(30,12),
    UNIQUE(task_id,attempt),
    CHECK ((ended_at IS NULL AND duration_seconds IS NULL AND cost_cad IS NULL AND outcome IS NULL)
        OR (ended_at IS NOT NULL AND duration_seconds IS NOT NULL AND cost_cad IS NOT NULL
            AND duration_seconds >= 0 AND cost_cad >= 0 AND outcome IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS usage_records_job ON usage_records(job_id,id);
CREATE INDEX IF NOT EXISTS usage_records_live ON usage_records(job_id) WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS jobs_with_usage_caps ON supervised_jobs(id)
    WHERE state IN ('active','paused') AND usage_cap_cad IS NOT NULL AND usage_cap_reached_at IS NULL;
ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS pricing_basis jsonb NOT NULL DEFAULT '{"model":"flat-v1"}';

-- Completed costs and attempt counts are maintained with the ledger transaction.
-- Hot-path reads combine these totals with only the indexed, still-live records.
CREATE TABLE IF NOT EXISTS usage_job_totals (
    job_id text PRIMARY KEY REFERENCES supervised_jobs(id),
    cost_cad numeric(30,12) NOT NULL DEFAULT 0,
    attempts bigint NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage_fleet_totals (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    cost_cad numeric(30,12) NOT NULL DEFAULT 0,
    attempts bigint NOT NULL DEFAULT 0
);
DO $$
BEGIN
    -- Backfill once, under the same startup transaction and scheduling lock.
    IF NOT EXISTS(SELECT 1 FROM usage_fleet_totals WHERE id) THEN
        INSERT INTO usage_job_totals(job_id,cost_cad,attempts)
        SELECT job_id,COALESCE(sum(cost_cad),0),count(*) FROM usage_records GROUP BY job_id
        ON CONFLICT(job_id) DO UPDATE SET cost_cad=EXCLUDED.cost_cad,attempts=EXCLUDED.attempts;
        INSERT INTO usage_fleet_totals(id,cost_cad,attempts)
        SELECT true,COALESCE(sum(cost_cad),0),count(*) FROM usage_records;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION update_usage_totals() RETURNS trigger AS $$
DECLARE
    cost_delta numeric := 0;
    attempt_delta bigint := 0;
    run_id text;
BEGIN
    IF TG_OP != 'INSERT' THEN
        run_id := OLD.job_id;
        cost_delta := -COALESCE(OLD.cost_cad,0);
        attempt_delta := -1;
    END IF;
    IF TG_OP != 'DELETE' THEN
        IF TG_OP = 'UPDATE' AND NEW.job_id IS DISTINCT FROM OLD.job_id THEN
            RAISE EXCEPTION 'Cannot move a usage record to another job';
        END IF;
        run_id := NEW.job_id;
        cost_delta := cost_delta+COALESCE(NEW.cost_cad,0);
        attempt_delta := attempt_delta+1;
    END IF;
    IF cost_delta != 0 OR attempt_delta != 0 THEN
        INSERT INTO usage_job_totals(job_id,cost_cad,attempts)
        VALUES(run_id,cost_delta,attempt_delta)
        ON CONFLICT(job_id) DO UPDATE SET
            cost_cad=usage_job_totals.cost_cad+EXCLUDED.cost_cad,
            attempts=usage_job_totals.attempts+EXCLUDED.attempts;
        UPDATE usage_fleet_totals SET cost_cad=cost_cad+cost_delta,attempts=attempts+attempt_delta WHERE id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Usage totals are missing; restart to rebuild them';
        END IF;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER usage_records_totals
AFTER INSERT OR UPDATE OR DELETE ON usage_records
FOR EACH ROW EXECUTE FUNCTION update_usage_totals();

-- Keep pricing in the database so every gateway and scheduler uses the same
-- rule. Missing specs use one core/one GiB and are explicitly marked inferred.
CREATE OR REPLACE FUNCTION worker_usage_quote(capabilities jsonb)
RETURNS TABLE(hourly_rate_cad numeric,pricing_basis jsonb) AS $$
    WITH specs AS (
        SELECT GREATEST(1,LEAST(4096,COALESCE((capabilities->'machine'->>'logical_cores')::numeric,1))) AS cores,
            GREATEST(1,LEAST(1048576,COALESCE(NULLIF((capabilities->'machine'->>'total_ram_mb')::numeric,0),1024)))/1024 AS ram_gib
    )
    SELECT COALESCE(p.hourly_rate_cad,
        round(LEAST(0.5,GREATEST(p.minimum_hour_cad,s.cores*p.core_hour_cad+s.ram_gib*p.ram_gib_hour_cad)),6)),
        jsonb_build_object(
            'currency','CAD','model',CASE WHEN p.hourly_rate_cad IS NULL THEN 'specs-v1' ELSE 'flat-override' END,
            'logical_cores',s.cores,'ram_gib',s.ram_gib,
            'inferred',capabilities->'machine'->>'logical_cores' IS NULL
                OR COALESCE((capabilities->'machine'->>'total_ram_mb')::numeric,0)<=0,
            'machine',capabilities->'machine',
            'core_hour_rate',p.core_hour_cad::text,'ram_gib_hour_rate',p.ram_gib_hour_cad::text,
            'minimum_hour_rate',p.minimum_hour_cad::text,'maximum_hour_rate','0.500000')
    FROM specs s CROSS JOIN (
        SELECT hourly_rate_cad,core_hour_cad,ram_gib_hour_cad,minimum_hour_cad
        FROM usage_pricing WHERE id
        UNION ALL SELECT NULL,0.002,0.001,0.01
        WHERE NOT EXISTS(SELECT 1 FROM usage_pricing WHERE id)
    ) p;
$$ LANGUAGE sql STABLE;

-- Task transitions are also made by supervision and simulation cancellation.
-- Meter them in the same transaction, regardless of which service changes state.
CREATE OR REPLACE FUNCTION meter_task_usage() RETURNS trigger AS $$
DECLARE
    stopped_at timestamptz;
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.state = 'running'
        AND (NEW.state != 'running' OR NEW.generation != OLD.generation) THEN
        stopped_at := LEAST(clock_timestamp(),OLD.lease_until,OLD.deadline);
        UPDATE usage_records SET
            ended_at=GREATEST(started_at,stopped_at),
            duration_seconds=GREATEST(0,extract(epoch FROM stopped_at-started_at)),
            cost_cad=round(GREATEST(0,extract(epoch FROM stopped_at-started_at))*hourly_rate_cad/3600,12),
            outcome=CASE WHEN NEW.state='queued' THEN 'retry' ELSE NEW.state END
        WHERE task_id=OLD.id AND attempt=OLD.generation AND ended_at IS NULL;
    END IF;
    IF NEW.state = 'running' AND NEW.worker_id IS NOT NULL
        AND NEW.spec->>'kind' != 'simulation_job' THEN
        INSERT INTO usage_records(job_id,task_id,attempt,worker_id,hourly_rate_cad,started_at,metering_until,pricing_basis)
        SELECT NEW.spec->>'job_id',NEW.id,NEW.generation,NEW.worker_id,q.hourly_rate_cad,
            COALESCE(NEW.started_at,clock_timestamp()),LEAST(NEW.lease_until,NEW.deadline),
            q.pricing_basis
        FROM workers w JOIN supervised_jobs j ON j.id=NEW.spec->>'job_id'
        CROSS JOIN LATERAL worker_usage_quote(w.capabilities) q WHERE w.id=NEW.worker_id
        ON CONFLICT(task_id,attempt) DO UPDATE SET metering_until=EXCLUDED.metering_until
            WHERE usage_records.ended_at IS NULL
                AND usage_records.metering_until IS DISTINCT FROM EXCLUDED.metering_until;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER tasks_meter_usage
AFTER INSERT OR UPDATE OF state,generation,lease_until,deadline ON tasks
FOR EACH ROW EXECUTE FUNCTION meter_task_usage();

-- Existing in-flight work starts tracking at migration time. Do not invent
-- historical usage for completed tasks or double count on subsequent startups.
INSERT INTO usage_records(job_id,task_id,attempt,worker_id,hourly_rate_cad,started_at,metering_until,pricing_basis)
SELECT t.spec->>'job_id',t.id,t.generation,t.worker_id,q.hourly_rate_cad,
    clock_timestamp(),LEAST(t.lease_until,t.deadline),
    q.pricing_basis
FROM tasks t JOIN supervised_jobs j ON j.id=t.spec->>'job_id'
JOIN workers w ON w.id=t.worker_id
CROSS JOIN LATERAL worker_usage_quote(w.capabilities) q
WHERE t.state='running' AND t.worker_id IS NOT NULL
    AND t.spec->>'kind' != 'simulation_job'
ON CONFLICT(task_id,attempt) DO NOTHING;

CREATE OR REPLACE VIEW usage_record_totals AS
SELECT u.id,u.job_id,u.task_id,u.attempt,u.worker_id,u.hourly_rate_cad,u.started_at,
    u.metering_until,u.ended_at,u.outcome,u.duration_seconds,u.cost_cad,
    COALESCE(duration_seconds,GREATEST(0,extract(epoch FROM LEAST(statement_timestamp(),metering_until)-started_at))) AS elapsed_seconds,
    COALESCE(cost_cad,round(GREATEST(0,extract(epoch FROM LEAST(statement_timestamp(),metering_until)-started_at))*hourly_rate_cad/3600,12)) AS estimated_cost_cad,
    u.pricing_basis
FROM usage_records u;

CREATE OR REPLACE TRIGGER usage_records_changed
AFTER INSERT OR UPDATE ON usage_records
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();


-- Manually granted prototype credit, scoped to the verified Supabase account.
-- No auth.users FK: isolated/demo databases need not install Supabase auth.
CREATE TABLE IF NOT EXISTS account_credits (
    id uuid PRIMARY KEY,
    account_id uuid NOT NULL,
    amount_cad numeric(16,2) NOT NULL CHECK (amount_cad > 0 AND amount_cad <= 1000000000),
    reason text NOT NULL CHECK (length(trim(reason)) > 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS account_credits_account ON account_credits(account_id);
ALTER TABLE account_credits ENABLE ROW LEVEL SECURITY;
CREATE OR REPLACE TRIGGER account_credits_changed
AFTER INSERT OR UPDATE OR DELETE ON account_credits
FOR EACH ROW EXECUTE FUNCTION notify_orchestrator_change();
-- Outputs survive worker cleanup. Only the accepted current attempt is downloadable.
CREATE TABLE IF NOT EXISTS job_outputs (
    id uuid PRIMARY KEY,
    job_id text NOT NULL REFERENCES supervised_jobs(id),
    task_id text NOT NULL REFERENCES tasks(id),
    attempt integer NOT NULL,
    name text NOT NULL,
    digest text NOT NULL,
    size bigint NOT NULL CHECK (size >= 0),
    content bytea,
    ready boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(task_id,attempt,name)
);
CREATE INDEX IF NOT EXISTS job_outputs_job ON job_outputs(job_id);
-- Preserve existing inline files while new outputs use bounded chunks.
ALTER TABLE job_outputs ALTER COLUMN content DROP NOT NULL;
ALTER TABLE job_outputs ADD COLUMN IF NOT EXISTS ready boolean NOT NULL DEFAULT true;
ALTER TABLE job_outputs DROP CONSTRAINT IF EXISTS job_outputs_size_check;
ALTER TABLE job_outputs ADD CONSTRAINT job_outputs_size_check
    CHECK (size >= 0);
CREATE TABLE IF NOT EXISTS job_output_chunks (
    output_id uuid NOT NULL REFERENCES job_outputs(id) ON DELETE CASCADE,
    part integer NOT NULL CHECK (part >= 0),
    content bytea NOT NULL CHECK (octet_length(content) <= 1048576),
    PRIMARY KEY(output_id,part)
);
