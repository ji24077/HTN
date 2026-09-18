create table users (
  id            uuid primary key default gen_random_uuid(),
  email         text not null unique,
  password_hash text not null,
  role          text not null default 'member' check (role in ('member','admin')),
  created_at    timestamptz not null default now()
);

create table sessions (
  token_hash  text primary key,
  user_id     uuid not null references users(id) on delete cascade,
  expires_at  timestamptz not null,
  created_at  timestamptz not null default now()
);

create table hosts (
  id                   uuid primary key default gen_random_uuid(),
  owner_id             uuid not null references users(id) on delete restrict,
  label                text not null,
  public_key           text not null,              -- base64 SPKI, Ed25519
  trust_tier           text not null default 'trusted' check (trust_tier in ('trusted','semi','untrusted')),
  allow_compute        boolean not null default true,
  allow_browser        boolean not null default false,
  paused               boolean not null default false,
  os                   text,
  arch                 text,
  cpu_model            text,
  logical_cores        int,
  total_ram_mb         int,
  free_ram_mb          int,
  agent_version        text,
  max_concurrency      int not null default 2,
  self_reported_region text,
  online               boolean not null default false,
  last_heartbeat_at    timestamptz,
  created_at           timestamptz not null default now(),
  revoked_at           timestamptz
);
create index hosts_owner_idx on hosts(owner_id);

-- Only ever stores a hash: a database read must not yield a usable pairing code.
create table pair_codes (
  code_hash   text primary key,
  user_id     uuid not null references users(id) on delete cascade,
  label       text not null,
  expires_at  timestamptz not null,
  consumed_at timestamptz,
  created_at  timestamptz not null default now()
);

create table jobs (
  id                  uuid primary key default gen_random_uuid(),
  owner_id            uuid not null references users(id) on delete restrict,
  adapter             text not null,
  status              text not null default 'running'
                        check (status in ('running','succeeded','failed','cancelled','cancelled_partial')),
  manifest_hash       text,
  total_items         int not null,
  constraints         jsonb not null default '{}'::jsonb,
  created_at          timestamptz not null default now(),
  started_at          timestamptz,
  ended_at            timestamptz,
  cancel_requested_at timestamptz
);

create table tasks (
  id               uuid primary key default gen_random_uuid(),
  job_id           uuid not null references jobs(id) on delete cascade,
  seq              int not null,
  input            jsonb not null,
  pin_host_id      uuid references hosts(id) on delete set null,
  state            text not null default 'pending'
                     check (state in ('pending','offered','leased','running','succeeded','failed','cancelled')),
  assigned_host_id uuid references hosts(id) on delete set null,
  lease_id         uuid,
  lease_expires_at timestamptz,
  attempts         int not null default 0,
  output           jsonb,
  output_hash      text,
  error_class      text,
  queued_at        timestamptz not null default now(),
  started_at       timestamptz,
  finished_at      timestamptz,
  unique (job_id, seq)
);
create index tasks_claim_idx on tasks(job_id, state, seq);
create index tasks_lease_idx on tasks(lease_expires_at) where lease_expires_at is not null;

-- Every attempt survives, so the run map can show "ran twice, first accepted".
create table task_attempts (
  id                uuid primary key default gen_random_uuid(),
  task_id           uuid not null references tasks(id) on delete cascade,
  attempt           int not null,
  host_id           uuid references hosts(id) on delete set null,
  outcome           text not null check (outcome in ('succeeded','failed','lease_expired','cancelled','duplicate')),
  host_signature    text,
  signature_ok      boolean,
  host_reported_ms  numeric,
  started_at        timestamptz,
  finished_at       timestamptz,
  created_at        timestamptz not null default now(),
  unique (task_id, attempt)
);

create table run_events (
  seq        bigserial primary key,
  id         uuid not null default gen_random_uuid(),
  job_id     uuid references jobs(id) on delete cascade,
  task_id    uuid references tasks(id) on delete cascade,
  host_id    uuid references hosts(id) on delete set null,
  actor      text not null,
  category   text not null,
  type       text not null,
  payload    jsonb not null default '{}'::jsonb,
  server_ts  timestamptz not null default now()
);
create index run_events_job_idx on run_events(job_id, seq);
