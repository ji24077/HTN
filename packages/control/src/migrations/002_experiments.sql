-- Long-running experiments that span many jobs.
--
-- A single job is one generation of an evolutionary run; the experiment is the thread
-- connecting them. Kept as one row of JSON because the shape belongs to the experiment,
-- not to the platform, and the platform should not need a migration to learn a new one.
create table experiments (
  name       text primary key,
  owner_id   uuid not null references users(id) on delete cascade,
  state      jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
