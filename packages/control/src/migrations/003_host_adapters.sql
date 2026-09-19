-- What each host can actually run.
--
-- Without this the scheduler offers work to any online host and relies on the agent to
-- decline what it cannot do. The task then returns to the queue and is offered straight
-- back to the same host — a loop that burns CPU on both sides and fills the log, for a
-- job that can never run there.
alter table hosts add column adapters text[] not null default '{}';
