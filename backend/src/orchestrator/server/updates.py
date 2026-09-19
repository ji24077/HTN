"""Committed database changes wake dashboard streams; no snapshot polling."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import asyncpg

from .db.connection import connection_options

log = logging.getLogger(__name__)
CHANNEL = "orchestrator_changes"


class ChangeFeed:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connected = False
        self.subscribers: set[asyncio.Event] = set()
        self.task: asyncio.Task | None = None

    def notify(self, *_args) -> None:
        # Coalesce bursts: a slow browser needs the latest state, not an
        # unbounded queue of snapshots. PostgreSQL remains authoritative.
        for changed in self.subscribers:
            changed.set()

    @asynccontextmanager
    async def subscribe(self):
        changed = asyncio.Event()
        self.subscribers.add(changed)
        changed.set()  # Every connection starts with a fresh snapshot.
        try:
            yield changed
        finally:
            self.subscribers.discard(changed)

    def start(self) -> None:
        self.task = asyncio.create_task(self.run())

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def run(self) -> None:
        while True:
            conn = None
            try:
                # Dedicated LISTEN connection, shared by all dashboard clients.
                conn = await asyncpg.connect(
                    self.database_url,
                    timeout=5,
                    command_timeout=5,
                    **connection_options(self.database_url),
                )
                disconnected = asyncio.Event()
                conn.add_termination_listener(lambda _conn: disconnected.set())
                await conn.add_listener(CHANNEL, self.notify)
                self.connected = True
                self.notify()  # Resync changes missed while disconnected.
                while not disconnected.is_set():
                    try:
                        await asyncio.wait_for(disconnected.wait(), timeout=30)
                    except TimeoutError:
                        # Detect a broken DB link; this does not fetch UI state.
                        await conn.execute("SELECT 1")
            except (asyncpg.PostgresError, OSError, TimeoutError, asyncpg.InterfaceError):
                log.warning("dashboard change feed disconnected; reconnecting")
            finally:
                self.connected = False
                self.notify()
                if conn is not None:
                    with suppress(Exception):
                        await conn.close(timeout=2)
            await asyncio.sleep(1)
