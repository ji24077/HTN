"""Periodic recovery of expired leases and worker presence."""

import asyncio
import logging

from .db.store import Store

log = logging.getLogger(__name__)


async def reconcile_loop(store: Store) -> None:
    while True:
        try:
            await store.reconcile()
            from .services import ServiceStore
            await ServiceStore(store).reconcile()
        except Exception:
            log.exception("reconciliation failed")
        await asyncio.sleep(1)
