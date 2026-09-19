"""Application assembly, service lifecycle, health, and server entry point."""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from ..shared.protocol import MESSAGE_LIMIT
from ..shared.security import authorized
from .config import ServerConfig
from .dashboard import router as dashboard_router
from .db.store import Conflict, NotFound, Store
from .routes import router as api_router
from .scheduler import reconcile_loop
from .supabase_auth import SupabaseAuth
from .updates import ChangeFeed
from .worker_connection import serve_worker

log = logging.getLogger(__name__)


def create_app(surface: str = "combined") -> FastAPI:
    if surface not in {"combined", "public", "worker"}:
        raise ValueError("Unknown server surface")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = ServerConfig.from_env()
        if surface == "public" and not config.public_origin:
            raise ValueError("PUBLIC_ORIGIN is required for the public server")
        if surface == "public" and not (
            config.supabase_url and config.supabase_publishable_key
            and (config.supabase_admin_ids or config.supabase_admin_emails)
        ):
            raise ValueError(
                "Public server requires Supabase URL/key and approved admin IDs or emails"
            )
        if surface == "combined" and config.public_origin:
            raise ValueError(
                "Use orchestrator-public and orchestrator-worker-gateway with PUBLIC_ORIGIN"
            )
        store = await Store.open(config.database_url, schema=config.database_schema)
        cache = None
        reconciler = None
        updates = ChangeFeed(config.database_url)
        auth = SupabaseAuth(config.supabase_url) if config.supabase_url else None
        try:
            if config.redis_url:
                cache = Redis.from_url(
                    config.redis_url,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                    retry=Retry(NoBackoff(), 0),
                    decode_responses=True,
                )
            app.state.config, app.state.store, app.state.cache = config, store, cache
            app.state.supabase_auth = auth
            # Keep the local demo cookie valid across server restarts so the
            # browser can reconnect its stream. Rotating the admin token revokes it.
            app.state.ui_session = hmac.new(
                config.admin_token.encode(), b"orchestrator-local-demo-session-v1", "sha256"
            ).hexdigest()
            app.state.updates = updates
            updates.start()
            reconciler = asyncio.create_task(reconcile_loop(store))
            yield
        finally:
            await updates.close()
            if reconciler is not None:
                reconciler.cancel()
                await asyncio.gather(reconciler, return_exceptions=True)
            try:
                if auth is not None:
                    await auth.close()
                if cache is not None:
                    await cache.aclose()
            finally:
                await store.close()

    app = FastAPI(
        title="GPU orchestration prototype",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.surface = surface

    @app.exception_handler(Conflict)
    async def conflict(_request: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"error": str(exc)})

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    async def unavailable(_request: Request, exc: Exception):
        log.error("database operation unavailable: %s", type(exc).__name__)
        return JSONResponse(status_code=503, content={"error": "database unavailable"})

    app.add_exception_handler(asyncpg.PostgresError, unavailable)
    app.add_exception_handler(asyncpg.InterfaceError, unavailable)
    app.add_exception_handler(TimeoutError, unavailable)
    app.add_exception_handler(ConnectionError, unavailable)

    @app.get("/healthz")
    async def health(request: Request):
        async with asyncio.timeout(2):
            await request.app.state.store.pool.fetchval("SELECT 1")
        return {"status": "ok"}

    async def worker(socket: WebSocket):
        config = socket.app.state.config
        worker_id = socket.headers.get("x-worker-id", "")
        if not authorized(socket.headers.get("authorization"), config.worker_tokens.get(worker_id)):
            await socket.close(code=1008)
            return
        await serve_worker(socket, socket.app.state.store, socket.app.state.cache, worker_id)

    if surface in {"combined", "worker"}:
        app.add_api_websocket_route("/v1/worker", worker)
    if surface in {"combined", "public"}:
        app.include_router(api_router)
        app.include_router(dashboard_router)
    return app


def create_public_app() -> FastAPI:
    return create_app("public")


def create_worker_app() -> FastAPI:
    return create_app("worker")


def run(factory: str, host: str, port: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        f"orchestrator.server.app:{factory}",
        factory=True,
        host=host,
        port=port,
        ws="websockets",
        ws_max_size=MESSAGE_LIMIT,
        timeout_graceful_shutdown=10,
        proxy_headers=False,
    )


def main() -> None:
    run("create_app", os.getenv("LISTEN_HOST", "127.0.0.1"), int(os.getenv("LISTEN_PORT", "8080")))


def public_main() -> None:
    run(
        "create_public_app",
        os.getenv("PUBLIC_BIND_HOST", "127.0.0.1"),
        int(os.getenv("PUBLIC_PORT", "8080")),
    )


def worker_main() -> None:
    run("create_worker_app", "127.0.0.1", int(os.getenv("WORKER_PORT", "8081")))


if __name__ == "__main__":
    main()
