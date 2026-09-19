"""Application assembly, service lifecycle, health, and server entry point."""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import sentry_sdk
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from ..agent import AgentLoop
from ..llm import OpenAIClient
from ..shared.protocol import MESSAGE_LIMIT
from ..shared.security import authorized
from ..shared.telemetry import init_sentry
from ..preprocessing.routes import router as preprocessing_router, worker_router as artifact_router
from ..preprocessing.service import PreprocessingService
from ..supervisor.routes import router as supervisor_router
from ..supervisor.sentry import SentryReader
from ..supervisor.service import SupervisorService
from .chat import router as chat_router
from .chat_store import ChatStore
from .config import ServerConfig
from .dashboard import router as dashboard_router
from .db.store import Conflict, NotFound, Store
from .dwp import router as dwp_router
from .dwp_assets import router as dwp_assets_router
from .enrollment import TailscaleEnrollment
from .enrollment import router as enrollment_router
from .routes import router as api_router
from .scheduler import reconcile_loop
from .supabase_auth import SupabaseAuth
from .updates import ChangeFeed
from .worker_connection import serve_worker

log = logging.getLogger(__name__)


def create_app(surface: str = "combined") -> FastAPI:
    if surface not in {"combined", "public", "worker"}:
        raise ValueError("Unknown server surface")
    init_sentry("server", surface=surface)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = ServerConfig.from_env()
        if surface == "public" and not config.public_origin:
            raise ValueError("PUBLIC_ORIGIN is required for the public server")
        if surface == "public" and not (
            config.supabase_url
            and config.supabase_publishable_key
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
        model_client = None
        supervisor_task = None
        preprocessing_task = None
        sentry_reader = None
        updates = ChangeFeed(config.database_url)
        auth = SupabaseAuth(config.supabase_url) if config.supabase_url else None
        enrollment = (
            TailscaleEnrollment(config)
            if surface == "public"
            and config.tailscale_oauth_client_id
            and config.tailscale_oauth_client_secret
            and config.worker_gateway_url
            else None
        )
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
            app.state.enrollment = enrollment
            app.state.chat_store = ChatStore(store.pool)
            if surface != "worker" and os.getenv("OPENAI_API_KEY"):
                try:
                    model_client = OpenAIClient.from_env()
                except ValueError as exc:
                    # A partial or malformed OPENAI_* configuration disables chat;
                    # the dashboard and worker gateway keep running.
                    log.warning("chat disabled: %s", exc)
                else:
                    app.state.chat_agent = AgentLoop(model_client)
            if (
                surface == "combined"
                and not config.worker_tokens
                and not await store.has_active_enrollments()
            ):
                log.warning(
                    "no worker can authenticate: set WORKER_TOKENS or enroll workers "
                    "through the public server"
                )
            # Keep the local demo cookie valid across server restarts so the
            # browser can reconnect its stream. Rotating the admin token revokes it.
            app.state.ui_session = hmac.new(
                config.admin_token.encode(), b"orchestrator-local-demo-session-v1", "sha256"
            ).hexdigest()
            app.state.updates = updates
            updates.start()
            reconciler = asyncio.create_task(reconcile_loop(store))
            if (
                model_client is not None
                and os.getenv("SUPERVISOR_ENABLED", "true").lower() == "true"
            ):
                sentry_reader = SentryReader.from_env()
                app.state.supervisor_sentry = sentry_reader
                app.state.supervisor = SupervisorService(store, model_client, sentry_reader)
                supervisor_task = asyncio.create_task(app.state.supervisor.run(updates))
            if model_client is not None:
                app.state.preprocessing = PreprocessingService(store, model_client)
                preprocessing_task = asyncio.create_task(app.state.preprocessing.run(updates))
            yield
        finally:
            if preprocessing_task is not None:
                preprocessing_task.cancel()
                await asyncio.gather(preprocessing_task, return_exceptions=True)
            if supervisor_task is not None:
                supervisor_task.cancel()
                await asyncio.gather(supervisor_task, return_exceptions=True)
            if sentry_reader is not None:
                await sentry_reader.close()
            await updates.close()
            if reconciler is not None:
                reconciler.cancel()
                await asyncio.gather(reconciler, return_exceptions=True)
            try:
                if model_client is not None:
                    await model_client.aclose()
                if auth is not None:
                    await auth.close()
                if enrollment is not None:
                    await enrollment.close()
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
    app.state.chat_agent = None
    app.state.preprocessing = None
    app.state.supervisor = None
    app.state.supervisor_sentry = None
    app.state.chat_slots = asyncio.Semaphore(2)

    @app.exception_handler(Conflict)
    async def conflict(_request: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"error": str(exc)})

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    async def unavailable(_request: Request, exc: Exception):
        sentry_sdk.capture_exception(exc)
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
        header = socket.headers.get("authorization")
        allowed = authorized(header, config.worker_tokens.get(worker_id))
        enrolled = False
        if not allowed and worker_id and header and worker_id not in config.worker_tokens:
            allowed = enrolled = await socket.app.state.store.worker_authorized(worker_id, header)
        if not allowed:
            await socket.close(code=1008)
            return
        await serve_worker(
            socket, socket.app.state.store, socket.app.state.cache, worker_id, enrolled
        )

    if surface in {"combined", "worker"}:
        app.add_api_websocket_route("/v1/worker", worker)
        app.include_router(artifact_router)
    if surface in {"combined", "public"}:
        app.include_router(api_router)
        app.include_router(chat_router)
        app.include_router(supervisor_router)
        app.include_router(preprocessing_router)
        app.include_router(dashboard_router)
        app.include_router(dwp_router)
        app.include_router(dwp_assets_router)
    if surface == "public":
        app.include_router(enrollment_router)
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
