"""Outbound asyncio worker; execution remains behind the Executor protocol."""

import asyncio
import hashlib
import logging
import os
import platform
import random
import signal
import time
import traceback
from collections.abc import Callable

import sentry_sdk
from websockets.asyncio.client import ClientConnection, connect

from ..shared.protocol import (
    HEARTBEAT_INTERVAL,
    MESSAGE_LIMIT,
    UNHEALTHY_AFTER,
    VERSION,
    Executor,
    Message,
    Ref,
    Task,
    bounded_json,
    json_loads,
    json_text,
    task_ref,
)
from ..shared.telemetry import init_sentry
from .config import WorkerConfig
from .execution import ExecutionJournal, ExecutionReporter
from .executors import StubExecutor
from .tunnel import EmbeddedTunnel

log = logging.getLogger(__name__)


class DirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # Never forward the enrolled worker's credential to a redirected host.
        return exc


async def send(socket: ClientConnection, message: Message) -> None:
    async with asyncio.timeout(5):
        await socket.send(message.model_dump_json(exclude_none=True))


async def receive(socket: ClientConnection) -> Message:
    message = Message.model_validate(json_loads(await socket.recv()))
    if message.version != VERSION:
        raise ValueError("unsupported server protocol version")
    return message


async def execute(executor: Executor, task: Task, report: Callable[[float], None]) -> Message:
    ref = task_ref(task)
    started = time.monotonic()

    def record(kind, **data):
        if isinstance(report, ExecutionReporter):
            report.emit(kind, {"duration_ms": (time.monotonic() - started) * 1000, **data})

    record(
        "started",
        adapter=task.spec.kind,
        runtime=f"python {platform.python_version()}",
        os=platform.system(),
        arch=platform.machine(),
        task_hash=hashlib.sha256(json_text(task.spec.model_dump(mode="json")).encode()).hexdigest(),
    )
    with (
        sentry_sdk.new_scope() as scope,
        sentry_sdk.start_transaction(op="task.execute", name=f"task {task.spec.kind}") as span,
    ):
        scope.set_tag("execution_id", f"{task.spec.id}:{task.generation}")
        scope.set_tag("task_id", task.spec.id)
        scope.set_tag("job_id", task.spec.job_id)
        scope.set_tag("worker_id", task.worker_id)
        scope.set_tag("reservation_id", f"{task.spec.id}:{task.generation}")
        scope.set_tag("attempt", task.generation)
        span.set_tag("task_id", task.spec.id)
        span.set_tag("job_id", task.spec.job_id)
        span.set_tag("execution_id", f"{task.spec.id}:{task.generation}")
        span.set_data("generation", task.generation)
        try:
            async with asyncio.timeout(task.spec.timeout_seconds):
                result = bounded_json(await executor.execute(task.spec, report))
            if (
                task.spec.kind == "python_project"
                and isinstance(result, dict)
                and result.get("ok") is False
            ):
                error = (
                    "project_code: " + str(result.get("error", "Project execution failed"))[:1900]
                )
                record("failed", message=error)
                return Message(type="failed", ref=ref, error=error, retryable=False)
            report(100)
            record("succeeded", result_url=f"/v1/tasks/{task.spec.id}")
            return Message(type="complete", ref=ref, result=result)
        except TimeoutError as exc:
            record("timed_out", message="Execution deadline exceeded")
            sentry_sdk.capture_exception(exc)
            span.set_status("deadline_exceeded")
            return Message(type="failed", ref=ref, error="execution deadline", retryable=True)
        except asyncio.CancelledError:
            record("interrupted", message="Execution stopped or assignment revoked")
            raise
        except Exception as exc:
            record(
                "failed",
                error={
                    "name": type(exc).__name__,
                    "message": str(exc),
                    "stack": traceback.format_exc()[-4096:],
                },
            )
            # Cancellation is BaseException and must propagate without a completion.
            sentry_sdk.capture_exception(exc)
            span.set_status("internal_error")
            return Message(type="failed", ref=ref, error=(str(exc) or type(exc).__name__)[:2048])


class Agent:
    def __init__(
        self, config: WorkerConfig, executor: Executor, tunnel: EmbeddedTunnel | None = None
    ):
        self.config = config
        self.executor = executor
        self.tunnel = tunnel
        self.journal = ExecutionJournal(config.url, config.worker_id)

    async def run(self) -> None:
        delay = 1.0
        while True:
            started = asyncio.get_running_loop().time()
            try:
                await self.session()
            except Exception as exc:
                # Log the exception class, not connection headers or credentials.
                # Sentry receives the traceback only after the telemetry scrubber.
                sentry_sdk.capture_exception(exc)
                log.warning("worker disconnected: %s", type(exc).__name__)
            if asyncio.get_running_loop().time() - started > 60:
                delay = 1.0
            await asyncio.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 30)

    async def session(self) -> None:
        sock = await self.tunnel.open_socket() if self.tunnel else None
        try:
            await self.connected_session(sock)
        finally:
            if sock is not None:
                sock.close()

    async def connected_session(self, sock) -> None:
        config = self.config
        # Keep the original WSS URL: TLS still verifies the gateway's hostname.
        transport = {"sock": sock} if sock is not None else {}
        async with DirectConnect(
            config.url,
            additional_headers={
                "Authorization": f"Bearer {config.token}",
                "X-Worker-ID": config.worker_id,
            },
            open_timeout=10,
            close_timeout=2,
            max_size=MESSAGE_LIMIT,
            max_queue=4,
            ping_interval=None,
            compression=None,
            proxy=None,
            **transport,
        ) as socket:
            await send(
                socket,
                Message(type="hello", capabilities=config.capabilities, execution_events=True),
            )
            async with asyncio.timeout(10):
                welcome = await receive(socket)
            if welcome.type != "welcome" or not welcome.session_id:
                raise ValueError("invalid server handshake")
            log.info("worker connected worker=%s session=%s", config.worker_id, welcome.session_id)
            await self.work_loop(socket, welcome.session_id, welcome.execution_events is True)

    async def work_loop(
        self, socket: ClientConnection, session: str, tracking: bool = False
    ) -> None:
        current: Task | None = None
        work: asyncio.Task | None = None
        result_sent = False
        progress = 0.0
        sequence = 0
        pending: dict[int, Ref | None] = {}
        loop = asyncio.get_running_loop()
        last_heartbeat_ack = loop.time()

        def matching(ref: Ref | None) -> bool:
            return ref is not None and current is not None and ref == task_ref(current)

        async def clear() -> None:
            nonlocal current, work, result_sent, progress
            if work is None and current is not None and current.spec.kind == "python_project":
                # Revoked before execution started: there are no project files to remove.
                self.journal.emit(
                    current.spec.id,
                    current.generation,
                    "cleaned",
                    {"workspace_removed": True, "processes_stopped": True},
                )
            if work is not None:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            current, work, result_sent = None, None, False
            progress = 0.0

        def report(value: float) -> None:
            nonlocal progress
            progress = min(100.0, max(0.0, float(value)))
            log.info("task progress %.0f%%", progress)

        async def heartbeat() -> None:
            nonlocal sequence
            sequence += 1
            ref = task_ref(current) if current else None
            pending[sequence] = ref
            await send(
                socket,
                Message(
                    type="heartbeat",
                    sequence=sequence,
                    active=[ref] if ref else [],
                    paused=self.config.paused,
                    progress=progress,
                ),
            )

        reader = asyncio.create_task(receive(socket))
        ticker = asyncio.create_task(asyncio.sleep(HEARTBEAT_INTERVAL))
        event_tick = asyncio.create_task(asyncio.sleep(0.25))
        try:
            await heartbeat()
            while True:
                waiting = {reader, ticker, event_tick}
                if work is not None and not result_sent:
                    waiting.add(work)
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

                # Handle server messages first, so revocation wins over a local
                # completion when both are ready. The database also fences it.
                if reader in done:
                    message = reader.result()
                    reader = asyncio.create_task(receive(socket))
                    match message.type:
                        case "execution_events_ack":
                            if message.execution_batch:
                                self.journal.acknowledge(message.execution_batch)
                        case "heartbeat_ack":
                            if message.sequence not in pending:
                                raise ValueError("unknown heartbeat acknowledgment")
                            sent = pending.pop(message.sequence)
                            last_heartbeat_ack = loop.time()
                            # An old idle heartbeat cannot revoke newer work.
                            if matching(sent) and sent not in message.active:
                                log.info("assignment revoked task=%s", current.spec.id)
                                await clear()
                        case "assign":
                            task = message.task
                            if current is not None or task is None:
                                raise ValueError("unexpected assignment")
                            if (
                                task.session_id != session
                                or task.worker_id != self.config.worker_id
                                or task.spec.kind
                                not in getattr(self.executor, "kinds", (self.executor.kind,))
                            ):
                                raise ValueError("invalid assignment identity or executor")
                            current = task
                            await send(socket, Message(type="ack", ref=task_ref(task)))
                        case "ack_accepted":
                            if matching(message.ref) and work is None:
                                log.info(
                                    "task started task=%s generation=%s",
                                    current.spec.id,
                                    current.generation,
                                )
                                reporter = ExecutionReporter(self.journal, current, report)
                                work = asyncio.create_task(
                                    execute(self.executor, current, reporter)
                                )
                        case "result_accepted" | "revoke":
                            if matching(message.ref):
                                log.info("%s task=%s", message.type, current.spec.id)
                                await clear()
                        case _:
                            raise ValueError("unexpected server message")

                if work is not None and work in done and not result_sent:
                    await send(socket, work.result())
                    # Do not free the local slot until the server accepts or
                    # revokes the result. Exclude completed work from wait().
                    result_sent = True

                if ticker in done:
                    if loop.time() - last_heartbeat_ack >= UNHEALTHY_AFTER:
                        raise TimeoutError("heartbeat acknowledgment timeout")
                    await heartbeat()
                    ticker = asyncio.create_task(asyncio.sleep(HEARTBEAT_INTERVAL))
                if event_tick in done:
                    if tracking and (batch := self.journal.next_batch()):
                        await send(socket, Message(type="execution_events", execution_batch=batch))
                    event_tick = asyncio.create_task(asyncio.sleep(0.25))
        finally:
            reader.cancel()
            ticker.cancel()
            event_tick.cancel()
            await clear()
            await asyncio.gather(reader, ticker, event_tick, return_exceptions=True)


async def run_worker() -> None:
    executor = StubExecutor()
    if os.getenv("WORKER_EXECUTOR") == "python_project":
        from .executors.python_project import PythonProjectExecutor

        config = WorkerConfig.from_env("python_project")
        executor = PythonProjectExecutor(config.url, config.worker_id)
    else:
        config = WorkerConfig.from_env(executor.kind)
    config.capabilities.kinds = list(getattr(executor, "kinds", (executor.kind,)))
    init_sentry("worker", worker_id=config.worker_id)

    async def run():
        if config.transport == "direct":
            await Agent(config, executor).run()
            return
        async with EmbeddedTunnel(config) as tunnel:
            async with asyncio.TaskGroup() as group:
                group.create_task(Agent(config, executor, tunnel).run())
                group.create_task(tunnel.wait())

    task = asyncio.create_task(run())
    loop = asyncio.get_running_loop()
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
            installed.append(sig)
        except NotImplementedError:
            pass  # asyncio.run still handles Ctrl-C on unsupported platforms.
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
