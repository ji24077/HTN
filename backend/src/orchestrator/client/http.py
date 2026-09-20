"""Authenticated async client for planners, scripts, and agent tool adapters."""

import asyncio
import ipaddress
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
from pydantic import TypeAdapter

from ..shared.protocol import Identifier, Submission, Task, TaskSpec, Worker, json_loads

TERMINAL = {"succeeded", "failed", "cancelled"}


class ClientError(Exception):
    def __init__(
        self, code: str, message: str, *, status: int | None = None, retryable: bool = False
    ):
        super().__init__(message)
        self.code, self.status, self.retryable = code, status, retryable

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "status": self.status,
            "retryable": self.retryable,
        }


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("URL must have a host and no credentials, query, or fragment")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise ValueError("Use HTTPS, except for HTTP on loopback")
    return url.rstrip("/")


def identifier(value: str) -> str:
    return TypeAdapter(Identifier).validate_python(value)


class OrchestratorClient:
    def __init__(self, url: str, token: str):
        url = validate_url(url)
        if not token or any(c.isspace() for c in token):
            raise ValueError("A nonempty admin token without whitespace is required")
        self.http = httpx.AsyncClient(
            base_url=url + "/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(30, connect=5),
            follow_redirects=False,
            trust_env=False,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    @staticmethod
    async def _check(response: httpx.Response) -> None:
        if response.is_success:
            return
        await response.aread()
        try:
            data = response.json()
            message = data.get("error") or data.get("detail")
        except (ValueError, AttributeError):
            message = None
        codes = {
            400: "invalid_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            413: "too_large",
            429: "rate_limited",
        }
        status = response.status_code
        raise ClientError(
            codes.get(status, "http_error"),
            str(message or f"HTTP {status}"),
            status=status,
            retryable=status == 429 or status >= 500,
        )

    async def _request(self, method: str, path: str, **kwargs):
        try:
            response = await self.http.request(method, path, **kwargs)
            await self._check(response)
            return json_loads(response.content)
        except httpx.TransportError as exc:
            # A timed-out POST may have committed. Caller must reuse task IDs;
            # never silently retry a mutation with newly generated IDs.
            raise ClientError(
                "connection_error",
                "Control plane request failed; outcome may be unknown",
                retryable=True,
            ) from exc

    async def list_workers(self) -> list[Worker]:
        return TypeAdapter(list[Worker]).validate_python(await self._request("GET", "v1/workers"))

    async def list_tasks(self) -> list[Task]:
        return TypeAdapter(list[Task]).validate_python(await self._request("GET", "v1/tasks"))

    async def get_task(self, task_id: str) -> Task:
        return Task.model_validate(await self._request("GET", f"v1/tasks/{identifier(task_id)}"))

    async def submit_tasks(self, tasks: list[TaskSpec]) -> list[Task]:
        body = Submission(tasks=tasks).model_dump(mode="json")
        return TypeAdapter(list[Task]).validate_python(
            await self._request("POST", "v1/tasks", json=body)
        )

    async def cancel_task(self, task_id: str) -> Task:
        return Task.model_validate(
            await self._request("POST", f"v1/tasks/{identifier(task_id)}/cancel")
        )

    async def list_events(self, after: int = 0) -> list[dict]:
        if isinstance(after, bool) or not isinstance(after, int) or not 0 <= after <= 2**63 - 1:
            raise ValueError("after must be an integer from 0 to 2**63-1")
        return await self._request("GET", "v1/events", params={"after": after})

    async def updates(self) -> AsyncIterator[dict]:
        """Receive current snapshots over SSE; reconnect/resync transient failures."""
        while True:
            try:
                async with self.http.stream("GET", "v1/updates") as response:
                    await self._check(response)
                    event, data = "", []
                    async for line in response.aiter_lines():
                        if not line:
                            if event == "snapshot" and data:
                                yield json_loads("\n".join(data))
                            elif event == "unavailable":
                                break
                            event, data = "", []
                        elif line.startswith("event:"):
                            event = line[6:].lstrip(" ")
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip(" "))
            except httpx.TransportError:
                pass
            except ClientError as exc:
                if not exc.retryable:
                    raise
            await asyncio.sleep(2)

    async def wait_task(self, task_id: str, timeout_seconds: float = 300) -> Task:
        """Wait for a terminal state using pushed updates; timeout does not cancel work."""
        task_id = identifier(task_id)
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be greater than 0 and at most 3600")
        try:
            async with asyncio.timeout(timeout_seconds):
                task = await self.get_task(task_id)
                if task.state in TERMINAL:
                    return task
                stream = self.updates()
                try:
                    async for snapshot in stream:
                        row = next(
                            (t for t in snapshot["tasks"] if t["spec"]["id"] == task_id), None
                        )
                        # The fleet snapshot contains only the newest 500 tasks.
                        # Older tasks are read on change notifications, never a timer.
                        if row is None:
                            task = await self.get_task(task_id)
                        else:
                            task = Task.model_validate({k: v for k, v in row.items() if k != "id"})
                        if task.state in TERMINAL:
                            return task
                finally:
                    await stream.aclose()
        except TimeoutError as exc:
            raise ClientError(
                "wait_timeout", "Wait timed out; the task was not cancelled", retryable=True
            ) from exc
