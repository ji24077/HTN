"""Read-only Sentry API adapter. Queries and returned rows are job-scoped."""

import asyncio
import json
import os
import re
from datetime import timedelta
from urllib.parse import urlsplit

import httpx

from ..shared.execution import scrub_execution


class SentryUnavailable(Exception):
    pass


class SentryReader:
    def __init__(
        self, token, organization, project, *, base_url="https://sentry.io", transport=None
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("SENTRY_API_URL must be an HTTPS origin")
        if not re.fullmatch(r"[\w-]+", organization) or not re.fullmatch(r"[\w-]+", project):
            raise ValueError("Sentry organization and project must be IDs or slugs")
        self.organization, self.project = organization, project
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    @classmethod
    def from_env(cls):
        values = [os.getenv(k, "") for k in ("SENTRY_API_TOKEN", "SENTRY_ORG", "SENTRY_PROJECT")]
        if not all(values):
            return None
        return cls(*values, base_url=os.getenv("SENTRY_API_URL", "https://sentry.io"))

    async def close(self):
        await self.http.aclose()

    async def search(self, job_id, search, *, dataset="logs", event_id=None):
        fields = (
            [
                "id",
                "timestamp",
                "message",
                "severity",
                "trace",
                "job_id",
                "task_id",
                "worker_id",
                "attempt",
                "reservation_id",
            ]
            if dataset == "logs"
            else ["id", "timestamp", "title", "issue", "job_id", "task_id", "worker_id"]
        )
        # No raw Sentry syntax is accepted from the agent. Build an AND of literals.
        filters = {"job_id": job_id, "task_id": search.task_id, "worker_id": search.worker_id}
        if dataset == "logs":
            filters.update(
                attempt=search.attempt,
                severity=search.severity,
                reservation_id=search.reservation_id,
                trace=search.trace_id,
            )
        if event_id:
            filters["id"] = event_id
        query = " AND ".join(
            f"{key}:{json.dumps(str(value))}" for key, value in filters.items() if value is not None
        )
        if search.text:
            field = "message" if dataset == "logs" else "title"
            query += f" AND {field}:{json.dumps('*' + search.text + '*')}"
        params = [
            ("dataset", dataset),
            ("project", self.project),
            ("query", query),
            ("per_page", str(search.limit)),
            ("sort", "timestamp"),
        ]
        params.extend(("field", field) for field in fields)
        if search.start:
            params.append(("start", search.start.isoformat()))
        elif search.end:
            params.append(("start", (search.end - timedelta(hours=24)).isoformat()))
        else:
            params.append(("statsPeriod", "24h"))
        if search.end:
            params.append(("end", search.end.isoformat()))
        if search.cursor:
            params.append(("cursor", search.cursor))
        try:
            async with (
                asyncio.timeout(15),
                self.http.stream(
                    "GET", f"/api/0/organizations/{self.organization}/events/", params=params
                ) as response,
            ):
                if not response.is_success:
                    raise SentryUnavailable(f"Sentry API returned HTTP {response.status_code}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 512 * 1024:
                        raise SentryUnavailable("Sentry response exceeded limit")
                payload = json.loads(body)
                rows = payload["data"]
                if not isinstance(rows, list) or len(rows) > search.limit:
                    raise ValueError("invalid rows")
                # Fail closed even if provider query semantics change.
                if any(not isinstance(row, dict) or row.get("job_id") != job_id for row in rows):
                    raise SentryUnavailable("Sentry returned records outside the job scope")
                next_link = response.links.get("next", {})
                cursor = next_link.get("cursor") if next_link.get("results") == "true" else None
                return scrub_execution(
                    {"events": rows, "next_cursor": cursor, "has_more": bool(cursor)}
                )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, TimeoutError) as exc:
            raise SentryUnavailable("Sentry search unavailable or returned invalid data") from exc
