"""HTTP client for the gpushare service.

gpushare runs as its own process: it holds SSH forwards to rented RunPod
machines and keeps a model resident on one of them, so it cannot be a library
call from this server. The dashboard only needs to *read* that state, so this
client speaks a fixed set of paths rather than forwarding whatever it is given
— an open proxy reachable from the dashboard would let any authenticated user
aim this server at an arbitrary host.
"""

import httpx

# Every path this client will ever request. gpushare's write endpoints
# (train, serve, migrate) are deliberately absent: those actions run from
# gpushare's own console, where their long-running job output already has a
# place to go.
READ_PATHS = {
    "state": "/api/state",
    "pods": "/api/pods",
    "jobs": "/api/jobs",
    "models": "/api/models",
}


class GpushareUnavailable(Exception):
    """gpushare is configured but did not answer."""


class GpushareClient:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        # The read budget is wider than a localhost hop needs because
        # /api/pods is a live RunPod API call the first time it is asked and
        # cached afterwards; five seconds timed that cold call out. Connect
        # stays short — if gpushare is not listening, say so immediately
        # rather than holding a dashboard poll open.
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(10, connect=2),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    async def read(self, name: str) -> dict:
        try:
            path = READ_PATHS[name]
        except KeyError:
            raise ValueError(f"unknown gpushare resource {name!r}") from None
        try:
            response = await self._http.get(path)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GpushareUnavailable(str(exc)) from exc
        # gpushare answers objects everywhere; a list would mean the path
        # changed under us, and passing it through would break the panel with
        # a type error far from here.
        if not isinstance(payload, dict):
            raise GpushareUnavailable(f"{path} did not return an object")
        return payload

    async def aclose(self) -> None:
        await self._http.aclose()
