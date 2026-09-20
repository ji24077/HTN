from fastapi import APIRouter, Depends, HTTPException, Request

from ..server.auth import require_admin
from .client import GpushareUnavailable

router = APIRouter(prefix="/v1/gpushare", dependencies=[Depends(require_admin)])


def _client(request: Request):
    client = getattr(request.app.state, "gpushare", None)
    if client is None:
        raise HTTPException(503, "gpushare is not configured")
    return client


async def _read(request: Request, name: str) -> dict:
    try:
        return await _client(request).read(name)
    except GpushareUnavailable as exc:
        # 502, not 503: this server is fine and configured, the service it was
        # pointed at is the one that did not answer. The panel says which.
        raise HTTPException(502, f"gpushare did not answer: {exc}") from exc


@router.get("/config")
async def config(request: Request):
    """Whether the panel should render at all, without waiting on a round trip."""
    return {"enabled": getattr(request.app.state, "gpushare", None) is not None}


@router.get("/state")
async def state(request: Request):
    return await _read(request, "state")


@router.get("/pods")
async def pods(request: Request):
    return await _read(request, "pods")


@router.get("/jobs")
async def jobs(request: Request):
    return await _read(request, "jobs")


@router.get("/models")
async def models(request: Request):
    return await _read(request, "models")
