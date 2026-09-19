"""Serve packaged dashboard assets under the local-demo boundary."""

import re
from importlib.resources import files
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .auth import local_demo, require_admin

router = APIRouter()


def session_cookie(response: Response, request: Request) -> None:
    response.set_cookie(
        "orchestrator_demo",
        request.app.state.ui_session,
        httponly=True,
        samesite="strict",
        max_age=86400,
    )
    response.headers["Cache-Control"] = "no-store"


@router.get("/demo/session")
async def session(request: Request):
    """Bootstrap the local React app, including through the Vite dev proxy."""
    if not local_demo(request):
        raise HTTPException(status_code=404)
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site" or (
        origin and urlsplit(origin).netloc != request.url.netloc
    ):
        raise HTTPException(status_code=403)
    response = JSONResponse({"status": "ok"})
    session_cookie(response, request)
    return response


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not local_demo(request):
        raise HTTPException(status_code=404)
    try:
        html = files("orchestrator.server").joinpath("web", "index.html").read_text()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Dashboard build unavailable") from exc
    response = HTMLResponse(html)
    session_cookie(response, request)
    return response


@router.get("/assets/{filename}", dependencies=[Depends(require_admin)])
async def asset(filename: str, request: Request):
    if not local_demo(request) or not re.fullmatch(r"[A-Za-z0-9_-]+\.(js|css)", filename):
        raise HTTPException(status_code=404)
    try:
        data = files("orchestrator.server").joinpath("web", "assets", filename).read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404) from exc
    content_type = "application/javascript" if filename.endswith(".js") else "text/css"
    return Response(data, media_type=content_type, headers={"Cache-Control": "no-store"})
