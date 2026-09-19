"""Dashboard assets and authenticated browser sessions."""

import re
import time
from importlib.resources import files

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .auth import (
    SESSION_COOKIE,
    local_demo,
    public_dashboard,
    require_admin,
    same_origin,
)

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
    if not same_origin(request):
        raise HTTPException(status_code=403)
    response = JSONResponse({"status": "ok", "mode": "demo"})
    session_cookie(response, request)
    return response


@router.get("/auth/session")
async def browser_session(request: Request):
    if local_demo(request):
        return await session(request)
    if not public_dashboard(request):
        raise HTTPException(status_code=404)
    await require_admin(request)
    return JSONResponse({"status": "ok", "mode": "public"}, headers={"Cache-Control": "no-store"})


@router.get("/auth/config")
async def auth_config(request: Request):
    if not public_dashboard(request):
        raise HTTPException(status_code=404)
    config = request.app.state.config
    return JSONResponse(
        {"url": config.supabase_url, "publishableKey": config.supabase_publishable_key},
        headers={"Cache-Control": "no-store"},
    )


@router.post("/auth/session")
async def login(request: Request):
    """Use a verified Supabase access token for native EventSource cookie auth."""
    if not public_dashboard(request):
        raise HTTPException(status_code=404)
    if not same_origin(request, mutation=True):
        raise HTTPException(status_code=403, detail="origin not allowed")
    # Do not exchange the automation admin key for a browser session.
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Supabase access token required")
    claims = await request.app.state.supabase_auth.verify(header[7:])
    if claims["sub"] not in request.app.state.config.supabase_admin_ids:
        raise HTTPException(status_code=403, detail="This account does not have fleet access")
    response = JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})
    response.set_cookie(
        SESSION_COOKIE,
        header[7:],
        secure=True,
        httponly=True,
        samesite="strict",
        max_age=max(0, int(claims["exp"] - time.time())),
        path="/",
    )
    return response


@router.delete("/auth/session")
async def logout(request: Request):
    if not public_dashboard(request):
        raise HTTPException(status_code=404)
    if not same_origin(request, mutation=True):
        raise HTTPException(status_code=403, detail="origin not allowed")
    response = JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})
    response.delete_cookie(SESSION_COOKIE, secure=True, httponly=True, samesite="strict")
    return response


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not (local_demo(request) or public_dashboard(request)):
        raise HTTPException(status_code=404)
    try:
        html = files("orchestrator.server").joinpath("web", "index.html").read_text()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Dashboard build unavailable") from exc
    response = HTMLResponse(html)
    response.headers["Cache-Control"] = "no-store"
    if local_demo(request):
        session_cookie(response, request)
    return response


@router.get("/assets/{filename}")
async def asset(filename: str, request: Request):
    if not public_dashboard(request):
        await require_admin(request)
    if not (local_demo(request) or public_dashboard(request)) or not re.fullmatch(
        r"[A-Za-z0-9_-]+\.(js|css)", filename
    ):
        raise HTTPException(status_code=404)
    try:
        data = files("orchestrator.server").joinpath("web", "assets", filename).read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404) from exc
    content_type = "application/javascript" if filename.endswith(".js") else "text/css"
    return Response(data, media_type=content_type, headers={"Cache-Control": "no-store"})
