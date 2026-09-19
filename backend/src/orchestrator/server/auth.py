"""Admin authentication and the loopback-only demo boundary."""

import hmac
import os
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import HTTPException, Request

from ..shared.security import authorized

SESSION_COOKIE = "__Host-orchestrator_session"


def public_dashboard(request: Request) -> bool:
    origin = getattr(request.app.state.config, "public_origin", "")
    return bool(origin) and request.url.netloc == urlsplit(origin).netloc


def same_origin(request: Request, *, mutation: bool = False) -> bool:
    expected = (
        request.app.state.config.public_origin
        if public_dashboard(request)
        else f"{request.url.scheme}://{request.url.netloc}"
    )
    origin = request.headers.get("origin")
    return request.headers.get("sec-fetch-site") != "cross-site" and (
        origin == expected or (not origin and not mutation)
    )


def local_demo(request: Request) -> bool:
    return (
        getattr(request.app.state, "surface", "combined") == "combined"
        and os.getenv("DEMO_UI") == "true"
        and request.client is not None
        and request.client.host in {"127.0.0.1", "::1"}
        and request.url.hostname in {"127.0.0.1", "localhost", "::1"}
    )


async def require_fleet_access(request: Request, claims: dict) -> None:
    config = request.app.state.config
    if claims["sub"] in config.supabase_admin_ids:
        return
    emails = getattr(config, "supabase_admin_emails", frozenset())
    if emails:
        # Use Supabase's confirmed account record, never editable user metadata
        # or an unverified email supplied by the browser.
        approved = await request.app.state.store.pool.fetchval(
            """SELECT EXISTS (
                SELECT 1 FROM auth.users
                WHERE id = $1 AND lower(email) = ANY($2::text[])
                  AND email_confirmed_at IS NOT NULL AND deleted_at IS NULL
            )""",
            UUID(claims["sub"]),
            list(emails),
        )
        if approved:
            return
    raise HTTPException(status_code=403, detail="This account does not have fleet access")


async def require_admin(request: Request) -> None:
    if authorized(request.headers.get("authorization"), request.app.state.config.admin_token):
        return
    cookie = request.cookies.get("orchestrator_demo", "")
    if (
        local_demo(request)
        and same_origin(request)
        and hmac.compare_digest(cookie, request.app.state.ui_session)
    ):
        return
    token = None
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        token = header[7:]
    elif public_dashboard(request) and same_origin(
        request, mutation=request.method not in {"GET", "HEAD", "OPTIONS"}
    ):
        token = request.cookies.get(SESSION_COOKIE)
    verifier = getattr(request.app.state, "supabase_auth", None)
    if token and verifier:
        claims = await verifier.verify(token)
        await require_fleet_access(request, claims)
        request.state.auth_claims = claims
        return
    raise HTTPException(status_code=401, detail="unauthorized")
