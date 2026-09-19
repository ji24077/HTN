"""Admin authentication and the loopback-only demo boundary."""

import hmac
import os
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from ..shared.security import authorized


def local_demo(request: Request) -> bool:
    return (
        os.getenv("DEMO_UI") == "true"
        and request.client is not None
        and request.client.host in {"127.0.0.1", "::1"}
        and request.url.hostname in {"127.0.0.1", "localhost", "::1"}
    )


async def require_admin(request: Request) -> None:
    if authorized(request.headers.get("authorization"), request.app.state.config.admin_token):
        return
    cookie = request.cookies.get("orchestrator_demo", "")
    origin = request.headers.get("origin")
    same_origin = not origin or urlsplit(origin).netloc == request.url.netloc
    if (
        local_demo(request)
        and same_origin
        and request.headers.get("sec-fetch-site") != "cross-site"
        and hmac.compare_digest(cookie, request.app.state.ui_session)
    ):
        return
    raise HTTPException(status_code=401, detail="unauthorized")
