"""Approved-user enrollment: single-use tailnet keys and hashed worker credentials."""

import asyncio
import logging
import re
import secrets
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .auth import require_admin
from .db.store import PENDING_ENROLLMENT_SECONDS, EnrollmentLimit

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/worker-enrollments", dependencies=[Depends(require_admin)])
WORKER_ID = re.compile(r"worker-[0-9a-f]{20}")
# Key issuance and activation must finish before the reconciler could expire the reservation.
PROVISIONING_SECONDS = 60
assert PROVISIONING_SECONDS < PENDING_ENROLLMENT_SECONDS
NOT_CONFIGURED = {
    "code": "enrollment_not_configured",
    "message": "Automatic worker enrollment is not configured",
}


class EnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    name: str = Field(min_length=1, max_length=80, pattern=r"^[^\x00-\x1f\x7f]+$")


class EnrollmentUnavailable(Exception):
    pass


class TailscaleEnrollment:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url="https://api.tailscale.com/api/v2/",
            timeout=20,
            follow_redirects=False,
            transport=transport,
        )

    async def close(self):
        await self.client.aclose()

    async def access_token(self):
        response = await self.client.post(
            "oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.config.tailscale_oauth_client_id,
                "client_secret": self.config.tailscale_oauth_client_secret,
                "scope": "auth_keys",
                "tags": self.config.tailscale_enrollment_tag,
            },
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        if not isinstance(token, str) or not token:
            raise ValueError("missing access token")
        return token

    async def create(self):
        try:
            token = await self.access_token()
            response = await self.client.post(
                "tailnet/-/keys",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "capabilities": {
                        "devices": {
                            "create": {
                                "reusable": False,
                                "ephemeral": False,
                                "preauthorized": True,
                                "tags": [self.config.tailscale_enrollment_tag],
                            }
                        }
                    },
                    "expirySeconds": 600,
                },
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data.get("id"), str) or not data["id"]:
                raise ValueError("missing key ID")
            if not isinstance(data.get("key"), str) or not data["key"].startswith("tskey-auth-"):
                raise ValueError("invalid enrollment key")
            return data["id"], data["key"], token
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # API errors can echo credentials; never expose their body or exception text.
            raise EnrollmentUnavailable from None

    async def revoke(self, key_id, token=None) -> bool:
        """Delete an unused key, reusing the hour-long access token that created it."""
        try:
            token = token or await self.access_token()
            response = await self.client.delete(
                "tailnet/-/keys/" + quote(key_id, safe=""),
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return True
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            log.warning("Failed to revoke unused Tailscale auth key; it expires within ten minutes")
            return False


@router.post("", status_code=201)
async def enroll(request: Request):
    claims = getattr(request.state, "auth_claims", None)
    if not claims:
        raise HTTPException(status_code=403, detail="Sign in with an approved Supabase account")
    issuer = getattr(request.app.state, "enrollment", None)
    if issuer is None:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    body = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 4096:
                    raise HTTPException(status_code=413, detail="Enrollment request too large")
        data = EnrollmentRequest.model_validate_json(body)
    except (ValidationError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid enrollment request") from None
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Enrollment request timed out") from None
    worker_id, token = "worker-" + uuid4().hex[:20], secrets.token_urlsafe(32)
    store = request.app.state.store
    try:
        await store.reserve_enrollment(
            worker_id, str(data.request_id), claims["sub"], data.name, token
        )
    except EnrollmentLimit:
        raise HTTPException(status_code=429, detail="Worker enrollment limit reached") from None
    key_id = oauth_token = None
    try:
        async with asyncio.timeout(PROVISIONING_SECONDS):
            key_id, auth_key, oauth_token = await issuer.create()
            if not await store.finish_enrollment(worker_id, key_id):
                # The reconciler expired the reservation first; never hand out its key.
                raise EnrollmentUnavailable
    except BaseException:
        revoked = await issuer.revoke(key_id, oauth_token) if key_id else True
        try:
            # A key that survived revocation stays on the record so withdrawal can retry it.
            await store.finish_enrollment(worker_id, None, retain_key=None if revoked else key_id)
        except Exception:
            log.warning("could not close enrollment %s; the reconciler expires it", worker_id)
        # Preserve cancellation; ordinary provisioning failures have a fixed public message.
        if asyncio.current_task().cancelling():
            raise
        raise HTTPException(
            status_code=502, detail="Worker enrollment failed; try again later"
        ) from None
    return JSONResponse(
        {
            "worker_id": worker_id,
            "worker_token": token,
            "server_url": request.app.state.config.worker_gateway_url,
            "tailscale_auth_key": auth_key,
            "tailscale_hostname": "orch-" + worker_id,
            "auth_key_expires_in": 600,
        },
        status_code=201,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.delete("/{worker_id}", status_code=204)
async def withdraw(worker_id: str, request: Request):
    """Release an enrollment whose computer never joined, so it stops counting against the cap."""
    claims = getattr(request.state, "auth_claims", None)
    if not claims:
        raise HTTPException(status_code=403, detail="Sign in with an approved Supabase account")
    if not WORKER_ID.fullmatch(worker_id):
        raise HTTPException(status_code=404, detail="Enrollment not found")
    # Ownership, idempotency, and the never-connected rule are enforced under the store lock;
    # NotFound and Conflict map to 404 and 409 through the application handlers.
    store = request.app.state.store
    key_id = await store.cancel_enrollment(worker_id, claims["sub"])
    issuer = getattr(request.app.state, "enrollment", None)
    # Outside the transaction: a failed revoke keeps the key ID on the row, and repeating
    # the request retries it.
    if key_id and issuer is not None and await issuer.revoke(key_id):
        await store.clear_enrollment_key(worker_id)
    return Response(status_code=204)
