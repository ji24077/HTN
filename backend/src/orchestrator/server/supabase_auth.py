"""Verify Supabase user JWTs against the project's public signing keys."""

import asyncio
import time
from uuid import UUID

import httpx
import jwt
from fastapi import HTTPException


class SupabaseAuth:
    def __init__(self, url: str):
        self.issuer = url.rstrip("/") + "/auth/v1"
        self.http = httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False)
        self.keys: dict[str, jwt.PyJWK] = {}
        self.checked_at = float("-inf")
        self.lock = asyncio.Lock()

    async def close(self) -> None:
        await self.http.aclose()

    async def verify(self, token: str) -> dict:
        try:
            if len(token) > 16384:
                raise ValueError("Token too large")
            header = jwt.get_unverified_header(token)
            if header.get("alg") not in {"ES256", "RS256"} or not isinstance(
                header.get("kid"), str
            ):
                raise ValueError("Unsupported signing key")
            async with self.lock:
                # Bound refreshes, including requests with unknown key IDs.
                if time.monotonic() - self.checked_at >= 60:
                    try:
                        response = await self.http.get(self.issuer + "/.well-known/jwks.json")
                        response.raise_for_status()
                        keys = jwt.PyJWKSet.from_dict(response.json()).keys
                        self.keys = {key.key_id: key for key in keys}
                        self.checked_at = time.monotonic()
                    except (httpx.HTTPError, ValueError, jwt.PyJWTError) as exc:
                        raise HTTPException(
                            status_code=503, detail="Sign-in verification unavailable"
                        ) from exc
            key = self.keys.get(header["kid"])
            if key is None or key.algorithm_name != header["alg"]:
                raise ValueError("Unknown signing key")
            claims = jwt.decode(
                token,
                key.key,
                algorithms=[key.algorithm_name],
                issuer=self.issuer,
                audience="authenticated",
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
            UUID(claims["sub"])
            if claims.get("role") != "authenticated" or claims.get("is_anonymous", False):
                raise ValueError("A signed-in user is required")
            return claims
        except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=401, detail="Invalid or expired sign-in") from exc
