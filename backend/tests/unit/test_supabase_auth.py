"""Real JWT signature, issuer, audience, expiry and signing-key checks."""

import json
import time
import unittest
from unittest.mock import patch

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from orchestrator.server.supabase_auth import SupabaseAuth


class SupabaseAuthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(self.key.public_key()))
        self.jwk.update(kid="test-key", alg="ES256", use="sig")
        self.auth = SupabaseAuth("https://project.supabase.co")
        await self.auth.http.aclose()
        self.requests = []

        def response(request):
            self.requests.append(request)
            return httpx.Response(200, json={"keys": [self.jwk]})

        self.auth.http = httpx.AsyncClient(transport=httpx.MockTransport(response))
        self.addAsyncCleanup(self.auth.close)

    def token(self, **changes):
        now = int(time.time())
        claims = {
            "iss": "https://project.supabase.co/auth/v1",
            "aud": "authenticated",
            "sub": "f19ed307-6827-4b38-9a22-fc291727dd97",
            "role": "authenticated",
            "iat": now,
            "exp": now + 3600,
        }
        claims.update(changes)
        return jwt.encode(claims, self.key, algorithm="ES256", headers={"kid": "test-key"})

    async def test_valid_token_and_cached_public_keys(self):
        self.assertEqual((await self.auth.verify(self.token()))["role"], "authenticated")
        await self.auth.verify(self.token())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            str(self.requests[0].url), "https://project.supabase.co/auth/v1/.well-known/jwks.json"
        )
        self.assertNotIn("authorization", self.requests[0].headers)

    async def test_invalid_claims_are_rejected(self):
        for changes in (
            {"iss": "https://other.supabase.co/auth/v1"},
            {"aud": "other"},
            {"exp": int(time.time()) - 1},
            {"role": "service_role"},
            {"sub": "not-a-uuid"},
            {"is_anonymous": True},
            {"iat": int(time.time()) + 3600},
        ):
            with self.subTest(changes=changes), self.assertRaises(HTTPException) as error:
                await self.auth.verify(self.token(**changes))
            self.assertEqual(error.exception.status_code, 401)

    async def test_forged_signature_and_symmetric_tokens_are_rejected(self):
        other_key = ec.generate_private_key(ec.SECP256R1())
        forged = jwt.encode(
            {"sub": "admin"}, other_key, algorithm="ES256", headers={"kid": "test-key"}
        )
        symmetric = jwt.encode(
            {"sub": "admin"}, "test-symmetric-key-not-for-use-12345", algorithm="HS256"
        )
        for token in (forged, symmetric, "not-a-jwt"):
            with self.assertRaises(HTTPException) as error:
                await self.auth.verify(token)
            self.assertEqual(error.exception.status_code, 401)

    async def test_key_rotation_after_cache_expiry(self):
        await self.auth.verify(self.token())
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(self.key.public_key()))
        self.jwk.update(kid="test-key", alg="ES256", use="sig")
        with patch(
            "orchestrator.server.supabase_auth.time.monotonic",
            return_value=self.auth.checked_at + 61,
        ):
            await self.auth.verify(self.token())
        self.assertEqual(len(self.requests), 2)

    async def test_jwks_outage_fails_closed(self):
        await self.auth.http.aclose()
        self.auth.http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(503))
        )
        with self.assertRaises(HTTPException) as error:
            await self.auth.verify(self.token())
        self.assertEqual(error.exception.status_code, 503)
