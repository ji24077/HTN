"""Jack's device identity and signed-result wire format, without a second scheduler."""

import base64
import hashlib
import json
import math
import re
import time
from datetime import datetime
from uuid import UUID

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .protocol import MESSAGE_LIMIT, bounded_json, json_text, reject_constant

ASSERTION_TTL = 120
CLOCK_SKEW = 120


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


_DECODER = json.JSONDecoder(
    object_pairs_hook=_object, parse_constant=reject_constant, parse_float=_finite
)


def _json(value: str) -> object:
    try:
        return _DECODER.decode(value)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid JSON") from exc


def _member(value: str, name: str) -> str | None:
    """Locate a member's exact JSON bytes in an already validated object."""
    offset = len(value) - len(value.lstrip()) + 1
    while True:
        while value[offset].isspace():
            offset += 1
        if value[offset] == "}":
            return None
        key, offset = _DECODER.raw_decode(value, offset)
        while value[offset].isspace():
            offset += 1
        offset += 1  # colon; the complete object has already been validated
        while value[offset].isspace():
            offset += 1
        start = offset
        _, offset = _DECODER.raw_decode(value, offset)
        if key == name:
            return value[start:offset]
        while value[offset].isspace():
            offset += 1
        if value[offset] == "}":
            return None
        offset += 1


def parse_frame(raw: str | bytes) -> tuple[dict, str | None]:
    """Reject ambiguous JSON and retain the bytes hashed by JS and Swift agents.

    Python reserialization changes floating-point notation and Unicode escaping.
    Hashing the original output member keeps signatures portable across runtimes.
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("invalid frame encoding") from exc
    if not isinstance(raw, str):
        raise ValueError("frame must be JSON text")
    try:
        if len(raw.encode("utf-8")) > MESSAGE_LIMIT:
            raise ValueError("frame too large")
        frame = _json(raw)
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        output = None
        if isinstance(frame.get("payload"), dict):
            payload = _member(raw, "payload")
            output = _member(payload, "output")
        return frame, output
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid JSON frame") from exc


def _b64url(value: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64url")
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except ValueError as exc:
        raise ValueError("invalid base64url") from exc
    if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != value:
        raise ValueError("noncanonical base64url")
    return decoded


def _public_key(value: str) -> Ed25519PublicKey:
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("invalid device public key")
    try:
        key = serialization.load_der_public_key(base64.b64decode(value, validate=True))
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        raise ValueError("invalid device public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("device key must be Ed25519")
    return key


def validate_public_key(value: str) -> str:
    key = _public_key(value)
    return base64.b64encode(
        key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    ).decode()


def _uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid device identifier")
    try:
        normalized = str(UUID(value))
    except ValueError as exc:
        raise ValueError("invalid device identifier") from exc
    if value != normalized:
        raise ValueError("device identifier must be a canonical UUID")
    return normalized


def _assertion(token: str) -> tuple[str, dict, bytes]:
    if not isinstance(token, str) or len(token) > 4096:
        raise ValueError("invalid assertion")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid assertion")
    try:
        header = _json(_b64url(parts[0]).decode("utf-8"))
        claims = _json(_b64url(parts[1]).decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("invalid assertion") from exc
    if not isinstance(header, dict) or header != {"alg": "EdDSA", "typ": "JWT"}:
        raise ValueError("unsupported assertion header")
    if not isinstance(claims, dict):
        raise ValueError("invalid assertion claims")
    _uuid(claims.get("iss"))
    return ".".join(parts[:2]), claims, _b64url(parts[2])


def assertion_identity(token: str) -> str:
    """Untrusted lookup key only; verify_assertion must run before authorization."""
    return _assertion(token)[1]["iss"]


def verify_assertion(
    token: str, public_key: str, *, audience: str = "dwp-control", now: float | None = None
) -> dict:
    signed, claims, signature = _assertion(token)
    now = time.time() if now is None else now
    if claims.get("aud") != audience:
        raise ValueError("invalid assertion audience")
    _uuid(claims.get("jti"))
    issued, expires = claims.get("iat"), claims.get("exp")
    if any(type(value) is not int or not 0 < value < 2**53 for value in (issued, expires)):
        raise ValueError("invalid assertion timestamps")
    if not 0 < expires - issued <= ASSERTION_TTL:
        raise ValueError("invalid assertion lifetime")
    if expires <= now or issued > now + CLOCK_SKEW:
        raise ValueError("assertion expired or issued in the future")
    try:
        _public_key(public_key).verify(signature, signed.encode("ascii"))
    except InvalidSignature as exc:
        raise ValueError("invalid assertion signature") from exc
    return claims


def verify_result(
    payload: dict,
    raw_output: str | None,
    public_key: str,
    *,
    task_id: str,
    attempt: int,
    worker_id: str,
) -> dict:
    """Verify both output bytes and the identity/generation-bound signature."""
    if (
        payload.get("taskId") != task_id
        or type(payload.get("attempt")) is not int
        or payload["attempt"] != attempt
        or attempt < 1
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", task_id)
    ):
        raise ValueError("result does not match assignment")
    _uuid(worker_id)
    if raw_output is None or "output" not in payload:
        raise ValueError("result output is missing")
    bounded_json(payload["output"])
    if json_text(_json(raw_output)) != json_text(payload["output"]):
        raise ValueError("result output differs from signed bytes")
    output_hash = payload.get("outputHash")
    if (
        not isinstance(output_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", output_hash)
        or hashlib.sha256(raw_output.encode("utf-8")).hexdigest() != output_hash
    ):
        raise ValueError("result output hash mismatch")
    dates = []
    for field in ("startedAt", "finishedAt"):
        value = payload.get(field)
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value
        ):
            raise ValueError("invalid result timestamp")
        dates.append(datetime.fromisoformat(value))
    if dates[1] < dates[0]:
        raise ValueError("result finished before it started")
    canonical_key = validate_public_key(public_key)
    signed = (
        f"dwp-attest/v1\n{task_id}\n{attempt}\n{worker_id}\n{output_hash}\n"
        f"{payload['startedAt']}\n{payload['finishedAt']}\n"
    )
    signature = _b64url(payload.get("signature"))
    try:
        _public_key(canonical_key).verify(signature, signed.encode("utf-8"))
    except InvalidSignature as exc:
        raise ValueError("invalid result signature") from exc
    return {
        "version": "dwp-attest/v1",
        "taskId": task_id,
        "attempt": attempt,
        "hostId": worker_id,
        "outputHash": output_hash,
        "startedAt": payload["startedAt"],
        "finishedAt": payload["finishedAt"],
        "signature": payload["signature"],
        "publicKey": canonical_key,
        "rawOutput": raw_output,
    }
