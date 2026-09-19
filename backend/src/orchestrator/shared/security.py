"""Credential checks shared by server and worker configuration."""

import hmac


def authorized(header: str | None, token: str | None) -> bool:
    return bool(token) and hmac.compare_digest((header or "").encode(), f"Bearer {token}".encode())


def credential(value: str) -> str:
    if len(value) < 24 or value.startswith("replace-with-") or any(c.isspace() for c in value):
        raise ValueError("tokens must be random values of at least 24 non-whitespace characters")
    return value
