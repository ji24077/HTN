"""Server environment configuration; importing opens no connections."""

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import TypeAdapter

from ..shared.protocol import Identifier
from ..shared.security import credential


@dataclass(frozen=True)
class ServerConfig:
    database_url: str
    redis_url: str | None
    admin_token: str
    worker_tokens: dict[str, str]
    public_origin: str = ""
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_admin_ids: frozenset[str] = frozenset()
    database_schema: str = "public"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        database_url = os.environ["DATABASE_URL"]
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        admin = os.getenv("ADMIN_TOKEN", "")
        if admin:
            credential(admin)
        tokens = TypeAdapter(dict[Identifier, str]).validate_python(
            json.loads(os.environ["WORKER_TOKENS"]), strict=True
        )
        if not tokens:
            raise ValueError("WORKER_TOKENS must enroll at least one worker")
        for token in tokens.values():
            credential(token)
        if admin in tokens.values() or len(set(tokens.values())) != len(tokens):
            raise ValueError("admin and each worker must have distinct tokens")
        origin = os.getenv("PUBLIC_ORIGIN", "").rstrip("/")
        if origin:
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "PUBLIC_ORIGIN must be an HTTPS origin, e.g. https://app.example.com"
                )
            _ = parsed.port  # Reject malformed ports at startup.
        supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        if supabase_url:
            parsed = urlsplit(supabase_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("SUPABASE_URL must be an HTTPS project origin")
        publishable = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
        if publishable and not publishable.startswith("sb_publishable_"):
            raise ValueError("Use a Supabase publishable key, never a secret or service-role key")
        admins = frozenset(
            str(UUID(value.strip()))
            for value in os.getenv("SUPABASE_ADMIN_IDS", "").split(",")
            if value.strip()
        )
        schema = os.getenv("DATABASE_SCHEMA", "orchestrator" if supabase_url else "public")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("DATABASE_SCHEMA must be a lowercase SQL identifier")
        if supabase_url and schema in {"public", "auth", "storage", "realtime", "extensions"}:
            raise ValueError("Use a private DATABASE_SCHEMA such as orchestrator")
        if supabase_url:
            database = urlsplit(database_url)
            if database.port == 6543:
                raise ValueError(
                    "Use a direct or session-mode database connection (port 5432), not transaction pooling"
                )
            if "sslmode=verify-full" not in database.query.split("&"):
                raise ValueError("Supabase DATABASE_URL must include sslmode=verify-full")
        return cls(
            database_url,
            os.getenv("REDIS_URL") or None,
            admin,
            tokens,
            origin,
            supabase_url,
            publishable,
            admins,
            schema,
        )
