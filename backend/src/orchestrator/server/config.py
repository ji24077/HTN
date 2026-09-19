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
    supabase_admin_emails: frozenset[str] = frozenset()
    tailscale_oauth_client_id: str = ""
    tailscale_oauth_client_secret: str = ""
    tailscale_enrollment_tag: str = "tag:htn-worker"
    worker_gateway_url: str = ""
    self_serve_join: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        database_url = os.environ["DATABASE_URL"]
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        admin = os.getenv("ADMIN_TOKEN", "")
        if admin:
            credential(admin)
        tokens = TypeAdapter(dict[Identifier, str]).validate_python(
            json.loads(os.getenv("WORKER_TOKENS", "{}")), strict=True
        )
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
        admin_emails = frozenset(
            value.strip().lower()
            for value in os.getenv("SUPABASE_ADMIN_EMAILS", "").split(",")
            if value.strip()
        )
        if any(not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) for email in admin_emails):
            raise ValueError("SUPABASE_ADMIN_EMAILS must contain valid email addresses")
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
        gateway = os.getenv("WORKER_GATEWAY_URL", "")
        if gateway:
            parsed = urlsplit(gateway)
            if (
                parsed.scheme != "wss"
                or not parsed.hostname
                or not parsed.hostname.endswith(".ts.net")
                or parsed.username
                or parsed.password
                or parsed.path != "/v1/worker"
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("WORKER_GATEWAY_URL must be the private wss:// .ts.net worker URL")
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError("invalid worker gateway port")
        tag = os.getenv("TAILSCALE_ENROLLMENT_TAG", "tag:htn-worker")
        if not re.fullmatch(r"tag:[a-z][a-z0-9-]{0,62}", tag):
            raise ValueError("invalid TAILSCALE_ENROLLMENT_TAG")
        # Whether /join hands an invite to whoever asks, with no admin token.
        #
        # The default follows who can reach the page rather than a fixed answer, because
        # those are two different deployments. A combined surface is the fleet on a
        # tailnet or a LAN: everyone who can open the page was already let onto the
        # network, so making them ask an admin for a code is friction with nothing behind
        # it. Setting PUBLIC_ORIGIN puts the same page on the internet, where "anyone who
        # can reach it" stops being a meaningful restriction -- so there it is off until
        # the operator says otherwise, and never on by a deploy they did not think about.
        self_serve = os.getenv("DWP_SELF_SERVE_JOIN", "").strip().lower()
        if self_serve and self_serve not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("DWP_SELF_SERVE_JOIN must be a boolean, e.g. 1 or 0")
        open_join = self_serve in {"1", "true", "yes", "on"} if self_serve else not origin
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
            admin_emails,
            os.getenv("TAILSCALE_OAUTH_CLIENT_ID", ""),
            os.getenv("TAILSCALE_OAUTH_CLIENT_SECRET", ""),
            tag,
            gateway,
            open_join,
        )
