"""Server environment configuration; importing opens no connections."""

import json
import os
from dataclasses import dataclass

from pydantic import TypeAdapter

from ..shared.protocol import Identifier
from ..shared.security import credential


@dataclass(frozen=True)
class ServerConfig:
    database_url: str
    redis_url: str | None
    admin_token: str
    worker_tokens: dict[str, str]

    @classmethod
    def from_env(cls) -> "ServerConfig":
        database_url = os.environ["DATABASE_URL"]
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        admin = credential(os.environ["ADMIN_TOKEN"])
        tokens = TypeAdapter(dict[Identifier, str]).validate_python(
            json.loads(os.environ["WORKER_TOKENS"]), strict=True
        )
        if not tokens:
            raise ValueError("WORKER_TOKENS must enroll at least one worker")
        for token in tokens.values():
            credential(token)
        if admin in tokens.values() or len(set(tokens.values())) != len(tokens):
            raise ValueError("admin and each worker must have distinct tokens")
        return cls(database_url, os.getenv("REDIS_URL") or None, admin, tokens)
