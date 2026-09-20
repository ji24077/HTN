"""Service submission and reverse HTTP transport contracts."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator

from .dependencies import CPURequirements, DependencyPlan
from .protocol import Model

CHUNK_SIZE = 64 * 1024
BODY_LIMIT = 1024 * 1024


class ServiceConfig(DependencyPlan):
    entrypoint: str | None = Field(default=None, max_length=240)
    args: list[Annotated[str, Field(max_length=256)]] = Field(default_factory=list, max_length=32)
    working_directory: str = Field(default=".", max_length=240)
    readiness_path: str = Field(default="/health", max_length=256)
    startup_timeout_seconds: int = Field(default=600, ge=1, le=7200)
    request_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    lifetime_seconds: int | None = Field(default=None, ge=1, le=31536000)
    concurrency: int = Field(default=4, ge=1, le=16)
    requirements: CPURequirements = Field(default_factory=CPURequirements)

    @field_validator("readiness_path")
    @classmethod
    def local_path(cls, value):
        if not value.startswith("/") or value.startswith("//") or "\\" in value or "#" in value:
            raise ValueError("Readiness must be a local HTTP path")
        return value


class ServiceAction(Model):
    action_id: UUID
    operation: Literal["stop", "restart"]


def safe_headers(headers):
    """Forward API headers without platform credentials or HTTP hop semantics."""
    pairs = list(headers)
    denied = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    for name, value in pairs:
        if name.lower() == "connection":
            denied.update(part.strip().lower() for part in value.split(","))
    return [
        (k, v)
        for k, v in pairs
        if k.lower() not in denied and not k.lower().startswith("x-dispatch-")
    ]
