"""Public ingest settings distributed with worker enrollment, never Sentry API credentials."""

import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class WorkerTelemetry(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    dsn: str | None = Field(default=None, max_length=2048)
    environment: str = Field(default="development", max_length=200)
    release: str | None = Field(default=None, max_length=200)

    @field_validator("dsn")
    @classmethod
    def public_dsn(cls, value):
        if value is None:
            return value
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
            or not url.username
            or url.password is not None
            or url.query
            or url.fragment
            or not url.path.rsplit("/", 1)[-1].isdigit()
            or any(character.isspace() for character in value)
        ):
            raise ValueError("Expected an HTTPS public Sentry DSN")
        _ = url.port
        return value

    @field_validator("environment", "release")
    @classmethod
    def single_line(cls, value):
        if value is not None and any(ord(character) < 32 for character in value):
            raise ValueError("Telemetry metadata must be a single line")
        return value


def worker_telemetry() -> dict:
    """An explicit empty worker DSN disables reporting; otherwise inherit the backend DSN."""
    try:
        settings = WorkerTelemetry(
            dsn=os.getenv("SENTRY_WORKER_DSN", os.getenv("SENTRY_DSN", "")).strip() or None,
            environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
            release=os.getenv("SENTRY_RELEASE") or None,
        )
    except ValidationError:
        # Optional telemetry must not prevent enrollment or expose a malformed credential.
        settings = WorkerTelemetry()
    return settings.model_dump()
