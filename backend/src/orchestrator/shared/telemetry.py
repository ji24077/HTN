"""Optional Sentry reporting shared by the server and worker; blank SENTRY_DSN disables it."""

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

import sentry_sdk
from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber

REDACTED = "[redacted]"
SENSITIVE_KEY = re.compile(
    r"authorization|cookie|token|secret|passw|pwd|credential|authkey|api[-_]?key|publishable"
    r"|signature|database_url|redis_url|dsn|private[-_]?key|^code$",
    re.IGNORECASE,
)
# Credential shapes that appear inside free text: bearer headers, JWTs, Tailscale
# and Supabase keys, and passwords embedded in connection URLs.
SENSITIVE_VALUE = re.compile(
    r"Bearer\s+[A-Za-z0-9._~+/=-]+"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"
    r"|tskey-[A-Za-z0-9-]+"
    r"|sb_(?:secret|publishable)_[A-Za-z0-9_-]+"
    r"|(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"
    r"|-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----[\s\S]*?-----END (?:[A-Z ]*PRIVATE KEY)-----"
    r"|(?i:[?&#]code=)[^&#\s]+"
    r"|(?i:\bcode[\"']?\s*[:=]\s*[\"']?)[0-9a-fA-F]{32}"
)


def secret_values() -> frozenset[str]:
    """Configured credentials, redacted by value wherever they appear (e.g. a repr)."""
    found: set[str] = set()
    for name, value in os.environ.items():
        if not value or name == "SENTRY_DSN":
            continue
        if name == "WORKER_TOKENS":
            try:
                found.update(str(token) for token in json.loads(value).values())
            except (ValueError, AttributeError):
                found.add(value)
        elif name.endswith("_URL"):
            try:
                password = urlsplit(value).password
            except ValueError:
                password = None
            if password:
                found.add(password)
        elif SENSITIVE_KEY.search(name):
            found.add(value)
    # Short values would redact ordinary words; real credentials are >= 24 chars.
    return frozenset(value for value in found if len(value) >= 8)


class Scrubber:
    def __init__(self, secrets: frozenset[str]):
        self.secrets = sorted(secrets, key=len, reverse=True)

    def text(self, value: str) -> str:
        for secret in self.secrets:
            if secret in value:
                value = value.replace(secret, REDACTED)
        return SENSITIVE_VALUE.sub(REDACTED, value)

    def scrub(self, value: Any, depth: int = 0) -> Any:
        if depth > 12:
            return REDACTED
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {
                key: REDACTED
                if isinstance(key, str) and SENSITIVE_KEY.search(key) and item is not None
                else self.scrub(item, depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            # Header lists arrive as [name, value] pairs rather than mappings.
            if (
                len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], str)
                and SENSITIVE_KEY.search(value[0])
            ):
                return [value[0], REDACTED]
            return [self.scrub(item, depth + 1) for item in value]
        return value

    def event(self, event: Any, _hint: Any = None) -> Any:
        return self.scrub(event)

    def log(self, record: Any, _hint: Any = None) -> Any:
        return self.scrub(record)


def init_sentry(component: str, **tags: str) -> bool:
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        return False
    scrubber = Scrubber(secret_values())
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE") or None,
        traces_sample_rate=1.0,
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",
        enable_logs=True,
        # Model reprs lose sensitive field boundaries before before_send runs.
        # Keep traceback locations, but do not collect potentially unsafe locals.
        include_local_variables=False,
        send_default_pii=True,
        max_request_body_size="medium",
        event_scrubber=EventScrubber(
            denylist=[*DEFAULT_DENYLIST, "authkey", "worker_tokens", "publishable_key"],
            recursive=True,
        ),
        before_send=scrubber.event,
        before_send_transaction=scrubber.event,
        before_send_log=scrubber.log,
        before_breadcrumb=scrubber.event,
    )
    sentry_sdk.set_tag("component", component)
    for key, value in tags.items():
        sentry_sdk.set_tag(key, value)
    return True
