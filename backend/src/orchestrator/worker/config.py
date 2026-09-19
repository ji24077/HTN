"""Worker identity, transport, and reported capabilities."""

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import TypeAdapter

from ..shared.protocol import Capabilities, Identifier
from ..shared.security import credential


@dataclass(frozen=True)
class WorkerConfig:
    url: str
    worker_id: str
    token: str
    capabilities: Capabilities
    paused: bool = False
    transport: str = "direct"

    @classmethod
    def from_env(cls, kind: str) -> "WorkerConfig":
        url = os.getenv("SERVER_URL", "ws://127.0.0.1:8080/v1/worker")
        parsed = urlsplit(url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("invalid SERVER_URL")
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if parsed.scheme != "wss" and not (parsed.scheme == "ws" and local):
            raise ValueError("SERVER_URL requires wss except for ws on loopback")
        paused = os.getenv("WORKER_PAUSED", "false").lower()
        if paused not in {"true", "false"}:
            raise ValueError("WORKER_PAUSED must be true or false")
        transport = os.getenv("WORKER_TRANSPORT", "direct")
        if transport not in {"direct", "tailscale"}:
            raise ValueError("WORKER_TRANSPORT must be direct or tailscale")
        return cls(
            url=url,
            worker_id=TypeAdapter(Identifier).validate_python(os.environ["WORKER_ID"]),
            token=credential(os.environ["WORKER_TOKEN"]),
            capabilities=Capabilities(
                runtime=os.getenv("WORKER_RUNTIME", "cpu"),
                vram_mib=int(os.getenv("WORKER_VRAM_MIB", "0")),
                kinds=[kind],
            ),
            paused=paused == "true",
            transport=transport,
        )
