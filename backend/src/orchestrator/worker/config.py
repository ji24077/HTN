"""Worker identity, transport, and reported capabilities."""

import ipaddress
import os
import platform
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import TypeAdapter

from ..shared.protocol import Capabilities, Identifier, Machine
from ..shared.security import credential
from .devices import capabilities


def machine_specs() -> Machine:
    """Best-effort host metadata; unavailable RAM stays unknown, never fabricated."""
    ram_mb = None
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            ram_mb = pages * page_size // (1024 * 1024)
    except (AttributeError, OSError, ValueError):
        pass
    cores = os.cpu_count()
    return Machine(
        os=platform.system()[:32],
        arch=platform.machine()[:32],
        cpu_model=platform.processor()[:128] or None,
        logical_cores=min(cores, 4096) if cores else None,
        total_ram_mb=ram_mb,
    )


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
        worker_caps = capabilities(kind, os.getenv("WORKER_RUNTIME", "cpu"))
        worker_caps.machine = machine_specs().model_copy(
            update={"max_concurrency": 1, "runtime_control": "startup"}
        )
        return cls(
            url=url,
            worker_id=TypeAdapter(Identifier).validate_python(os.environ["WORKER_ID"]),
            token=credential(os.environ["WORKER_TOKEN"]),
            capabilities=worker_caps,
            paused=paused == "true",
            transport=transport,
        )
