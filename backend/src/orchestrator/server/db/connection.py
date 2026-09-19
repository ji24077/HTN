"""Use verified system trust roots for remote PostgreSQL connections."""

import ssl
from urllib.parse import parse_qs, urlsplit


def connection_options(url: str) -> dict:
    query = parse_qs(urlsplit(url).query)
    if query.get("sslmode") == ["verify-full"]:
        # asyncpg otherwise requires ~/.postgresql/root.crt even when the
        # server certificate chains to an already trusted system CA.
        roots = query.get("sslrootcert", [None])[0]
        return {"ssl": ssl.create_default_context(cafile=roots)}
    return {}
