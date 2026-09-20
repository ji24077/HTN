"""Start the local service dashboard with the existing supervising-agent configuration.

Run from the repository root:
uv run --project backend --python 3.12 --extra demo python scripts/start-service-demo.py --port 8080
"""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from orchestrator.demo import main

# Keep the local demo database, worker credentials, and loopback authentication.
# Import only agent settings from the app environment, never production DB/auth settings.
for key, value in dotenv_values(Path(".env")).items():
    if value is not None and key.startswith(("OPENAI_", "SUPERVISOR_")):
        os.environ.setdefault(key, value)
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Configure OPENAI_API_KEY in .env to enable the supervising agent.")
os.environ["SUPERVISOR_ENABLED"] = "true"
os.environ.pop("PUBLIC_ORIGIN", None)
sys.argv.insert(1, "server")
main()
