# Hosting uploaded services

Choose **Service** in the existing project upload form. Upload a Python HTTP
server, select its runtime/VRAM requirements, and optionally set an expiry.
The result is a stable authenticated endpoint at `/serve/{job_id}/`.

The program reads `DISPATCH_SERVICE_PORT`, binds to `127.0.0.1`, and returns 2xx
from `/health` only when ready. A custom readiness path is supported. One Python
file is selected automatically; multi-file projects can specify an entrypoint
or answer the existing clarification form. The original uploaded code is never
rewritten. The service uses one worker slot and starts again on a compatible
worker after failure. There is no result-file validator for Service mode.

Upload [the standard-library example](../examples/projects/service/server.py)
to exercise `/health`, `/stream`, and a binary POST echo without GPUs or model
calls. The detail panel shows the URL, readiness, worker, expiry, restarts, logs,
execution history, and Stop/Restart controls. Logs reuse the bounded worker journal:
roughly 1,000 events or 1 MiB per attempt, then a truncation marker. Continuous log
rotation/export is not implemented; a long-lived service can exhaust that log budget
while continuing to serve normally. Use the existing fleet bearer token
when calling the endpoint from a client. Platform credentials are stripped before
forwarding to the uploaded server.

## Backend configuration

The public and worker listeners share PostgreSQL but run in separate processes.
Set the same `SERVICE_BRIDGE_TOKEN` (a random secret) in both backend environments.
Set `SERVICE_BRIDGE_URL=http://127.0.0.1:8081` in the public backend, adjusting the
port if necessary. If the listeners are on different hosts, use an HTTPS bridge
URL. The token stays in the backend and is not passed into uploaded processes.
The combined local server does not need these settings.

Run exactly one worker-gateway process for this version; the public endpoint
must reach that process. Live reverse sockets are owned in memory there. Multiple
worker gateways behind a load balancer require a future routing-owner layer.
A full gateway restart invalidates the main worker session and restarts the model
process; expect a cold load. A brief service-control-channel interruption alone
reconnects within the current worker session and keeps the serving process alive. Workers need the updated `python_project` executor, which
advertises `python_service`. Desktop/iOS and older Python workers are not eligible.

Service bundles and HTTP traffic use outbound WebSocket connections to the
existing private gateway host, including through the embedded tunnel. There are
no public worker ports. Each worker opens a small pool of one-request reverse
sockets, separately from its lease heartbeat connection. The default is four
concurrent requests; response queues hold at most two 64 KiB frames per socket.
Configure upstream reverse proxies to preserve streaming rather than buffer SSE.
The Vite development proxy includes `/serve`.

## Lifetime, errors, and recovery

Service assignments use renewable short leases without a finite execution
runtime. Ordinary jobs still require finite deadlines. Startup defaults to ten
minutes. Requests default to a five-minute total lifetime, including streamed response
delivery and slow consumers, independently of service lifetime. This is not just
a time-to-first-byte or idle timeout; set `request_timeout_seconds` for longer streams. Expiry is fixed at submission and does not move when a worker restarts.

Readiness is checked every two seconds with a two-second timeout. A failed check
withdraws routing; three consecutive failures after readiness restart the process.
Retries back off from one to sixty seconds. Five failed attempts in ten minutes
suspend automatic recovery and show Failed; Restart clears that suspension.
Absent compatible capacity remains Pending. The deterministic scheduler manages
this even when no LLM client is configured. The existing supervisor can diagnose
failures and cancel the service; it does not run on each HTTP request.

Stop and Restart revoke the old assignment immediately. Existing requests may
be interrupted; there is no graceful request-drain window in this version. The
worker terminates the serving process group and removes its temporary workspace.
The URL remains the same across Restart. In-flight streams cannot migrate or
resume on another worker, and the gateway never automatically replays a request.

HTTP errors: 401 for missing authentication, 413 above the 1 MiB body limit,
429 when all service slots are occupied, 503 while unavailable/stopped,
502 for upstream connection failure, and 504 for a request timeout before headers.
A failure after streaming begins closes the response. Client disconnect closes
the upstream request; the model server must honor cancellation to stop computation.

This endpoint supports HTTP APIs and streaming responses, not hosted interactive
websites: cookies and platform credentials are stripped and responses carry a
sandbox CSP. Dependencies and model files must already be available to the worker
or be fetched by the uploaded code. Temporary workspaces are not persistent disks.

## Fine-tuned model serving

[The vLLM launcher template](../examples/projects/model_service/server.py) starts
`vllm serve` with the assigned port. Change `MODEL_PATH` to your pinned model
location and install a compatible vLLM runtime in each eligible worker's Python
environment. Set the correct CUDA/VRAM requirements when uploading. The inference
client base URL becomes `/serve/{job_id}/v1`. The template follows vLLM's documented
[HTTP serving interface](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
This template has not been validated on a physical GPU in this change.

For an adapter-only fine-tune, the uploaded serving code also needs the matching
base weights and adapter configuration. Check that every replacement worker can
load the same model version; the orchestrator does not yet distribute or cache
large model files automatically. Python workers verify CUDA/MPS access with a bounded
PyTorch probe before advertising it. This checks access, not whether a particular
model fits or meets its performance target.

## API and validation

For a hands-on CPU test, build the frontend and run these in three terminals
from the repository root:

```sh
uv run --project backend --python 3.12 --extra demo orchestrator-demo server --port 8080
WORKER_EXECUTOR=python_project uv run --project backend --python 3.12 --extra demo orchestrator-demo worker-a --port 8080
WORKER_EXECUTOR=python_project uv run --project backend --python 3.12 --extra demo orchestrator-demo worker-b --port 8080
```

Open `http://127.0.0.1:8080`, choose **New job → Service**, upload
`examples/projects/service/server.py`, enter a description, and submit with CPU,
zero VRAM, and `/health`. Once Ready, open the endpoint with `/health` or `/stream`
appended in the same browser session. Stop should make the endpoint return 503;
Restart should restore it at the same URL. Stop the terminal for the worker shown
in the service panel to watch the other worker take over. These are separate worker
processes on one host; this verifies orchestration, not GPU or cross-machine networking.
Demo state lives in `.demo/`; Ctrl-C stops each launcher.

To enable the supervising agent using the existing `OPENAI_*` settings in `.env`,
start the server with this launcher instead:

```sh
uv run --project backend --python 3.12 --extra demo python scripts/start-service-demo.py --port 8080
```

The Jobs list and service details both provide **Stop service**. A worker hosting
a service shows **Busy · serving** and **Slot reserved**, including while the
service is idle between requests. Stop withdraws the endpoint and releases the
worker; Restart brings the service back at the same URL.

Python workers now probe the serving interpreter at startup and report a reason
when no usable device is found. GPU availability requires a successful PyTorch
tensor operation. To start the demo worker with the GPU runtime installed:

```sh
WORKER_EXECUTOR=python_project WORKER_RUNTIME=auto uv run --project backend --python 3.12 --extra demo --with torch==2.14.0 --with numpy orchestrator-demo worker-b --port 8080
```

`WORKER_RUNTIME=cpu` keeps CPU scheduling while still reporting detected hardware.
`auto` selects a verified CUDA or MPS runtime. Python workers select their runtime
at startup; the remote CPU/GPU control belongs to paired desktop agents. Upload
with MPS requirements and zero dedicated VRAM for the Apple GPU; unified memory
is not reported as a separate VRAM pool. Multiple local workers share one machine.
The tensor check follows the [PyTorch MPS interface](https://docs.pytorch.org/docs/2.14/notes/mps.html).

`POST /v1/jobs` accepts `execution_mode: "service"` and optional `service` settings:
`entrypoint`, `args` (argument array supporting `{port}`), `working_directory`,
`readiness_path`, `requirements`, `startup_timeout_seconds`,
`request_timeout_seconds`, `concurrency`, and `lifetime_seconds`.

`GET /v1/jobs/{id}` includes service state and endpoint. Service controls use
`POST /v1/jobs/{id}/service-actions` with an `action_id` UUID and `operation` equal
to `stop` or `restart`. Actions and uploads are idempotent. Existing task cancellation
also stops the whole service when cancelling its root or current attempt.
Historical terminal attempts remain idempotent no-ops. Services use Stop/Restart;
the supervisor rejects generic pause/resume to avoid treating a deliberate pause as a failure. These routes retain fleet-admin authentication.

Run the integration suite with temporary PostgreSQL, real serving subprocesses,
real worker WebSockets, and separate public/worker HTTP listeners:

```sh
RUN_SUPERVISOR_TESTS=1 PYTHONPATH=backend/tests uv run --project backend --python 3.12 --extra demo python -m unittest integration.test_hosted_services integration.test_uploaded_programs -v
```

The CPU fixture establishes routing, binary transfer, streaming, auth, expiry,
worker replacement, stale-assignment rejection, request cancellation, capacity,
and crash-loop recovery. It does not establish GPU/model performance.
