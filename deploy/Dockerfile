FROM golang:1.27.1-bookworm AS tunnel
WORKDIR /build
COPY transport/tailscale/go.mod transport/tailscale/go.sum ./
RUN go mod download
COPY transport/tailscale/*.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /orchestrator-tunnel .

FROM node:24-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/src ./src
COPY --from=frontend /app/backend/src/orchestrator/server/web ./src/orchestrator/server/web
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home worker \
    && mkdir /state && chown worker:worker /state && chmod 700 /state
WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
COPY --from=tunnel /orchestrator-tunnel /usr/local/bin/orchestrator-tunnel
COPY --chown=10001:10001 fixtures /app/fixtures
ENV DWP_FIXTURES_DIR=/app/fixtures DWP_RELEASES_DIR=/app/releases
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENV PUBLIC_BIND_HOST=0.0.0.0 TAILSCALE_STATE_DIR=/state TAILSCALE_HOSTNAME=orch-backend
USER 10001:10001
EXPOSE 8080
CMD ["orchestrator-backend"]
