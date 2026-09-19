.PHONY: help setup-server setup-cuda setup-rocm setup-agent check server worker mock test fmt

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup-server:  ## Jack's server box (no torch)
	uv sync --extra server --extra dev
setup-cuda:    ## NVIDIA training box
	uv sync --extra cuda --extra worker --extra dev
setup-rocm:    ## AMD box
	uv sync --extra rocm --extra worker --extra dev
setup-agent:   ## Ji's laptop (no torch)
	uv sync --extra agent --extra dev

check:         ## verify the GPU stack actually works on this machine
	uv run python scripts/check_env.py

server:        ## run the hub
	uv run gpushare-server
worker:        ## run the worker daemon
	PYTHONUNBUFFERED=1 uv run gpushare-worker
mock:          ## replay fake events into the dashboard
	uv run gpushare-mock

test:          ## contract tests - run before every push
	uv run pytest -q
fmt:
	uv run ruff format . && uv run ruff check --fix .
