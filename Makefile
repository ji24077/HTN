.PHONY: help setup-server setup-cuda setup-rocm setup-agent check probe fit data train eval server worker mock ui test fmt

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

probe:         ## measure real step time on THIS gpu (run on a pod) -> probes.jsonl
	uv run python scripts/probe.py
fit:           ## fit per-chip constants from probes.jsonl and report the error
	uv run python scripts/fit_calibration.py probes.jsonl

data:          ## generate the SFT set with a hosted model -> data/
	uv run python scripts/gen_data.py --n 2000
train:         ## fine-tune Qwen2.5-0.5B on the task (run on a pod)
	uv run python scripts/train.py --out ckpt/run
eval:          ## score a model on the held-out set (run on a pod)
	uv run python scripts/evaluate.py --model ckpt/run

server:        ## run the hub
	uv run gpushare-server
worker:        ## run the worker daemon
	PYTHONUNBUFFERED=1 uv run gpushare-worker
mock:          ## replay fake events into the dashboard
	uv run gpushare-mock
ui:            ## research console — data -> RunPod training -> verified agent actions
	uv run --extra server --extra agent gpushare-ui

test:          ## contract tests - run before every push
	uv run pytest -q
fmt:
	uv run ruff format . && uv run ruff check --fix .
