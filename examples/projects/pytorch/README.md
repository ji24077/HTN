# PyTorch example

Upload `train.py`, `validate.py`, and `device.py`, then request:

> Train the model for 200 steps on an available compatible GPU, or CPU if none is available.
> Use 2 steps for the probe. Reload the checkpoint and require held-out MSE <= 0.001.
> Return checkpoint.pt and metrics.json.

For GPU-only execution, explicitly request CUDA or MPS with no CPU fallback.
The scripts honor the worker's `DISPATCH_DEVICE` selection and fail if that device is unavailable.
The agent selects `torch` from imports; the worker preserves its installed PyTorch build.
The standard Docker image includes CPU PyTorch. NVIDIA hosts use the GPU compose override
in [the Docker guide](../../../docs/docker-agent.md#uploaded-python-projects-on-gpu-workers).
Apple MPS requires a native macOS Python worker.
