# CPU PyTorch example

Upload `train.py` and `validate.py`, then request:

> Train the model on CPU for 200 steps. Use 2 steps for the probe. Reload the
> checkpoint and require held-out MSE <= 0.001. Return checkpoint.pt and metrics.json.

The agent selects `torch` from the imports; no requirements file is needed.
The standard Docker agent includes CPU PyTorch. Dependencies inferred by the agent
are installed in each execution's private environment. CUDA and Blender are outside
the current supported scope.
