# Chip profiles

`src/gpushare/agent/profiles.py` is the executable version of this file. A test
asserts every field there is named here, so the two cannot drift.

Adding a vendor is a row in that table. It used to be six `if vendor == "amd"`
branches; a third vendor would have meant finding all six.

---

## Why a table and not branches

Every field below was paid for. The AMD row cost **$1.36 and three destroyed
pods**, and each wrong guess looked like a different problem:

| Guess | What it looked like | What it was |
|---|---|---|
| `rocm6.4.1 / py3.10` image | nothing, yet | Python below the project floor, ROCm major mismatched to the pinned wheels |
| `rocm7.14.1` image, default pod body | a slow image pull for 22 minutes | the AMD image never starts sshd |
| first request after serving | a hung server | ROCm compiling kernels, 9.26s once |

All three answers were already in `docs/nvidia-amd-migration.md` and
`demo/HANDOFF.md`. Nobody read them — including the agent that wrote this file.

---

## Fields

### `vendor`
`nvidia` or `amd`. `profile_for` raises on anything else rather than defaulting.
A default would make a third vendor look supported: it would rent, install CUDA
wheels onto hardware that cannot run them, and fail far from the cause.

### `project_extra`
What `uv sync --extra <this>` installs. Only used where the project's own lock
is used at all.

### `wheel_index`
The PyTorch index for the isolated venv — `cu128` / `rocm7.1`.

**Must match the host's major version.** The repo pins `rocm7.0` in
`pyproject.toml`; the MI300X host reports **ROCm 7.1.1**, and 7.1 is what was
installed and what ran. The pinned 7.0 is therefore still unproven, and this
table does not pretend otherwise.

### `serve_python`
How to invoke python for serving.

AMD cannot use uv. The image ships none, and Ubuntu 24.04's PEP 668 guard
refuses `pip install --user uv` — silently, under `-q`, which is how it first
appeared as `make: uv: No such file or directory`. It borrows the migration
venv, which already installs matched torch.

### `pod_image`
Verified to boot on RunPod and expose a GPU. The AMD entry is the one Phin's
runs used; picking a newer tag is how the 7.14.1 attempt happened.

### `needs_explicit_sshd`
RunPod's NVIDIA images start sshd. **The AMD image does not.** Without a
`dockerStartCmd` the pod sits at `uptimeInSeconds: 0` with an empty `publicIp`
forever, which is indistinguishable from an image still downloading.

```json
"dockerStartCmd": ["bash","-lc",
  "apt-get update -qq; apt-get install -y -qq openssh-server; \
   mkdir -p /run/sshd /root/.ssh; printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; \
   chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D -e"]
```

### `torch_location`
Where the image's own torch lives, when it ships one. On AMD it is in
`/opt/venv`, so `python3 -c "import torch"` fails on a perfectly healthy pod
and looks like a broken image.

### `jit_cold_start`
Whether the first request compiles kernels.

Measured on MI300X, same checkpoint, same prompt:

```
cold   9.26s
warm   0.45s / 0.45s / 0.45s      (64.7 tok/s)
4090   0.35s                      (82.9 tok/s)
```

**Twenty times between the first request and the second.** A demo has to warm
the pod before the first question or it stalls for nine seconds on stage. Worth
showing rather than hiding — cold start is a real property of moving a workload
to a new chip.

### `bf16`
Both current vendors support it natively, so no dtype conversion happens on
migration. A vendor without it would need one, and the checkpoint's header says
`BF16` in all 290 tensors.

---

## What is not in this table

**Per-chip performance.** That lives in `agent/specs.py` (`CHIPS`: VRAM,
TFLOPS, bandwidth, price) and is used by the cost model. This file is about how
to *operate* a vendor, not how fast it is.

**Rental catalog.** `agent/gpu_matrix.py` holds which chips to rent and where.

**Anything the catalog claims about stock.** RunPod reported MI300X
`lowestPrice: null` and `stockStatus: null` while creation succeeded three
times. Availability is decided by attempting to create.
