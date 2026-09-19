# gpushare

**Rent GPUs from anyone. An agent pools them and makes them fast.**

Design and tickets: `docs/handbook.md`. Your own part: `docs/jack.md` / `docs/ethan.md` / `docs/ji.md`.
This file is how we work together.

---

## Setup (5 minutes, per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # once
git clone <repo> && cd gpushare
cp .env.example .env                               # edit worker_id / owner / server ip
```

Then **one line, depending on what the machine is:**

| Machine | Command | Gets torch? |
|---|---|---|
| Jack's server box | `make setup-server` | no |
| NVIDIA training box | `make setup-cuda` | CUDA wheel |
| AMD box | `make setup-rocm` | ROCm wheel |
| Ji's laptop | `make setup-agent` | no |

**Then verify, before anything else:**

```bash
make check
```

```
torch          2.5.1+cu124
build          CUDA 12.4
available      True
gpu            NVIDIA GeForce RTX 3090  25.4GB  cc8.6
chip_class     ampere_24gb
trainable      True
```

If `available` is `False` on the AMD box, the ROCm wheel doesn't match the driver.
Run `rocminfo`, read the version, and fix the index URL in `pyproject.toml`.
The install succeeding tells you nothing — only `make check` does.

### Why uv and not pip

One reason that matters: **torch is a different package on each machine.** The server needs none, Ethan needs the CUDA wheel, the AMD box needs the ROCm wheel. With pip that's three `--index-url` incantations somebody has to remember at 3am. Here it's declared once in `pyproject.toml` and each machine runs one `make` target.

Two extras that can never coexist are declared as conflicting, so uv resolves them separately instead of failing:

```toml
[tool.uv]
conflicts = [[{ extra = "cuda" }, { extra = "rocm" }]]
```

If you'd rather use pip or conda and you're faster that way, do that — learning a new tool mid-hackathon is a net loss. Just keep `make check` passing.

---

## Repo layout — ownership is the conflict-avoidance strategy

```
src/gpushare/
├── contracts.py      ★ SHARED — all three. Frozen at hour 0.
├── settings.py       ★ shared config (env / .env)
│
├── server/           JACK    S-1 hub, S-8 sqlite, N-4 checkpoint transfer
├── worker/           JACK    N-3 daemon, N-5 cli, spec
│
├── trainer/          ETHAN   T-1 loop, T-2/T-3 diloco, T-4 ckpt, T-5 migrate, T-6 probe
│
├── agent/            JI      S-4 chips+router, S-5 H, S-9 pricing, S-2 mock
└── dashboard/        JI      S-3 static html, served by Jack's hub

scripts/              N-1 netcheck.sh, N-2 gloo_check.py, check_env.py
tests/                contract round-trip tests
docs/                 handbook + per-person briefs
```

**Each of you owns a directory. Nobody edits someone else's.** That's not politeness, it's how three people push to one branch without merge conflicts — you literally never touch the same file.

The one shared file is `contracts.py`, and it's frozen.

---

## Branching: `main` only

No feature branches, no PRs. Three people for twenty hours — review overhead costs more than it saves, and directory ownership already prevents the conflicts that branches exist to manage.

```bash
git pull --rebase && make test && git push
```

**Rules:**

1. **Pull with `--rebase`.** Merge commits from three people on one branch turn the history into soup.
2. **`make test` before every push.** It's six contract tests and takes a second. A broken `contracts.py` on main blocks all three of you; a broken `trainer/` blocks nobody.
3. **Prefix commits with the ticket ID** — `N-3: worker registers and heartbeats`. Makes the integration checkpoints (T+3, T+8, T+13, T+17) verifiable at a glance.
4. **Push often.** Small pushes, many of them. Sitting on four hours of work is how integration fails at T+8.
5. **One exception for branches:** if you're doing something long and risky — Ethan rewriting T-3 onto the server-mediated fallback, say — branch it so main stays runnable. Merge as soon as it's green.

### Changing `contracts.py`

**Requires all three of you, in the same conversation, at the same time.** Then everyone pulls immediately. This is the one file where a unilateral change breaks the other two silently.

If you think you need a new field, ask first whether you actually need it. Most of the time you don't.

---

## The contracts are enforced, not documented

`contracts.py` is pydantic, not prose. Two things that used to be "please be careful" are now impossible:

**A `chip_class` typo fails loudly.** It's a `Literal`, not a `str`. Jack writes it, Ethan tags probes with it, Ji dispatches the router on it — if any of you writes `ampere24gb` instead of `ampere_24gb`, you get a `ValidationError` at the boundary instead of per-chip learning silently never happening.

**The fixed-work invariant is a model validator.** `JobConfig` refuses to construct if `micro_batch × grad_accum × workers × seq_len != global_batch_tokens`. Ji cannot hand Ethan a config that does *less work* and call it faster. That number is the spine of the whole before/after demo, so it's checked by the type system rather than by whoever remembers.

`extra="forbid"` is deliberate: a field the receiver doesn't know about is an error, not a silent drop.

### The stdout protocol (Ethan → Jack)

The trainer emits events as JSON lines on **stdout**; the daemon parses them.

```python
from gpushare.contracts import TrainStep, emit
emit(TrainStep(worker_id="w1", step=1234, loss=1.83, step_time_s=0.21, tokens=32768))
```

**All human-readable logging goes to stderr.** torch will print warnings, and if they land on stdout the parser chokes. `logging.basicConfig(stream=sys.stderr)` is already in `trainer/loop.py`. Run workers with `PYTHONUNBUFFERED=1` (the `make worker` target does).

The daemon skips unparseable lines rather than crashing — one stray warning shouldn't kill a run.

---

## Commands

For the skill-guided CUDA/HIP code translation demo, see the
[demo instructions](demo/README.md) and [real GPU results](demo/RUN_RESULTS.md).
The demo verifies both translation directions and checkpoint continuation;
its benchmark does not establish a general training speedup.

For the checkpointed NVIDIA-to-AMD experiment, see the
[migration runbook](docs/nvidia-amd-migration.md). It covers the approved rental,
SSH setup, preserved training state, prediction gates, and cleanup limits.

```bash
make check        # what is this machine? run first, on every box
make server       # the hub
make worker       # worker daemon
make mock         # replay fake events -> dashboard (Ji's unblocking trick)
make test         # contract tests
make fmt          # ruff format + fix
```

---

## Day one, in order

**All three, first 60 minutes:** sit together, read `contracts.py` line by line, agree it, commit it. Nothing else until that's done.

Then:

| | First thing |
|---|---|
| **Jack** | `scripts/netcheck.sh` — is Tailscale `direct` or `relay`? Then gloo across two machines |
| **Ethan** | Does torchft nightly run here? Pin the version. Before you write any code |
| **Ji** | `make mock` — the fake event emitter. Thirty minutes, and nothing blocks you all weekend |

Jack and Ethan report at **T+3**. Their two answers determine everyone's remaining plan.
