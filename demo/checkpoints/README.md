# Saved checkpoints for the handoff

The repository includes the exact LoRA adapters and saved tokenizers for:

| Directory | Status |
|---|---|
| `ckpt/laptop-lora-v2/` | Existing default of `scripts/extract.py`; not speed-tested across the full matrix |
| `ckpt/laptop-lora-v3b/` | Frozen experimental speed-test checkpoint; rejected as a replacement for v2 because of appointment-year regressions |

Each adapter weight file is about 17.6 MB. These are adapters requiring the
Qwen base model, not standalone language models. Both have saved training
metadata in `meta.json`. The [manifest](manifest.json) identifies the exact
files and combined base/adapter hashes. The model-selection decision and data
are in [optimization-v3](../../docs/optimization-v3.md).

The base is **Qwen/Qwen2.5-0.5B**, revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Its roughly 1 GB weight file is
downloaded from the public upstream model, not stored in Git. A GPU/API key is
not needed for the download or checksum verification.

## Populate a dedicated cache

Run this from the repository root in Bash, after installing the dependencies
for the experiment. It deliberately uses a dedicated cache: the unchanged
evaluator resolves the base through the cache's `main` reference. Do not point
these commands at a shared cache used by other projects.

```bash
export HF_HOME="$PWD/.cache/benchmark-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

base = json.loads(Path("demo/checkpoints/manifest.json").read_text())["base"]
cache = Path(os.environ["HF_HUB_CACHE"])
ref = cache / "models--Qwen--Qwen2.5-0.5B" / "refs" / "main"
if ref.exists() and ref.read_text().strip() != base["revision"]:
    raise RuntimeError("Use an empty dedicated cache; main already names another revision")
snapshot = snapshot_download(
    repo_id=base["repo_id"], revision=base["revision"], cache_dir=str(cache),
    allow_patterns=list(base["files"]),
)
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(base["revision"], encoding="utf-8")
print(snapshot)
PY
export BASE="$HF_HUB_CACHE/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
PYTHONPATH=src python scripts/verify_benchmark_handoff.py --base "$BASE"
```

Keep `HF_HOME` and `HF_HUB_CACHE` set when running the unchanged evaluator. The
validation script checks that its cached loader resolves the same bytes as
`--base`. The allowlist also keeps unrelated downloaded JSON/text files out of
the snapshot: those files would alter the experiment's combined model hash.

The checker without `--base` verifies both published adapters and the evidence
using only local files. With `--base`, it additionally verifies both complete
base/adapter identities. Neither command starts a model, contacts RunPod, or
modifies reports.

## Attribution and changes

Qwen2.5-0.5B is provided by the Qwen team / Alibaba Cloud. The upstream license
states Copyright 2024 Alibaba Cloud and Apache License 2.0. A complete copy of
the [upstream license](QWEN_LICENSE) is included, retrieved from the
[pinned revision](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/LICENSE).

The adapters are this project's task-specific fine-tuning changes, with training
records and model-selection results retained in the repository. Tokenizer files
and the chat template are Qwen-derived files serialized with the local training
setup; the saved tokenizer configuration records the experiment's padding and
loading settings. The original base weights are unchanged. The included license
and this attribution live outside the adapter directories so they do not alter
the frozen experiment's file identity.
