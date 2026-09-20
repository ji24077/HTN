# Relay demo

This Windows PowerShell demo uses three companion checkouts or worktrees. Place
them under one parent directory with these names and branches:

```text
workspace/
  HTN/          # relay/fleet-safety: fleet backend and provider fixes
  ji-review/    # relay/gpu-safety: GPUShare backend and recorded artifacts
  relay-demo/   # relay/demo-fixes: GPU Lab frontend and this guide
```

The scripts are packaged in `relay-demo/scripts/relay-demo/`, but must run from
the shared parent directory. They resolve the checkout names above relative to
their own location. Copy all three together; running them inside this checkout
does not use the intended layout.

Install dependencies before launching. The launcher expects Node.js 24.15+ and
`node.exe` on PATH, the two Python environments below, installed frontend
dependencies, and built fleet dashboard assets. A fresh checkout does not contain
these generated files. From the shared parent directory, a setup sequence is:

```powershell
uv sync --project .\HTN\backend --python 3.12 --extra demo
uv sync --project .\ji-review --python 3.12 --extra server --extra agent --extra dev
npm --prefix .\HTN\frontend ci
npm --prefix .\HTN\frontend run build
npm --prefix .\relay-demo\frontend ci
```

Keep the GPU checkout's recorded `demo/` artifacts in place. Live GPU execution
has additional provider and remote-runtime requirements described by that
checkout; the setup above does not provision GPUs.

Then copy the launchers and start the default recorded-evidence demo from the
same parent directory:

```powershell
Copy-Item -LiteralPath .\relay-demo\scripts\relay-demo\Start-RelayDemo.ps1 -Destination .
Copy-Item -LiteralPath .\relay-demo\scripts\relay-demo\Stop-RelayDemo.ps1 -Destination .
Copy-Item -LiteralPath .\relay-demo\scripts\relay-demo\Stop-RelayFleetDatabase.py -Destination .
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-RelayDemo.ps1
```

Open the **main app at http://127.0.0.1:5175/** and choose **GPU Lab**, or open its chat directly at **http://127.0.0.1:5175/gpu-lab**. The frontend uses `relay-demo` on `relay/demo-fixes` (based on `FE-WEB`); the GPU backend uses `ji-review` on `relay/gpu-safety`; the local fleet backend uses `HTN` on `relay/fleet-safety`.

The launcher starts the main fleet backend on port 8091 with its existing local browser-session flow and an isolated PostgreSQL database under `.relay-fleet-local/.demo/postgres`. This local fleet starts empty; it does not copy existing workers or jobs. External AI calls and integrations are disabled for this backend. Startup checks the main session and authenticated fleet snapshot through the frontend proxy before reporting success; stopping the demo also stops its database while preserving its files. GPU requests are pinned to the local frontend proxy. Logs and process state live in `.relay-demo/` beside the checkouts; keep these runtime directories and local credentials out of commits.

The GPU lab reads saved hardware measurements. It does not rent GPUs, discover provider resources, or change workloads in the default mode. Each result is explicitly marked recorded. Model and dataset hashes identify the exact experiment; these results do not describe whichever model is currently serving elsewhere.

For existing, configured GPU resources and live controls, stop this demo and run the launcher with `-Live`. GPU jobs and model API calls can incur existing-provider charges. No new rental workflow is added. Serving refuses an implicit replacement of an active model; stopping it is an explicit action. This is not automatic traffic migration or rollback.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Stop-RelayDemo.ps1
```

## Ninety-second script

“Relay compares compute configurations and checks whether the faster result still gives the same answers.

“These are recorded measurements of a Qwen model on real NVIDIA and AMD GPUs. The checkpoint is experimental v3b; the results are tied to the model and evaluation hashes shown here.

“On A5000, this optimization reduced resident response time from about 1.38 seconds to 0.23 seconds, while preserving every JSON value on our 300-case regression suite. That does not mean every answer is correct—it means the optimization preserved the baseline.

“The optimized MI300X path was also faster, but it changed 13 answers relative to its own baseline. Relay rejects that candidate. Moving from the original 4090 to this candidate changes 15 answers, so it also fails the migration check.

“Our fleet platform already connects and monitors worker machines. Paid marketplace matching and automatic deployment are the next steps. Today’s proof is the measured comparison and the decision to reject a faster result when it breaks the agreed output policy.”

Use the evidence panel’s own-GPU optimization and RTX 4090 migration modes separately. Do not claim the accepted A5000/L40S optimization is an accepted migration from the 4090. Do not present recorded results as a live GPU run.

## Interactive walkthrough

1. GPU Lab opens the original chat interface from `FE-WEB` commit `ccb01d8`. The conversation stays visible when Evidence opens.
2. In **Workflow**, type `Compare MI300X`. The reply summarizes the recorded decision and opens its evidence beside the chat.
3. Expand changed answers to inspect the original sentence and reference, candidate and expected fields. Switch to **From RTX 4090** to see 15 changed answers with the correct 4090 latency reference.
4. Type `Recheck A5000` to recompute its saved decision. This is a real file-based verification, not a fresh GPU run. The panel also exposes practical diagnostic steps.
5. With live resources connected using `-Live`, select a pod and use commands such as `Train baseline`, `Optimize training`, `Optimize inference`, or `Migrate RTX 4090 -> MI300X`. Each produces an inline plan with **Approve and run** before a GPU mutation. Progress, results and logs stay attached to the conversation.
6. Switch to **Model replies** to use the original streamed model chat, first-token/total timing, baseline/cache toggles and repeated-prompt comparisons. Load a model on the selected GPU and approve before sending inference requests.

Workflow accepts a bounded set of supported commands; it is not a general autonomous planner. `/help` lists them. Hard budgets, deadlines, multi-step goals and automatic deployment are explicitly unsupported rather than silently ignored. Detailed training, data generation, migration and job-history controls remain in the expandable Evidence panel. The default recorded mode keeps model replies and live mutations disabled until configured GPU resources are connected.

## Updated overview

**Relay — AI compute optimization with verified decisions.**

Relay combines GPU experiment workflows with a fleet platform for connected worker machines. The hackathon prototype focuses on Qwen training, measured inference comparisons, and explicit validation outcomes. Faster candidates are rejected when they change the agreed JSON outputs. Recorded results identify the checkpoint, evaluation suite and timing conditions.

Owner availability and resource controls are implemented in the fleet platform. Paid rentals, earnings, broader marketplace matching, QLoRA, Blender, DiLoCo and automatic traffic migration remain future work.

## Fixes and checks

- GPU comparisons now distinguish strict answer preservation from training's zero-regression policy. Missing/legacy evidence is not a pass; rejected checkpoints do not become the accepted latest model.
- Serving retains an active baseline until an explicit stop. Configuration/readiness checks, serialized prefix access and dynamic SSH forwarding fix the reviewed reload issues. This is explicit stop/start, not automatic rollback.
- Provider controls honor empty workload/day selections and overnight start days. Declined offers preserve execution retries, with backoff and durable accounting across restarts.
- GPU Lab preserves the original chat interface, with Workflow and Model replies modes, inline approval and job results, and expandable evidence. Ji’s vendor profiles (`551dd8a`) are integrated; capacity failures preserve running models and checkpoints instead of silently reclaiming them.
- Validation: 159 frontend tests; 81 focused GPU profile, placement, dashboard, safety and migration tests; 23 evidence/provenance backend tests; 173 main-backend tests (41 additional cases skipped); 26 provider/GUI tests; TypeScript/build and lint checks; and an actual isolated PostgreSQL refusal/retry regression passed. The 148-file recorded evidence verifier passed. HTTP checks confirmed the main session, fleet data, chat module, saved evidence and recheck endpoints through the frontend proxy.
- Live GPU workloads were not rerun. Browser automation was unavailable; HTTP/proxy and component tests passed. These results describe the reviewed integration snapshot across the three companion branches, not a new GPU campaign or production rollout.
