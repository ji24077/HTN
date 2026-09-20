"""N-2 (Jack). The blocker. Do NOT try NCCL - it fails over Tailscale across NAT
(NVIDIA/nccl #1606, closed "not planned").

    # machine 0
    RANK=0 WORLD=2 MASTER=100.64.0.1 python scripts/gloo_check.py
    # machine 1
    RANK=1 WORLD=2 MASTER=100.64.0.1 python scripts/gloo_check.py

The reported seconds scale to T_sync. Hand that number to Ji - it feeds compute_H().
"""

import os
import time

import torch
import torch.distributed as dist

os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ.get("IFACE", "tailscale0"))

dist.init_process_group(
    "gloo",
    init_method=f"tcp://{os.environ['MASTER']}:29500",
    rank=int(os.environ["RANK"]),
    world_size=int(os.environ["WORLD"]),
)

t = torch.ones(25_000_000)          # 100MB fp32
dist.all_reduce(t)                  # warmup
t0 = time.time()
dist.all_reduce(t)
dt = time.time() - t0

print(f"100MB all-reduce: {dt:.2f}s  ->  {100 / dt:.1f} MB/s")
print(f"for a 500MB model, T_sync ~= {dt * 5:.1f}s   <-- give this to Ji")
dist.destroy_process_group()
