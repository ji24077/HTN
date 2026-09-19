"""Build an evidence-free hardware inventory and budget plan from saved catalog reads.

This command does not rent pods. RunPod stock/runtime compatibility must be
rechecked by the bounded session immediately before any paid create request.
"""

import argparse
import json
from pathlib import Path

from gpushare.agent.gpu_matrix import NEW_TARGET_KEYS, expansion_plan, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secure", type=Path, required=True)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--prior-reserve-usd", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshots = {"SECURE": json.loads(args.secure.read_text(encoding="utf-8")),
                 "COMMUNITY": json.loads(args.community.read_text(encoding="utf-8"))}
    report = {"inventory": inventory(snapshots),
              "budget": expansion_plan(NEW_TARGET_KEYS, prior_reserve_usd=args.prior_reserve_usd)}
    # Exclusive write preserves dated plans and prevents relabelling old results.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"out": str(args.out), "targets": len(report["inventory"]["targets"]),
                      "budget": report["budget"]}))


if __name__ == "__main__":
    main()
