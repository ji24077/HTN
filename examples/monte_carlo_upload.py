"""Upload this file and ask: Run 100 independent trials and return the hit rate.

A trial draws one point uniformly in the unit square and returns whether it is
inside the quarter unit circle. Each trial is independently reproducible.
"""
import argparse
import json
import random


def simulate(seed):
    rng = random.Random(seed)
    x, y = rng.random(), rng.random()
    return int(x * x + y * y <= 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hits = sum(simulate(args.seed + trial) for trial in range(args.trials))
    print(json.dumps({"trials": args.trials, "hits": hits, "hit_rate": hits / args.trials}))
