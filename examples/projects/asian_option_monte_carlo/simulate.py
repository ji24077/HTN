"""Upload this file and ask: Run 360 independent trials and return the mean discounted
payoff with its standard error.

A heavy Monte Carlo case for the distribution pipeline. Each trial prices an
arithmetic-average Asian call by simulating PATHS_PER_TRIAL geometric Brownian
motion paths of STEPS daily steps each, all from one seeded generator, and returns
that trial's discounted mean payoff. One trial is a few seconds of pure-Python
work; a few hundred trials is a quarter of an hour on one core, which is enough
for the planner to prefer splitting the trials across machines.

Each trial is independently reproducible from its seed, so trials can run on any
machine in any order and the aggregate does not depend on where they ran.
"""
import argparse
import json
import math
import random

PATHS_PER_TRIAL = 5000
STEPS = 2000
SPOT = 100.0
STRIKE = 105.0
RATE = 0.03
VOLATILITY = 0.25
MATURITY_YEARS = 1.0


def simulate(seed):
    rng = random.Random(seed)
    dt = MATURITY_YEARS / STEPS
    drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * dt
    shock = VOLATILITY * math.sqrt(dt)
    total_payoff = 0.0
    for _ in range(PATHS_PER_TRIAL):
        price = SPOT
        running_sum = 0.0
        for _ in range(STEPS):
            price *= math.exp(drift + shock * rng.gauss(0.0, 1.0))
            running_sum += price
        total_payoff += max(running_sum / STEPS - STRIKE, 0.0)
    return math.exp(-RATE * MATURITY_YEARS) * total_payoff / PATHS_PER_TRIAL


def summarize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "trials": n,
        "paths": n * PATHS_PER_TRIAL,
        "mean_payoff": mean,
        "standard_error": math.sqrt(variance / n) if n > 1 else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    values = [simulate(args.seed + trial) for trial in range(args.trials)]
    print(json.dumps(summarize(values)))
