# Heavy Monte Carlo example

A simulation that is expensive enough for the planner to split it across machines.
Each trial prices an arithmetic-average Asian call from 5,000 simulated price paths
of 2,000 steps, about two seconds of pure-Python work per trial. The default 360
trials is roughly 12.5 minutes on one Apple M-series core and longer on slower CPUs.
Standard library only; nothing to install on the worker.

Upload `simulate.py` through **New job → Upload project** and request:

> Run 360 independent trials of simulate(seed) and return the mean discounted payoff
> with its standard error, exactly as summarize() in the file computes them. Each trial
> takes about two seconds of CPU; distribute the trials across the available workers
> when that finishes sooner than one machine would.

The simulation pipeline needs at least two connected CPU Python workers for its
independent validation gate, and it places batches on up to four. Start extra local
workers with distinct `WORKER_ID` values if the fleet has fewer.

## Measured on 2026-09-20

Four local CPU workers on one Mac, backend at commit `d40f5f2`:

| Stage | Time |
| --- | --- |
| profiling and local validation | 1 min 40 s |
| two-worker independent validation | 50 s |
| four batches of 90 trials, concurrent | 4 min 15 s |
| aggregation and result | 10 s |
| end to end | 7 min 6 s |

The planner measured 2.07 s per trial and chose four concurrent batches over a
single 744-second run. Result: mean payoff 4.2764 with standard error 0.0062 over
1.8 million paths.

`examples/asian_option_pipeline_check.py` submits this project to a running backend,
follows the phases, and fails if the batches were not spread across at least two
workers. It prints the end-to-end time.
