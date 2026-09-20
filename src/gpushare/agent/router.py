"""Ji - S-4, S-5, S-9. Pool-level decisions: what to rent, H, stragglers, migration."""

import statistics

from gpushare.agent.chips import REGISTRY, ChipAgent, DefaultAgent


def compute_H(t_sync: float, t_step: float, rho: float = 0.05) -> int:
    """H >= (T_sync / T_step) * (1 - rho) / rho, clamped to the DiLoCo range.

    The signature lever: the one setting whose value responds to MEASURED
    network conditions. Jack hands you t_sync from his gloo test.
    """
    h = int(t_sync / t_step * (1 - rho) / rho)
    return max(50, min(500, h))


def h_at_ceiling(t_sync: float, t_step: float, rho: float = 0.05) -> bool:
    """True -> warn the user: network too slow, recommend a smaller model."""
    return int(t_sync / t_step * (1 - rho) / rho) > 500


def pick_chips(min_vram: float, want: int, market: list) -> list:
    """Cheapest chips that are actually SUFFICIENT - not the fastest."""
    fits = [c for c in market if c.vram_gb >= min_vram and c.cc >= 7.0 and c.available]
    return sorted(fits, key=lambda c: c.credits_per_hour)[:want]


class Router:
    def agent_for(self, worker) -> ChipAgent:
        return REGISTRY.get(worker.chip_class, DefaultAgent())

    def check_stragglers(self, workers) -> None:
        med = statistics.median(w.t_step for w in workers if w.trainable)
        for w in workers:
            if w.t_step > 3 * med:
                w.role = "preprocess"          # a GTX 970 drags the whole round
            elif w.t_step > 1.2 * med:
                w.micro_batch = int(w.micro_batch * med / w.t_step)
        # WARNING: uneven batches -> the outer average must be weighted by each
        # worker's sample count. Agreed with Ethan. Do not skip this.
