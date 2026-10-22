"""Job model and synthetic workload generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Job:
    jid: int
    arrival: int
    gpus: int
    runtime: int        # TRUE runtime (hidden from the agent)
    est_runtime: int    # noisy user estimate (visible to agent and baselines)
    deadline: int       # absolute tick
    start: int = -1
    finish: int = -1


@dataclass
class WorkloadConfig:
    n_jobs: int = 60
    arrival_rate: float = 0.5          # mean jobs per tick (Poisson process)
    gpu_choices: tuple = (1, 2, 4, 8)
    gpu_probs: tuple = (0.5, 0.25, 0.15, 0.10)
    runtime_mu: float = 2.5            # lognormal params for true runtime
    runtime_sigma: float = 1.0
    max_runtime: int = 100
    est_noise: float = 0.3             # lognormal sigma of the estimate error
    slack_low: float = 1.5             # deadline = arrival + runtime * U(low, high)
    slack_high: float = 4.0


def generate_workload(cfg: WorkloadConfig, rng: np.random.Generator) -> list[Job]:
    """Synthetic jobs sorted by arrival. TODO: swap in / mix with real traces."""
    gaps = rng.exponential(1.0 / cfg.arrival_rate, size=cfg.n_jobs)
    arrivals = np.floor(np.cumsum(gaps)).astype(int)
    jobs = []
    for i in range(cfg.n_jobs):
        gpus = int(rng.choice(cfg.gpu_choices, p=cfg.gpu_probs))
        rt = int(np.clip(round(rng.lognormal(cfg.runtime_mu, cfg.runtime_sigma)), 1, cfg.max_runtime))
        est = int(np.clip(round(rt * rng.lognormal(0.0, cfg.est_noise)), 1, cfg.max_runtime))
        slack = rng.uniform(cfg.slack_low, cfg.slack_high)
        deadline = int(arrivals[i] + np.ceil(rt * slack))
        jobs.append(Job(i, int(arrivals[i]), gpus, rt, est, deadline))
    return jobs


def load_trace_csv(path: str) -> list[Job]:
    """TODO: parse a public cluster trace (e.g. Philly / Helios / Alibaba PAI)
    into Job objects: arrival tick, gpus, runtime; synthesize est_runtime/deadline."""
    raise NotImplementedError
