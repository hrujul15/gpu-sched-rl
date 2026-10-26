"""Job model and synthetic workload generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Job:
    jid: int
    arrival: int
    gpus: int
    mem: int            # total memory across requested GPUs (GB); mem_per_gpu is mem / gpus
    runtime: int        # TRUE runtime (hidden from the agent)
    est_runtime: int    # noisy user estimate (visible to agent and baselines)
    deadline: int       # absolute tick
    start: int = -1
    finish: int = -1
    gpu_ids: tuple[int, ...] = ()

    @property
    def mem_per_gpu(self) -> int:
        return self.mem // self.gpus


@dataclass
class WorkloadConfig:
    n_jobs: int = 60
    arrival_rate: float = 0.5          # mean jobs per tick (Poisson process)
    gpu_choices: tuple = (1, 2, 4, 8)
    gpu_probs: tuple = (0.5, 0.25, 0.15, 0.10)
    mem_per_gpu_choices: tuple = (8, 16, 24, 32, 40)  # GB required on each GPU
    mem_per_gpu_probs: tuple = (0.20, 0.25, 0.25, 0.20, 0.10)
    runtime_mu: float = 2.5            # lognormal params for true runtime
    runtime_sigma: float = 1.0
    max_runtime: int = 100
    est_noise: float = 0.3             # lognormal sigma of the estimate error
    slack_low: float = 1.5             # deadline = arrival + runtime * U(low, high)
    slack_high: float = 4.0
    # Fraction of jobs whose deadline is decoupled from runtime: absolute slack
    # U(abs_slack_low, abs_slack_high) ticks beyond the true runtime. 0.0 keeps
    # the original behaviour (deadline always proportional to runtime).
    decoupled_frac: float = 0.0
    abs_slack_low: float = 2.0
    abs_slack_high: float = 40.0


def workload_preset(name: str, arrival_rate: float = 0.5) -> WorkloadConfig:
    """Named workloads. 'hard' is built so that SJF is no longer near-optimal:
    deadlines decoupled from runtime, more wide/high-memory jobs (fragmentation
    and starvation of wide jobs), and noisier runtime estimates."""
    if name == "default":
        return WorkloadConfig(arrival_rate=arrival_rate)
    if name == "hard":
        return WorkloadConfig(
            arrival_rate=arrival_rate,
            gpu_probs=(0.40, 0.25, 0.20, 0.15),
            mem_per_gpu_probs=(0.15, 0.20, 0.25, 0.25, 0.15),
            est_noise=0.6,
            decoupled_frac=0.5,
            abs_slack_low=0.0,
            abs_slack_high=15.0,
        )
    raise ValueError(f"unknown workload preset: {name}")


def generate_workload(cfg: WorkloadConfig, rng: np.random.Generator) -> list[Job]:
    """Synthetic jobs sorted by arrival. TODO: swap in / mix with real traces."""
    gaps = rng.exponential(1.0 / cfg.arrival_rate, size=cfg.n_jobs)
    arrivals = np.floor(np.cumsum(gaps)).astype(int)
    jobs = []
    for i in range(cfg.n_jobs):
        gpus = int(rng.choice(cfg.gpu_choices, p=cfg.gpu_probs))
        mem = gpus * int(rng.choice(cfg.mem_per_gpu_choices, p=cfg.mem_per_gpu_probs))
        rt = int(np.clip(round(rng.lognormal(cfg.runtime_mu, cfg.runtime_sigma)), 1, cfg.max_runtime))
        est = int(np.clip(round(rt * rng.lognormal(0.0, cfg.est_noise)), 1, cfg.max_runtime))
        if cfg.decoupled_frac > 0 and rng.random() < cfg.decoupled_frac:
            deadline = int(
                arrivals[i] + rt + np.ceil(rng.uniform(cfg.abs_slack_low, cfg.abs_slack_high))
            )
        else:
            slack = rng.uniform(cfg.slack_low, cfg.slack_high)
            deadline = int(arrivals[i] + np.ceil(rt * slack))
        jobs.append(Job(i, int(arrivals[i]), gpus, mem, rt, est, deadline))
    return jobs

