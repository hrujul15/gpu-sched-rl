"""Gymnasium environment: online scheduling of training jobs on a shared GPU pool.

Design notes
- Time advances in integer ticks. Starting a job is a zero-time decision, so the
  agent can start several jobs in one tick; the WAIT action advances the clock by 1.
- Action space: Discrete(K+1). 0..K-1 = start the i-th job of the queue window
  (queue ordered by arrival), K = wait. Use `action_masks()` with MaskablePPO.
- The agent only sees *estimated* runtimes, never the true ones.
- Single resource (GPU count). TODO: add GPU memory as a second resource.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .jobs import Job, WorkloadConfig, generate_workload


@dataclass
class RewardConfig:
    w_util: float = 0.02          # per-tick bonus * utilization
    w_complete: float = 1.0       # per completed job (throughput)
    w_miss: float = 2.0           # extra penalty per missed deadline
    w_starve: float = 0.01        # per-tick penalty per starving job
    starve_threshold: int = 30    # ticks waiting before a job counts as starving
    w_invalid: float = 0.1        # penalty for an invalid action (unmasked algos)


JOB_FEATS = 5
CLUSTER_FEATS = 3


class GPUClusterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_gpus: int = 16,
        k_queue: int = 8,
        workload: WorkloadConfig | None = None,
        rate_range: tuple[float, float] | None = None,
        reward: RewardConfig | None = None,
        max_ticks: int = 2000,
    ):
        super().__init__()
        self.n_gpus = n_gpus
        self.K = k_queue
        self.wl_cfg = workload or WorkloadConfig()
        assert max(self.wl_cfg.gpu_choices) <= n_gpus
        self.rate_range = rate_range  # sample arrival_rate per episode (for generalization)
        self.rw = reward or RewardConfig()
        self.max_ticks = max_ticks

        self.action_space = spaces.Discrete(self.K + 1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(CLUSTER_FEATS + JOB_FEATS * self.K,), dtype=np.float32
        )
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.wl_cfg
        if self.rate_range is not None:
            lo, hi = self.rate_range
            cfg = WorkloadConfig(**{**cfg.__dict__, "arrival_rate": float(self._rng.uniform(lo, hi))})
        self.jobs: list[Job] = generate_workload(cfg, self._rng)
        self.t = 0
        self.pending = list(self.jobs)      # not yet arrived
        self.queue: list[Job] = []          # arrived, waiting
        self.running: list[Job] = []
        self.done: list[Job] = []
        self.free = self.n_gpus
        self._util_hist: list[float] = []
        self._admit()
        return self._obs(), {}

    def step(self, action: int):
        reward = 0.0
        valid = self.action_masks()
        if action < self.K and valid[action]:
            self._start(self.queue[action])
        elif action == self.K:
            reward += self._tick()
        else:  # invalid action: treat as wait, with a penalty
            reward += self._tick() - self.rw.w_invalid

        terminated = not (self.pending or self.queue or self.running)
        truncated = (not terminated) and self.t >= self.max_ticks
        info = self._metrics() if (terminated or truncated) else {}
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        m = np.zeros(self.K + 1, dtype=bool)
        for i, job in enumerate(self.queue[: self.K]):
            m[i] = job.gpus <= self.free
        m[self.K] = True
        return m

    # ------------------------------------------------------------ internals
    def _admit(self):
        while self.pending and self.pending[0].arrival <= self.t:
            self.queue.append(self.pending.pop(0))

    def _start(self, job: Job):
        self.queue.remove(job)
        job.start = self.t
        job.finish = self.t + job.runtime
        self.free -= job.gpus
        self.running.append(job)

    def _tick(self) -> float:
        """Advance one tick; return the shaped reward for that tick."""
        r = 0.0
        busy = self.n_gpus - self.free
        self._util_hist.append(busy / self.n_gpus)
        r += self.rw.w_util * (busy / self.n_gpus)
        r -= self.rw.w_starve * sum(1 for j in self.queue if self.t - j.arrival > self.rw.starve_threshold)

        self.t += 1
        for job in [j for j in self.running if j.finish <= self.t]:
            self.running.remove(job)
            self.free += job.gpus
            self.done.append(job)
            r += self.rw.w_complete
            if job.finish > job.deadline:
                r -= self.rw.w_miss
        self._admit()
        return r

    def _obs(self) -> np.ndarray:
        max_rt = self.wl_cfg.max_runtime
        if self.running:
            next_free = min(j.start + j.est_runtime - self.t for j in self.running)
            next_free = float(np.clip(next_free, 0, max_rt)) / max_rt
        else:
            next_free = 0.0
        cluster = [self.free / self.n_gpus, min(len(self.queue), 50) / 50.0, next_free]

        feats = []
        for i in range(self.K):
            if i < len(self.queue):
                j = self.queue[i]
                slack = (j.deadline - self.t - j.est_runtime) / max_rt
                feats += [
                    j.gpus / self.n_gpus,
                    j.est_runtime / max_rt,
                    float(np.clip(slack, -1, 1)),
                    min(self.t - j.arrival, 100) / 100.0,
                    1.0,
                ]
            else:
                feats += [0.0] * JOB_FEATS
        return np.asarray(cluster + feats, dtype=np.float32)

    def _metrics(self) -> dict:
        done = self.done
        n = max(len(done), 1)
        return {
            "jobs_done": len(done),
            "jobs_total": len(self.jobs),
            "avg_jct": float(np.mean([j.finish - j.arrival for j in done])) if done else float("nan"),
            "miss_rate": sum(j.finish > j.deadline for j in done) / n,
            "max_wait": max([j.start - j.arrival for j in done], default=0),
            "avg_util": float(np.mean(self._util_hist)) if self._util_hist else 0.0,
            "makespan": self.t,
        }
