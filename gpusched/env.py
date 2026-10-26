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
    w_idle: float = 0.05  # for idle penalty
    w_complete: float = 1.0  # per completed job (throughput)
    w_miss: float = 1.5  # extra penalty per missed deadline
    w_late: float = 0.02  # lateness penalty
    w_wait: float = 0.02  # wait penalty for each pending job in queue
    w_starve: float = 0.01  # per-tick penalty per starving job
    starve_threshold: int = 30  # ticks waiting before a job counts as starving
    w_invalid: float = 0.1  # penalty for an invalid action (unmasked algos)


JOB_FEATS = 6
CLUSTER_FEATS = 4


class GPUClusterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_gpus: int = 16,
        k_queue: int = 8,
        n_mem: int = 384,
        workload: WorkloadConfig | None = None,
        rate_range: tuple[float, float] | None = None,
        reward: RewardConfig | None = None,
        max_ticks: int = 2000,
    ):
        super().__init__()
        self.n_gpus = n_gpus
        self.n_mem = n_mem

        self.K = k_queue
        self.wl_cfg = workload or WorkloadConfig()
        assert max(self.wl_cfg.gpu_choices) <= n_gpus
        assert (
            max(self.wl_cfg.gpu_choices) * max(self.wl_cfg.mem_per_gpu_choices) <= n_mem
        )
        self.rate_range = (
            rate_range  # sample arrival_rate per episode (for generalization)
        )
        self.rw = reward or RewardConfig()
        self.max_ticks = max_ticks

        self.action_space = spaces.Discrete(self.K + 1)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(CLUSTER_FEATS + JOB_FEATS * self.K,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        # start new episode
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.wl_cfg
        if self.rate_range is not None:
            lo, hi = self.rate_range
            cfg = WorkloadConfig(
                **{**cfg.__dict__, "arrival_rate": float(self._rng.uniform(lo, hi))}
            )
        self.jobs: list[Job] = generate_workload(cfg, self._rng)
        self.t = 0
        self.pending = list(self.jobs)  # not yet arrived
        self.queue: list[Job] = []  # arrived, waiting
        self.running: list[Job] = []
        self.done: list[Job] = []
        self.free = self.n_gpus
        self.free_mem = self.n_mem
        self._mem_util_hist: list[float] = []
        self._util_hist: list[float] = []
        self._admit()
        # keeping log of terms, to adjust penalty weights
        self._terms = dict.fromkeys(["idle", "starve", "wait", "late", "miss"], 0.0)
        return self._obs(), {}

    def step(self, action: int):
        # take a decision, note doesnt advance time
        # allows agent to start many jobs at once
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
        # do masking, set false to those jobs which cant be chosen
        # at all, so network doesnt try to choose them
        m = np.zeros(self.K + 1, dtype=bool)
        for i, job in enumerate(self.queue[: self.K]):
            m[i] = self._fits(job)
        m[self.K] = True
        return m

    # ------------------------------------------------------------ internals

    # if moving to per gpu memory, only edit these 3 functions
    def _fits(self, job: Job) -> bool:
        return job.gpus <= self.free and job.mem <= self.free_mem

    def _allocate(self, job: Job):
        self.free -= job.gpus
        self.free_mem -= job.mem

    def _release(self, job: Job):
        self.free += job.gpus
        self.free_mem += job.mem

    def _admit(self):
        # admit job
        while self.pending and self.pending[0].arrival <= self.t:
            self.queue.append(self.pending.pop(0))

    def _start(self, job: Job):
        # start a job
        self.queue.remove(job)
        job.start = self.t
        job.finish = self.t + job.runtime
        self._allocate(job)
        self.running.append(job)

    def _tick(self) -> float:
        """Advance one tick; return the shaped reward for that tick."""
        # advance time

        r = 0.0
        busy = self.n_gpus - self.free
        self._util_hist.append(busy / self.n_gpus)
        self._mem_util_hist.append((self.n_mem - self.free_mem) / self.n_mem)
        # idle penalty, if gpu free and job can be put on it but not there
        idle = 0.0
        if any(self._fits(j) for j in self.queue[: self.K]):
            idle = self.rw.w_idle * (self.free / self.n_gpus)

        # starving penalty for jobs ignored for long
        starve = self.rw.w_starve * sum(
            1 for j in self.queue if self.t - j.arrival > self.rw.starve_threshold
        )

        # wait penalty for each waiting job in queue
        wait = self.rw.w_wait * len(self.queue)

        r -= idle + starve + wait
        self._terms["idle"] -= idle
        self._terms["starve"] -= starve
        self._terms["wait"] -= wait

        self.t += 1
        for job in [j for j in self.running if j.finish <= self.t]:
            self.running.remove(job)
            self._release(job)
            self.done.append(job)
            # reward for job done - lateness
            r += self.rw.w_complete
            late = self.rw.w_late * max(0, job.finish - job.deadline)
            r -= late
            self._terms["late"] -= late
            if job.finish > job.deadline:
                r -= self.rw.w_miss
                self._terms["miss"] -= self.rw.w_miss
        self._admit()
        return r

    def _obs(self) -> np.ndarray:
        # build the 43 numbers that descibr our current state
        max_rt = self.wl_cfg.max_runtime
        if self.running:
            next_free = min(j.start + j.est_runtime - self.t for j in self.running)
            next_free = float(np.clip(next_free, 0, max_rt)) / max_rt
        else:
            next_free = 0.0
        cluster = [
            self.free / self.n_gpus,
            self.free_mem / self.n_mem,
            min(len(self.queue), 50) / 50.0,
            next_free,
        ]

        feats = []
        for i in range(self.K):
            if i < len(self.queue):
                j = self.queue[i]
                slack = (j.deadline - self.t - j.est_runtime) / max_rt
                feats += [
                    j.gpus / self.n_gpus,
                    j.mem / self.n_mem,
                    j.est_runtime / max_rt,
                    float(np.clip(slack, -1, 1)),
                    min(self.t - j.arrival, 100) / 100.0,
                    1.0,
                ]
            else:
                feats += [0.0] * JOB_FEATS
        return np.asarray(cluster + feats, dtype=np.float32)

    def _metrics(self) -> dict:
        # metrics for the run
        done = self.done
        n = max(len(done), 1)

        return {
            "jobs_done": len(done),
            "jobs_total": len(self.jobs),
            "avg_jct": (
                float(np.mean([j.finish - j.arrival for j in done]))
                if done
                else float("nan")
            ),
            "miss_rate": sum(j.finish > j.deadline for j in done) / n,
            "max_wait": max([j.start - j.arrival for j in done], default=0),
            "avg_util": float(np.mean(self._util_hist)) if self._util_hist else 0.0,
            "avg_mem_util": (
                float(np.mean(self._mem_util_hist)) if self._mem_util_hist else 0.0
            ),
            "makespan": self.t,
            **{f"r_{k}": v for k, v in self._terms.items()},
        }


"""
# waiting cost: total over an episode ~ sum of wait times (JCT proxy)
r -= self.rw.w_wait * len(self.queue)
# idle cost: GPUs sitting free while a queued job could use them
if any(j.gpus <= self.free for j in self.queue[: self.K]):
    r -= self.rw.w_idle * (self.free / self.n_gpus)
"""
"""
- **Util term (`w_util`):** replace the flat bonus with an idle penalty. 
    Subtract `w_idle x free/n_gpus`, but only on ticks where some job in the 
    first K queue slots fits in the free GPUs. Keep `_util_hist.append` for metrics.

- **Completion term (`w_complete`):** make it depend on turnaround. On each completion, 
    subtract `w_flow x (finish - arrival) / max(runtime, 1)`. Keep a small flat base 
    if you want.

- **Waiting term (optional but recommended):** subtract `w_wait x len(queue)` every tick. 
    It is a dense turnaround signal, so use either this or the slowdown term at full weight, 
    or both at small weights. Don't stack both at full strength.

- **Keep:** the deadline-miss penalty. Reconsider the starvation penalty, 
    since the waiting term overlaps with it.

- **`RewardConfig`:** add `w_idle`, `w_flow` and `w_wait`. Drop `w_util` once it's unused.

- **Before training:** print per-episode totals of each reward term under FIFO and SJF, 
    and adjust the weights until no term is about 10 times bigger than the rest.

- **After changing:** retrain from scratch. `ppo_v1` was trained on the old reward, 
    so it isn't comparable.
"""
