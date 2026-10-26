"""Gymnasium environment: online scheduling of training jobs on a shared GPU pool.

Design notes
- Time advances in integer ticks. Starting a job is a zero-time decision, so the
  agent can start several jobs in one tick; the WAIT action advances the clock by 1.
- Action space: Discrete(2*K+1). For each visible job, choose PACK or SPREAD
  placement across physical GPUs; the final action waits one tick.
- Every GPU has its own memory capacity. Multiple jobs may share a GPU when each
  has enough remaining memory, so placement and fragmentation matter.
- The agent only sees *estimated* runtimes, never the true ones.
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
    w_miss: float = 2.0  # extra penalty per missed deadline (↑ from 1.5)
    w_late: float = 0.03  # lateness penalty (↑ from 0.02)
    w_wait: float = 0.015  # wait penalty for each pending job (↓ from 0.02)
    w_starve: float = 0.01  # per-tick penalty per starving job
    starve_threshold: int = 30  # ticks waiting before a job counts as starving
    w_invalid: float = 0.1  # penalty for an invalid action (unmasked algos)
    w_flow: float = 0.5  # penalise turnaround time on completion


JOB_FEATS = 12
BASE_CLUSTER_FEATS = 4
PACK = 0
SPREAD = 1


class GPUClusterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_gpus: int = 16,
        k_queue: int = 8,
        gpu_mem_gb: int = 40,
        n_mem: int | None = None,
        workload: WorkloadConfig | None = None,
        rate_range: tuple[float, float] | None = None,
        reward: RewardConfig | None = None,
        max_ticks: int = 2000,
        hide_rank_feats: bool = False,
    ):
        super().__init__()
        self.n_gpus = n_gpus
        # n_mem is accepted as a backwards-compatible total-capacity override.
        if n_mem is not None:
            if n_mem % n_gpus != 0:
                raise ValueError("n_mem must be divisible by n_gpus")
            gpu_mem_gb = n_mem // n_gpus
        self.gpu_mem_gb = int(gpu_mem_gb)
        self.n_mem = self.n_gpus * self.gpu_mem_gb
        if self.gpu_mem_gb <= 0:
            raise ValueError("gpu_mem_gb must be positive")

        self.K = k_queue
        self.wl_cfg = workload or WorkloadConfig()
        assert max(self.wl_cfg.gpu_choices) <= n_gpus
        assert max(self.wl_cfg.mem_per_gpu_choices) <= self.gpu_mem_gb, (
            "Every per-GPU memory request must fit on one GPU"
        )
        self.rate_range = (
            rate_range  # sample arrival_rate per episode (for generalization)
        )
        self.rw = reward or RewardConfig()
        self.max_ticks = max_ticks
        # Ablation: zero the precomputed SJF-rank / relative-runtime features so
        # the agent must derive any ordering itself from raw est_runtime.
        self.hide_rank_feats = hide_rank_feats

        self.action_space = spaces.Discrete(2 * self.K + 1)
        self.wait_action = 2 * self.K
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(BASE_CLUSTER_FEATS + self.n_gpus + JOB_FEATS * self.K,),
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
        self.gpu_free_mem = np.full(self.n_gpus, self.gpu_mem_gb, dtype=np.int32)
        self.free = self.n_gpus  # number of completely unused GPUs
        self.free_mem = self.n_mem
        self._mem_util_hist: list[float] = []
        self._util_hist: list[float] = []
        self._admit()
        # keeping log of terms, to adjust penalty weights
        self._terms = dict.fromkeys(["idle", "starve", "wait", "late", "miss", "flow"], 0.0)
        return self._obs(), {}

    def step(self, action: int):
        # take a decision, note doesnt advance time
        # allows agent to start many jobs at once
        reward = 0.0
        valid = self.action_masks()
        if 0 <= action < 2 * self.K and valid[action]:
            job_index, strategy = divmod(int(action), 2)
            self._start(self.queue[job_index], strategy)
        elif action == self.wait_action:
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
        m = np.zeros(2 * self.K + 1, dtype=bool)
        for i, job in enumerate(self.queue[: self.K]):
            if self._fits(job):
                m[2 * i + PACK] = True
                m[2 * i + SPREAD] = True
        m[self.wait_action] = True
        return m

    # ------------------------------------------------------------ internals

    def _fits(self, job: Job) -> bool:
        """A job fits when at least `job.gpus` physical GPUs have enough memory."""
        return int(np.count_nonzero(self.gpu_free_mem >= job.mem_per_gpu)) >= job.gpus

    def encode_action(self, queue_index: int, strategy: str = "pack") -> int:
        """Encode a visible queue index and deterministic placement strategy."""
        strategy_id = PACK if strategy == "pack" else SPREAD
        return 2 * int(queue_index) + strategy_id

    def _choose_gpus(self, job: Job, strategy: int) -> tuple[int, ...]:
        eligible = np.flatnonzero(self.gpu_free_mem >= job.mem_per_gpu).tolist()
        if strategy == PACK:
            # Consume the tightest-fitting GPUs first, preserving large free blocks.
            eligible.sort(key=lambda gpu: (self.gpu_free_mem[gpu] - job.mem_per_gpu, gpu))
        else:
            # Spread onto the roomiest GPUs first; deterministic ties by GPU id.
            eligible.sort(key=lambda gpu: (-self.gpu_free_mem[gpu], gpu))
        if len(eligible) < job.gpus:
            raise RuntimeError("Placement requested for a job that does not fit")
        return tuple(eligible[:job.gpus])

    def _refresh_capacity_counters(self):
        self.free = int(np.count_nonzero(self.gpu_free_mem == self.gpu_mem_gb))
        self.free_mem = int(self.gpu_free_mem.sum())

    def _allocate(self, job: Job, strategy: int):
        job.gpu_ids = self._choose_gpus(job, strategy)
        for gpu in job.gpu_ids:
            self.gpu_free_mem[gpu] -= job.mem_per_gpu
        self._refresh_capacity_counters()

    def _release(self, job: Job):
        for gpu in job.gpu_ids:
            self.gpu_free_mem[gpu] += job.mem_per_gpu
            if self.gpu_free_mem[gpu] > self.gpu_mem_gb:
                raise RuntimeError("GPU memory accounting exceeded capacity")
        self._refresh_capacity_counters()

    def _admit(self):
        # admit job
        while self.pending and self.pending[0].arrival <= self.t:
            self.queue.append(self.pending.pop(0))

    def _start(self, job: Job, strategy: int = PACK):
        # Start a job and reserve the selected physical GPUs' memory.
        self.queue.remove(job)
        job.start = self.t
        job.finish = self.t + job.runtime
        self._allocate(job, strategy)
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
            idle = self.rw.w_idle * (self.free_mem / self.n_mem)

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
            # reward for job done - flow time - lateness
            r += self.rw.w_complete
            # flow-time penalty: shorter turnaround → less penalty → favours SJF
            flow = self.rw.w_flow * (job.finish - job.arrival) / self.wl_cfg.max_runtime
            r -= flow
            self._terms["flow"] -= flow
            late = self.rw.w_late * max(0, job.finish - job.deadline)
            r -= late
            self._terms["late"] -= late
            if job.finish > job.deadline:
                r -= self.rw.w_miss
                self._terms["miss"] -= self.rw.w_miss
        self._admit()
        return r

    def _obs(self) -> np.ndarray:
        # 4 cluster + n_gpus per-device memory fractions + 12*8 job features
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
        # Per-device free-memory fractions expose fragmentation directly.
        gpu_memory = (self.gpu_free_mem / self.gpu_mem_gb).astype(np.float32).tolist()

        # Precompute rank and mean runtime for jobs in the visible window
        window = self.queue[: self.K]
        n_win = len(window)
        if n_win > 0:
            est_rts = [j.est_runtime for j in window]
            sorted_order = np.argsort(est_rts)
            ranks = np.empty(n_win)
            for rank, idx in enumerate(sorted_order):
                ranks[idx] = rank
            mean_rt = max(float(np.mean(est_rts)), 1.0)
        else:
            ranks = np.array([])
            mean_rt = 1.0

        feats = []
        for i in range(self.K):
            if i < n_win:
                j = window[i]
                slack = (j.deadline - self.t - j.est_runtime) / max_rt
                feats += [
                    j.gpus / self.n_gpus,
                    j.mem_per_gpu / self.gpu_mem_gb,
                    j.est_runtime / max_rt,
                    float(np.clip(slack, -1, 1)),
                    min(self.t - j.arrival, 100) / 100.0,
                    1.0,
                    # ── new SJF-informative features ──
                    0.0 if self.hide_rank_feats else ranks[i] / max(n_win - 1, 1),
                    0.0
                    if self.hide_rank_feats
                    else float(np.clip(j.est_runtime / mean_rt - 1.0, -1, 1)),
                    1.0 if self._fits(j) else 0.0,
                    # Deadline slack before execution, normalized to runtime scale.
                    float(np.clip((j.deadline - self.t) / max_rt, -1, 1)),
                    # Estimated runtime per requested GPU.
                    float(np.clip((j.est_runtime / max(j.gpus, 1)) / max_rt, 0, 1)),
                    # Normalized time already spent waiting in the queue.
                    float(np.clip((self.t - j.arrival) / max_rt, 0, 1)),
                ]
            else:
                feats += [0.0] * JOB_FEATS
        return np.asarray(cluster + gpu_memory + feats, dtype=np.float32)

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