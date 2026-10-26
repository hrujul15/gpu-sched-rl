"""Heuristic policies. Each takes the env and returns an action index.
They read env.queue / env.running directly but only use ESTIMATED runtimes."""

from __future__ import annotations


def _wait(env) -> int:
    return env.wait_action


def _pack(env, index: int) -> int:
    return env.encode_action(index, "pack")


def _fitting(env):
    return [(i, j) for i, j in enumerate(env.queue[: env.K]) if env._fits(j)]


def fifo(env) -> int:
    """Strict FIFO: only the head of the queue may start (head-of-line blocking)."""
    if env.queue and env._fits(env.queue[0]):
        return _pack(env, 0)
    return _wait(env)


def sjf(env) -> int:
    """Shortest (estimated) job first among jobs in the visible window that fit."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)
    return _pack(env, min(cand, key=lambda x: x[1].est_runtime)[0])


def edf(env) -> int:
    """Earliest deadline first among jobs that fit."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)
    return _pack(env, min(cand, key=lambda x: x[1].deadline)[0])


def backfill(env) -> int:
    """FIFO with backfilling if the head job is expected to fit after releases."""
    if not env.queue:
        return _wait(env)
    head = env.queue[0]
    if env._fits(head):
        return _pack(env, 0)

    # Simulate memory releases in estimated completion order. A released job
    # returns memory to the exact GPUs on which it was placed.
    simulated = env.gpu_free_mem.copy()
    shadow = env.t
    for running in sorted(env.running, key=lambda j: (j.start + j.est_runtime, j.jid)):
        for gpu in running.gpu_ids:
            simulated[gpu] += running.mem_per_gpu
        shadow = running.start + running.est_runtime
        if int((simulated >= head.mem_per_gpu).sum()) >= head.gpus:
            break
    else:
        # If estimates never predict a feasible placement, don't backfill on
        # a made-up reservation time; normal SJF-like backfill is safer.
        shadow = env.t

    for i, job in _fitting(env):
        if i != 0 and env.t + job.est_runtime <= shadow:
            return _pack(env, i)
    return _wait(env)


def bestfit(env) -> int:
    """Choose the job whose PACK placement leaves the least residual memory on selected GPUs."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)

    def placement_waste(item):
        _, job = item
        chosen = env._choose_gpus(job, 0)
        residual = sum(env.gpu_free_mem[gpu] - job.mem_per_gpu for gpu in chosen)
        # Prefer tighter fits; lightly break ties in favor of shorter jobs.
        return residual / env.n_mem, job.est_runtime

    return _pack(env, min(cand, key=placement_waste)[0])


BASELINES = {
    "fifo": fifo,
    "sjf": sjf,
    "edf": edf,
    "backfill": backfill,
    "bestfit": bestfit,
}
