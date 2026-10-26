"""Heuristic policies. Each takes the env and returns an action index.
They read env.queue / env.running directly but only use ESTIMATED runtimes."""

from __future__ import annotations


def _wait(env) -> int:
    return env.K


def _fitting(env):
    return [(i, j) for i, j in enumerate(env.queue[: env.K]) if env._fits(j)]


def fifo(env) -> int:
    """Strict FIFO: only the head of the queue may start (head-of-line blocking)."""
    if env.queue and env._fits(env.queue[0]):
        return 0
    return _wait(env)


def sjf(env) -> int:
    """Shortest (estimated) job first among jobs in the visible window that fit."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)
    return min(cand, key=lambda x: x[1].est_runtime)[0]


def edf(env) -> int:
    """Earliest deadline first among jobs that fit."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)
    return min(cand, key=lambda x: x[1].deadline)[0]


def backfill(env) -> int:
    """EASY-style backfilling: FIFO, but let later jobs jump ahead if they are
    estimated to finish before the blocked head job could start."""
    if not env.queue:
        return _wait(env)
    head = env.queue[0]
    if env._fits(head):
        return 0
    # shadow time: earliest (estimated) time enough GPUs free up for the head
    avail_g, avail_m, shadow = env.free, env.free_mem, env.t
    for j in sorted(env.running, key=lambda j: j.start + j.est_runtime):
        avail_g += j.gpus
        avail_m += j.mem
        shadow = j.start + j.est_runtime
        if avail_g >= head.gpus and avail_m >= head.mem:
            break
    for i, j in _fitting(env):
        if i != 0 and env.t + j.est_runtime <= shadow:
            return i
    return _wait(env)


def bestfit(env) -> int:
    """Start the fitting job that leaves the least unused capacity (both resources)."""
    cand = _fitting(env)
    if not cand:
        return _wait(env)

    def leftover(j):
        return (env.free - j.gpus) / env.n_gpus + (env.free_mem - j.mem) / env.n_mem

    return min(cand, key=lambda x: leftover(x[1]))[0]


BASELINES = {
    "fifo": fifo,
    "sjf": sjf,
    "edf": edf,
    "backfill": backfill,
    "bestfit": bestfit,
}
