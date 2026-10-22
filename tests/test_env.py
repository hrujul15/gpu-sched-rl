import numpy as np

from gpusched import GPUClusterEnv
from gpusched.baselines import BASELINES


def test_obs_shape_and_bounds():
    env = GPUClusterEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_gpu_accounting_never_negative():
    env = GPUClusterEnv()
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        valid = np.flatnonzero(env.action_masks())
        _, _, term, trunc, _ = env.step(int(rng.choice(valid)))
        assert 0 <= env.free <= env.n_gpus
        assert env.free + sum(j.gpus for j in env.running) == env.n_gpus
        if term or trunc:
            break


def test_baselines_finish_all_jobs():
    for name, pol in BASELINES.items():
        env = GPUClusterEnv()
        env.reset(seed=3)
        info = {}
        for _ in range(env.max_ticks * 20):
            _, _, term, trunc, info = env.step(pol(env))
            if term or trunc:
                break
        assert info["jobs_done"] == info["jobs_total"], name
