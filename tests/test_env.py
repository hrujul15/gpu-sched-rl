import numpy as np

from gpusched import GPUClusterEnv
from gpusched.baselines import BASELINES


def test_obs_shape_and_bounds():
    env = GPUClusterEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.shape == (4 + env.n_gpus + 12 * env.K,)
    assert env.observation_space.contains(obs)


def test_per_gpu_memory_accounting_never_exceeds_capacity():
    env = GPUClusterEnv()
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(env.max_ticks * 20):
        valid = np.flatnonzero(env.action_masks())
        _, _, term, trunc, _ = env.step(int(rng.choice(valid)))
        assert env.gpu_free_mem.shape == (env.n_gpus,)
        assert np.all(env.gpu_free_mem >= 0)
        assert np.all(env.gpu_free_mem <= env.gpu_mem_gb)
        assert env.free_mem == int(env.gpu_free_mem.sum())
        allocated = sum(j.mem for j in env.running)
        assert env.free_mem + allocated == env.n_mem
        assert env.free == int(np.count_nonzero(env.gpu_free_mem == env.gpu_mem_gb))
        if term or trunc:
            break


def test_job_requires_memory_on_each_selected_gpu():
    env = GPUClusterEnv(n_gpus=4, k_queue=2, gpu_mem_gb=40)
    env.reset(seed=7)
    # Artificially fragment free memory while preserving total free memory.
    env.gpu_free_mem[:] = [8, 8, 40, 40]
    env._refresh_capacity_counters()
    from gpusched.jobs import Job
    job = Job(999, env.t, 2, 32, 3, 3, 20)  # 16 GB on each of 2 GPUs
    assert env._fits(job)
    env.gpu_free_mem[:] = [8, 8, 15, 40]
    env._refresh_capacity_counters()
    assert not env._fits(job)


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
