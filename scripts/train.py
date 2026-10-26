"""Train a MaskablePPO agent.
Example:
    python scripts/train.py --steps 1000000 --out models/ppo_v5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gpusched import GPUClusterEnv


def linear_schedule(initial_value: float):
    """Linear decay from initial_value → 0 over training."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(seed=None):
    # Train on a RANGE of loads so evaluation on unseen loads is meaningful.
    env = GPUClusterEnv(rate_range=(0.3, 0.7))
    env = Monitor(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_v5")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # VecNormalize requires a vectorized environment.
    env = DummyVecEnv([lambda: make_env(args.seed)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=512,
        learning_rate=linear_schedule(3e-4),
        gamma=0.99,
        ent_coef=0.02,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log="runs/",
    )

    model.learn(total_timesteps=args.steps)

    model.save(args.out)
    env.save(args.out + "_vecnormalize.pkl")
    print(f"saved {args.out} + vecnormalize stats")