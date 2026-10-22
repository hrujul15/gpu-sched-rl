"""Train a MaskablePPO agent. Example:
    python scripts/train.py --steps 300000 --out models/ppo_v1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from gpusched import GPUClusterEnv


def make_env(seed=None):
    # Train on a RANGE of loads so evaluation on unseen loads is meaningful.
    env = GPUClusterEnv(rate_range=(0.3, 0.7))
    env = Monitor(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_v1")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env = make_env(args.seed)
    model = MaskablePPO("MlpPolicy", env, verbose=1, seed=args.seed,
                        n_steps=2048, batch_size=256,
                        learning_rate=3e-4, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print("saved", args.out)
