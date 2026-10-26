"""Compare baselines and a trained agent on fixed seeds.
python scripts/evaluate.py --model models/ppo_v5 --rate 0.9 --episodes 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gpusched import GPUClusterEnv, workload_preset
from gpusched.baselines import BASELINES


def run_episode(env, policy, seed):
    env.reset(seed=seed)
    while True:
        _, _, term, trunc, info = env.step(policy(env))
        if term or trunc:
            return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--rate", type=float, default=0.5, help="arrival rate of the eval workload"
    )
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--workload", default="default", choices=["default", "hard"])
    ap.add_argument("--hide-rank-feats", action="store_true",
                    help="must match how the model was trained")
    args = ap.parse_args()

    def make_eval_env():
        return GPUClusterEnv(
            workload=workload_preset(args.workload, args.rate),
            hide_rank_feats=args.hide_rank_feats,
        )

    env = make_eval_env()
    policies = dict(BASELINES)
    if args.model:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = MaskablePPO.load(args.model)

        # Load VecNormalize stats if available (needed for correct obs scaling)
        vec_norm = None
        base = args.model.replace(".zip", "")
        vec_norm_path = base + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            dummy = DummyVecEnv([make_eval_env])
            vec_norm = VecNormalize.load(vec_norm_path, dummy)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"Loaded VecNormalize from {vec_norm_path}")

        def agent(e):
            obs = e._obs()
            if vec_norm is not None:
                obs = vec_norm.normalize_obs(obs)
            a, _ = model.predict(
                obs, action_masks=e.action_masks(), deterministic=True
            )
            return int(a)

        policies["ppo"] = agent

    rows = []
    for name, pol in policies.items():
        for s in range(args.episodes):
            info = run_episode(env, pol, seed=10_000 + s)  # same seeds for every policy
            rows.append({"policy": name, "seed": s, **info})
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{args.tag}.csv", index=False)

    summary = (
        df.groupby("policy")[
            ["avg_jct", "miss_rate", "max_wait", "avg_util", "avg_mem_util"]
        ]
        .mean()
        .round(3)
    )
    print(summary)
    rcols = [c for c in df.columns if c.startswith("r_")]
    print(df.groupby("policy")[rcols].mean().round(1))

    fig, axes = plt.subplots(1, 5, figsize=(17, 3))
    for ax, col in zip(axes, summary.columns):
        summary[col].plot.bar(ax=ax, title=col)
    plt.tight_layout()
    plt.savefig(f"results/{args.tag}.png", dpi=150)