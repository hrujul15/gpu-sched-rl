"""Train MaskablePPO, optionally warm-started by SJF imitation learning.

Example:
    python scripts/train.py --pretrain-episodes 500 --steps 3000000 --out models/ppo_v7
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gpusched import GPUClusterEnv, workload_preset
from gpusched.baselines import sjf


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


WORKLOAD = "default"
HIDE_RANK = False


def new_env():
    return GPUClusterEnv(
        workload=workload_preset(WORKLOAD),
        rate_range=(0.25, 0.95),
        hide_rank_feats=HIDE_RANK,
    )


def make_env(seed=None, rank=0):
    env = new_env()
    env = Monitor(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    env.reset(seed=None if seed is None else seed + rank * 100_003)
    return env


def collect_sjf_demonstrations(episodes: int, seed: int):
    """Collect raw observations, valid-action masks and SJF+PACK expert actions."""
    obs_rows, mask_rows, action_rows = [], [], []
    for ep in range(episodes):
        env = new_env()
        obs, _ = env.reset(seed=seed + ep)
        while True:
            mask = env.action_masks()
            action = int(sjf(env))
            obs_rows.append(obs.copy())
            mask_rows.append(mask.copy())
            action_rows.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=bool),
        np.asarray(action_rows, dtype=np.int64),
    )


def pretrain_actor(model, vec_norm, raw_obs, masks, actions,
                   epochs=12, batch_size=512, learning_rate=1e-3):
    """Supervised warm start of the policy actor; critic remains for PPO."""
    # Fit VecNormalize's observation statistics on the expert state distribution.
    vec_norm.obs_rms.update(raw_obs)
    obs = vec_norm.normalize_obs(raw_obs)
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)

    policy = model.policy
    actor_params = list(policy.mlp_extractor.parameters()) + list(policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=learning_rate)
    n = len(actions)
    rng = np.random.default_rng(12345)
    policy.set_training_mode(True)
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            features = policy.extract_features(obs_t[idx])
            if isinstance(features, tuple):
                features = features[0]
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)
            # Match MaskablePPO's legal-action support during imitation.
            legal_logits = logits.masked_fill(~masks_t[idx], torch.finfo(logits.dtype).min)
            loss = F.cross_entropy(legal_logits, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"imitation epoch {epoch + 1:02d}/{epochs}: loss={np.mean(losses):.4f}")
    policy.set_training_mode(False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_v7")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--pretrain-episodes", type=int, default=500,
                    help="SJF demonstration episodes; set 0 to disable imitation warm-start")
    ap.add_argument("--imitation-epochs", type=int, default=12)
    ap.add_argument("--workload", default="default", choices=["default", "hard"])
    ap.add_argument("--hide-rank-feats", action="store_true",
                    help="ablation: zero the precomputed SJF-rank features")
    args = ap.parse_args()
    WORKLOAD, HIDE_RANK = args.workload, args.hide_rank_feats

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env = DummyVecEnv(
        [lambda rank=rank: make_env(args.seed, rank) for rank in range(args.envs)]
    )
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
        gamma=0.995,
    )
    model = MaskablePPO(
        "MlpPolicy", env, verbose=1, seed=args.seed,
        n_steps=1024, batch_size=512,
        learning_rate=linear_schedule(2.5e-4),
        gamma=0.995, gae_lambda=0.95, ent_coef=0.005, n_epochs=10,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log="runs/",
    )

    if args.pretrain_episodes > 0:
        print(f"Collecting SJF demonstrations from {args.pretrain_episodes} episodes...")
        demo_obs, demo_masks, demo_actions = collect_sjf_demonstrations(
            args.pretrain_episodes, args.seed + 1_000_000
        )
        print(f"Collected {len(demo_actions):,} expert decisions; pretraining actor...")
        pretrain_actor(
            model, env, demo_obs, demo_masks, demo_actions,
            epochs=args.imitation_epochs,
        )

    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    env.save(args.out + "_vecnormalize.pkl")
    print(f"Saved {args.out} + VecNormalize statistics")