# RL Based GPU Job Scheduler

A reinforcement-learning scheduler for allocating queued training jobs across a shared GPU cluster. The agent is trained with **MaskablePPO**.

## Team

| Name | Roll No. |
|---|---|
| Hrujul Mendhe | B24CM1077 |
| Atharv Sharma | B24CM1013 |

## Problem

The simulator models a cluster of 16 physical GPUs, each with 40 GB of memory. A job requests a number of distinct GPUs and a memory allocation per selected GPU. Multiple jobs may share a GPU when sufficient device-local memory remains available.

The scheduler observes cluster availability, per-GPU free memory, and features for the first eight queued jobs. Actual job runtime is hidden; the agent receives an estimated runtime. Its goal is to make scheduling and placement decisions.

## Agent and environment

- **Algorithm:** MaskablePPO from `sb3-contrib`.
- **Actions:** Select one of the visible queued jobs and choose `PACK` or `SPREAD` placement, or wait for one simulation tick. Invalid actions are masked.
- **Reward:** Combines completion rewards with penalties for deadline misses and lateness, flow time, queue waiting and starvation, and idle capacity. Reward weights are defined in `gpusched/env.py`.
- **Baselines:** FIFO, SJF, EDF, backfill, and best-fit use the same memory-feasibility rules.

The final run uses the `hard` workload, which introduces more challenging job characteristics, including decoupled deadlines, more wide jobs, and noisier runtime estimates.

## Setup

From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10 or newer. On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` command.

## Train

Run the following commands **in order** from the repository root. Training saves the model to `models/ppo_v9_hard`.

```bash
python scripts/train.py --workload hard --steps 3000000 --out models/ppo_v9_hard
```

The evaluation runs use arrival rate `0.4` for the seen-load setting and `0.7` for the unseen-load setting. Each evaluates 100 episodes. Evaluation outputs are saved under `results/` as CSV data and PNG plots.

Training may take some time.

## Evaluate

In order to evaluate using the current model,

```bash
python scripts/evaluate.py --model models/ppo_v9_hard --workload hard --rate 0.4 --episodes 100 --tag v9_hard_seen
python scripts/evaluate.py --model models/ppo_v9_hard --workload hard --rate 0.7 --episodes 100 --tag v9_hard_unseen
```

## Live simulator UI (PPO only)

A local browser dashboard is included for interacting with the scheduling environment in real time. Make sure model is trained and present otherwise train the model first.

```bash
python ui/server.py --model models/ppo_v9_hard --workload hard
```

Open **http://127.0.0.1:8765** in your browser.

## Repository structure

```text
gpusched/
  jobs.py        Job definitions and workload generation
  env.py         Scheduling environment, memory accounting, rewards, action masks
  baselines.py   FIFO, SJF, EDF, backfill, and best-fit baselines
scripts/
  train.py       MaskablePPO training
  evaluate.py    Agent-versus-baselines evaluation
models/          Saved model checkpoints and normalization statistics
results/          Evaluation CSV files and plots
requirements.txt Python dependencies
```

## Modelling assumptions

- GPU memory is tracked separately for each physical GPU.
- A job can share a physical GPU with other jobs only when its requested memory fits.
- The reported GPU allocation/utilization metric indicates whether a GPU has any allocation; it does not model actual compute utilization or compute sharing.
- Runtime estimates are observations available to the scheduler; true runtimes are not directly exposed.
