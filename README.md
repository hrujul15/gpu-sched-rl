# gpu-sched-rl

Reinforcement-learning scheduler for training jobs on a shared GPU cluster
(AI course, Assignment 2: real-life use cases of RL).

## Team

| Name           | Roll No. |
|----------------|----------|
| Hrujul Mendhe  | B24CM1077 |
| Atharv Sharma  | B24CM1013 |

## Problem
Jobs arrive online with a GPU request, a noisy runtime estimate and a deadline.
The agent decides which queued job to start (or to wait) so as to maximize
throughput and utilization while limiting deadline misses and starvation.

## RL formulation
- **State (3 + 5K dims, K = 8):** free GPUs, queue length, estimated time until the
  next GPU frees; per queued job (first K by arrival): GPU request, estimated
  runtime, deadline slack, wait time, valid flag.
- **Actions (Discrete K+1):** start the i-th queued job, or wait one tick. Invalid
  starts (job does not fit) are masked.
- **Reward (per step):** +`w_complete` per finished job, -`w_miss` per missed
  deadline, +`w_util` x utilization per tick, -`w_starve` per starving job per tick.
  Weights live in `RewardConfig`.
- **Algorithm:** MaskablePPO (sb3-contrib). Baselines: FIFO, SJF, EDF, EASY backfill.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Run
```bash
python scripts/train.py --steps 300000 --out models/ppo_v1
python scripts/evaluate.py --model models/ppo_v1 --rate 0.5 --episodes 30 --tag seen_load
python scripts/evaluate.py --model models/ppo_v1 --rate 0.9 --episodes 30 --tag unseen_load
```
Training samples the arrival rate from 0.3-0.7 per episode; evaluating at 0.9
tests generalization to an unseen load. Results (CSV + bar plots) go to `results/`.

## Layout
```
gpusched/jobs.py       job model, synthetic workload, trace-loader stub
gpusched/env.py        Gymnasium environment (own event loop, action masking)
gpusched/baselines.py  FIFO / SJF / EDF / backfill
scripts/train.py       MaskablePPO training
scripts/evaluate.py    baselines vs agent on fixed seeds
tests/test_env.py      sanity tests
```

## Roadmap
- [ ] Learning curves (log episode reward during training) and seed variance
- [ ] Reward-weight ablation (utilization vs deadline vs starvation)
- [ ] Load real traces (Philly / Helios / Alibaba PAI) in `load_trace_csv`
- [ ] Add GPU memory as a second resource
- [ ] Noisy-estimate ablation (`est_noise`) to show where SJF breaks down
- [ ] Compare PPO vs DQN
- [ ] 2-page report

## Notes
The simulator is a plain Python event loop rather than SimPy, because the agent
needs to step the clock decision by decision. SimPy can be added later if you
want process-style modelling.


