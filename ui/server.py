"""Live local GPU scheduler dashboard. Run: python ui/server.py --workload hard [--model models/ppo_v9_hard]"""

from __future__ import annotations
import argparse, json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpusched import GPUClusterEnv, Job, workload_preset


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else round(x, 4)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Session:
    def __init__(self, model=None, hide_rank=False, workload="default"):
        self.lock = threading.RLock()
        self.cfg = workload_preset(workload)
        self.hide_rank = hide_rank
        self.rng = np.random.default_rng()
        self.policies = {}
        self.model_error = None
        if model:
            try:
                self.policies["ppo"] = self.load_model(model)
            except Exception as e:
                self.model_error = f"{type(e).__name__}: {e}"
                print("PPO load failed:", self.model_error, file=sys.stderr)

        if "ppo" not in self.policies:
            raise RuntimeError(
                "PPO model is required. Start with --model models/ppo_v9_hard"
            )
        self.policy = "ppo"
        self.speed = 2.0
        self.paused = False
        self.auto_rate = 0.0
        self.reset()

    def new_env(self):
        e = GPUClusterEnv(
            workload=self.cfg, hide_rank_feats=self.hide_rank, max_ticks=10**9
        )
        e.reset(seed=0)
        e.jobs = []
        e.pending = []
        e.queue = []
        e.running = []
        e.done = []
        return e

    def load_model(self, path):
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = MaskablePPO.load(path)
        norm = None
        p = path.removesuffix(".zip") + "_vecnormalize.pkl"
        if os.path.exists(p):
            norm = VecNormalize.load(
                p,
                DummyVecEnv(
                    [
                        lambda: GPUClusterEnv(
                            workload=self.cfg, hide_rank_feats=self.hide_rank
                        )
                    ]
                ),
            )
            norm.training = False
            norm.norm_reward = False

        def act(e):
            obs = e._obs()
            if norm is not None:
                obs = norm.normalize_obs(obs)
            a, _ = model.predict(obs, action_masks=e.action_masks(), deterministic=True)
            return int(a)

        return act

    def reset(self):
        with self.lock:
            self.env = self.new_env()
            self.log = []
            self.next_id = 0
            self.history = []
            self.ended = False
            self.summary = None
            self.paused = False

    def add_job(self, gpus, mem_per_gpu, runtime, est_runtime=None, deadline_in=None):
        with self.lock:
            if self.ended:
                raise ValueError("Session ended; start a new session.")
            e = self.env
            gpus = int(gpus)
            mpg = int(mem_per_gpu)
            runtime = int(runtime)
            if not 1 <= gpus <= e.n_gpus:
                raise ValueError(f"GPUs must be 1..{e.n_gpus}")
            if not 1 <= mpg <= e.gpu_mem_gb:
                raise ValueError(f"Memory per GPU must be 1..{e.gpu_mem_gb} GB")
            if not 1 <= runtime <= self.cfg.max_runtime:
                raise ValueError(f"Runtime must be 1..{self.cfg.max_runtime}")
            est = (
                runtime
                if est_runtime in (None, "")
                else int(np.clip(int(est_runtime), 1, self.cfg.max_runtime))
            )
            dl = (
                math.ceil(runtime * 2.5)
                if deadline_in in (None, "")
                else max(1, int(deadline_in))
            )
            j = Job(self.next_id, e.t, gpus, gpus * mpg, runtime, est, e.t + dl)
            self.next_id += 1
            e.jobs.append(j)
            e.queue.append(j)
            self.log.append(
                dict(
                    jid=j.jid,
                    arrival=j.arrival,
                    gpus=gpus,
                    mem=j.mem,
                    runtime=runtime,
                    est=est,
                    deadline=j.deadline,
                )
            )
            return j.jid

    def random_spec(self):
        c, r = self.cfg, self.rng
        g = int(r.choice(c.gpu_choices, p=c.gpu_probs))
        mpg = int(r.choice(c.mem_per_gpu_choices, p=c.mem_per_gpu_probs))
        rt = int(
            np.clip(round(r.lognormal(c.runtime_mu, c.runtime_sigma)), 1, c.max_runtime)
        )
        est = int(np.clip(round(rt * r.lognormal(0, c.est_noise)), 1, c.max_runtime))
        dl = (
            rt + math.ceil(r.uniform(c.abs_slack_low, c.abs_slack_high))
            if c.decoupled_frac and r.random() < c.decoupled_frac
            else math.ceil(rt * r.uniform(c.slack_low, c.slack_high))
        )
        return dict(
            gpus=g, mem_per_gpu=mpg, runtime=rt, est_runtime=est, deadline_in=dl
        )

    def add_random(self, n):
        return [
            self.add_job(**self.random_spec()) for _ in range(max(1, min(int(n), 200)))
        ]

    def tick(self):
        with self.lock:
            if self.ended or self.paused:
                return
            for _ in (
                range(int(self.rng.poisson(self.auto_rate)))
                if self.auto_rate > 0
                else []
            ):
                self.add_job(**self.random_spec())
            e = self.env
            pol = self.policies[self.policy]
            for _ in range(2 * e.K + 2):
                a = int(pol(e))
                if not e.action_masks()[a]:
                    a = e.wait_action
                e.step(a)
                if a == e.wait_action:
                    break
            if not self.history or self.history[-1]["t"] != e.t:
                self.history.append(
                    dict(
                        t=e.t,
                        util=(e.n_gpus - e.free) / e.n_gpus,
                        mem_util=(e.n_mem - e.free_mem) / e.n_mem,
                        queue=len(e.queue),
                        running=len(e.running),
                    )
                )
                del self.history[:-600]

    def state(self):
        with self.lock:
            e = self.env
            m = e._metrics()
            return clean(
                dict(
                    t=e.t,
                    ended=self.ended,
                    paused=self.paused,
                    speed=self.speed,
                    auto_rate=self.auto_rate,
                    policy=self.policy,
                    policies=list(self.policies),
                    model_error=self.model_error,
                    n_gpus=e.n_gpus,
                    gpu_mem=e.gpu_mem_gb,
                    gpu_used=[int(e.gpu_mem_gb - v) for v in e.gpu_free_mem],
                    busy_now=(e.n_gpus - e.free) / e.n_gpus,
                    mem_now=(e.n_mem - e.free_mem) / e.n_mem,
                    metrics={
                        k: m[k]
                        for k in [
                            "jobs_done",
                            "avg_jct",
                            "miss_rate",
                            "max_wait",
                            "avg_util",
                            "avg_mem_util",
                            "makespan",
                        ]
                    },
                    terms={
                        k: m[k]
                        for k in [
                            "r_idle",
                            "r_starve",
                            "r_wait",
                            "r_late",
                            "r_miss",
                            "r_flow",
                        ]
                    },
                    n_jobs=len(e.jobs),
                    n_queue=len(e.queue),
                    n_running=len(e.running),
                    queue=[
                        dict(
                            id=j.jid,
                            gpus=j.gpus,
                            mpg=j.mem_per_gpu,
                            est=j.est_runtime,
                            deadline=j.deadline,
                            waited=e.t - j.arrival,
                            fits=e._fits(j),
                        )
                        for j in e.queue
                    ],
                    running=[
                        dict(
                            id=j.jid,
                            gpus=j.gpus,
                            mpg=j.mem_per_gpu,
                            on=list(j.gpu_ids),
                            progress=(e.t - j.start) / j.runtime,
                            finish=j.finish,
                            deadline=j.deadline,
                        )
                        for j in e.running
                    ],
                    done=[
                        dict(
                            id=j.jid,
                            jct=j.finish - j.arrival,
                            wait=j.start - j.arrival,
                            missed=j.finish > j.deadline,
                        )
                        for j in e.done[-12:]
                    ][::-1],
                    history=self.history,
                    summary=self.summary,
                )
            )

    def replay(self, name):
        e = self.new_env()
        jobs = [
            Job(
                s["jid"],
                s["arrival"],
                s["gpus"],
                s["mem"],
                s["runtime"],
                s["est"],
                s["deadline"],
            )
            for s in self.log
        ]
        e.jobs = jobs
        e.pending = sorted(jobs, key=lambda j: (j.arrival, j.jid))
        e._admit()
        pol = self.policies[name]
        for _ in range(500000):
            a = int(pol(e))
            if not e.action_masks()[a]:
                a = e.wait_action
            _, _, term, trunc, _ = e.step(a)
            if term or trunc:
                break
        m = e._metrics()
        return {
            k: m[k]
            for k in [
                "jobs_done",
                "avg_jct",
                "miss_rate",
                "max_wait",
                "avg_util",
                "avg_mem_util",
                "makespan",
            ]
        }

    def end(self):
        with self.lock:
            if self.ended:
                return
            self.ended = True
            e = self.env
            m = e._metrics()
            jobs = []
            for j in e.jobs:
                if j in e.done:
                    jobs.append(
                        dict(
                            id=j.jid,
                            status="done",
                            gpus=j.gpus,
                            wait=j.start - j.arrival,
                            jct=j.finish - j.arrival,
                            missed=j.finish > j.deadline,
                        )
                    )
                else:
                    jobs.append(
                        dict(
                            id=j.jid,
                            status="running" if j in e.running else "queued",
                            gpus=j.gpus,
                            wait=None,
                            jct=None,
                            missed=None,
                        )
                    )
            replays = {}
            if self.log:
                for name in self.policies:
                    try:
                        replays[name] = self.replay(name)
                    except Exception as ex:
                        replays[name] = {"error": str(ex)}
            self.summary = clean(
                dict(
                    policy=self.policy,
                    ticks=e.t,
                    submitted=len(e.jobs),
                    completed=len(e.done),
                    unfinished=len(e.jobs) - len(e.done),
                    live={
                        k: m[k]
                        for k in [
                            "jobs_done",
                            "avg_jct",
                            "miss_rate",
                            "max_wait",
                            "avg_util",
                            "avg_mem_util",
                            "makespan",
                        ]
                    },
                    terms={
                        k: m[k]
                        for k in [
                            "r_idle",
                            "r_starve",
                            "r_wait",
                            "r_late",
                            "r_miss",
                            "r_flow",
                        ]
                    },
                    jobs=jobs,
                    replays=replays,
                )
            )


SESSION = None


def loop():
    while True:
        s = SESSION
        time.sleep(1 / max(s.speed, 0.1))
        try:
            s.tick()
        except Exception as e:
            print("tick error:", e, file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def sendj(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/state":
            return self.sendj(200, SESSION.state())
        if self.path in ("/", "/index.html"):
            b = open(os.path.join(os.path.dirname(__file__), "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        self.sendj(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            b = json.loads(self.rfile.read(n) or b"{}")
            s = SESSION
            with s.lock:
                if self.path == "/api/add_job":
                    return self.sendj(
                        200,
                        {
                            "ok": True,
                            "id": s.add_job(
                                b.get("gpus"),
                                b.get("mem_per_gpu"),
                                b.get("runtime"),
                                b.get("est_runtime"),
                                b.get("deadline_in"),
                            ),
                        },
                    )
                if self.path == "/api/add_random":
                    return self.sendj(
                        200, {"ok": True, "ids": s.add_random(b.get("n", 1))}
                    )
                if self.path == "/api/policy":
                    if b["name"] not in s.policies:
                        raise ValueError("Unknown policy")
                    s.policy = b["name"]
                elif self.path == "/api/speed":
                    s.speed = float(np.clip(float(b["speed"]), 0.2, 50))
                elif self.path == "/api/pause":
                    s.paused = bool(b["paused"])
                elif self.path == "/api/auto":
                    s.auto_rate = float(np.clip(float(b["rate"]), 0, 5))
                elif self.path == "/api/end":
                    s.end()
                elif self.path == "/api/reset":
                    s.reset()
                else:
                    return self.sendj(404, {"error": "not found"})
            self.sendj(200, {"ok": True})
        except Exception as e:
            self.sendj(400, {"ok": False, "error": str(e)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--workload", choices=["default", "hard"], default="default")
    p.add_argument("--hide-rank-feats", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    SESSION = Session(a.model, a.hide_rank_feats, a.workload)
    threading.Thread(target=loop, daemon=True).start()
    print(
        f"Serving on http://127.0.0.1:{a.port} (policies: {', '.join(SESSION.policies)})"
    )
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
