"""Train one algorithm on one LBF task with one seed and write ``result.json``.

Evaluation protocol (paper v1 sec. 4.1 / v4 sec. 4.1): ``n_evals`` = 41 evaluations at equal intervals
(t = 0 included), each the mean team return of 100 episodes.

    python -m lbf_bench.run --algo qmix --task lbf-8x8-2p-3f --seed 0 --out results/lbf
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .buffer import Collector
from .configs import ALGOS
from .env import LBFEnv, make_envs

EVAL_ENVS = 20


def evaluate(algo, collector: Collector, n_episodes: int, t_env: int) -> np.ndarray:
    rets = []
    while len(rets) < n_episodes:
        b = collector.run(algo, explore=False, t_env=t_env)
        rets.extend(b["reward"].sum(1).tolist())
    return np.asarray(rets[:n_episodes])


def _bin(rows: List[tuple], t_max: int, n_bins: int) -> List[list]:
    """Average rows ``(t, v1, v2, ...)`` into ``n_bins`` equal bins of t; returns ``[t_mid, v1, ...]``."""
    if not rows:
        return []
    a = np.asarray(rows, dtype=np.float64)
    edges = np.linspace(0, t_max, n_bins + 1)
    which = np.clip(np.searchsorted(edges, a[:, 0], side="right") - 1, 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        sel = a[which == k]
        if len(sel):
            out.append([float(0.5 * (edges[k] + edges[k + 1]))] + sel[:, 1:].mean(0).tolist() + [int(len(sel))])
    return out


def train(algo_name: str, task: str, seed: int, out_root: str, t_max: Optional[int] = None,
          n_evals: int = 41, eval_episodes: int = 100, overrides: Optional[Dict] = None,
          tag: Optional[str] = None) -> Path:
    torch.set_num_threads(1)
    spec = ALGOS[algo_name]
    t_max = int(t_max or spec.t_max)
    probe = LBFEnv(task)
    kwargs = {**spec.kwargs, **(overrides or {})}
    algo = spec.cls(probe.n_agents, probe.n_actions, probe.obs_dim, probe.episode_limit, seed, **kwargs)
    collector = Collector(make_envs(task, spec.n_envs, seed))
    eval_collector = Collector(make_envs(task, EVAL_ENVS, seed + 7919))
    name = tag or algo_name
    out = Path(out_root) / task / name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    interval = t_max / (n_evals - 1)
    eval_t, eval_mean, eval_std = [], [], []
    train_rows: List[tuple] = []
    t_env, next_eval, episodes = 0, 0.0, 0
    t0 = time.time()
    last_print = t0
    while True:
        if t_env >= next_eval - 1e-9:
            rets = evaluate(algo, eval_collector, eval_episodes, t_env)
            eval_t.append(t_env), eval_mean.append(float(rets.mean())), eval_std.append(float(rets.std()))
            next_eval += interval
            if time.time() - last_print > 60 or len(eval_t) == 1:
                print(f"[{name} {task} s{seed}] t={t_env} eval={rets.mean():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                last_print = time.time()
        if t_env >= t_max:
            break
        batch = collector.run(algo, explore=True, t_env=t_env)
        steps = int(batch["length"].sum())
        t_env += steps
        episodes += len(batch["length"])
        algo.train(batch, t_env)
        for ret in batch["reward"].sum(1):
            train_rows.append((t_env, float(ret)))

    wall = time.time() - t0
    n_bins = 200
    result = dict(
        algo=algo_name, tag=name, label=spec.label, task=task, seed=seed, t_max=t_max, episodes=episodes,
        train_steps=algo.train_steps, wall_time=wall, n_envs=spec.n_envs,
        config={k: (v if not isinstance(v, float) or np.isfinite(v) else str(v))
                for k, v in vars(algo.cfg).items()},
        eval=dict(t=eval_t, mean=eval_mean, std=eval_std, episodes=eval_episodes),
        train=_bin(train_rows, t_max, n_bins),
        lam=_bin(algo.lam_log, t_max, n_bins),              # t, mean, std, p10, p50, p90, <0.1, >0.9, count
        atd=_bin(algo.atd_log, t_max, n_bins),              # t, bce, accuracy, count
    )
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out / "result.json")
    print(f"[{name} {task} s{seed}] done in {wall:.0f}s, final eval {eval_mean[-1]:.3f}, "
          f"max eval {max(eval_mean):.3f}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=sorted(ALGOS))
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-max", type=int, default=None)
    p.add_argument("--n-evals", type=int, default=41)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--out", default="results/lbf")
    p.add_argument("--tag", default=None, help="result directory name (default: the algorithm name)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key=value (python literals)")
    a = p.parse_args(argv)
    import ast
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v
    train(a.algo, a.task, a.seed, a.out, a.t_max, a.n_evals, a.eval_episodes, overrides, a.tag)


if __name__ == "__main__":
    main()
