"""Example 8 - MAPPO vs VDN (and IPPO) on Fetch.

Trains, for every Fetch map given (default ``open`` and ``single``), three learners over three
seeds in parallel processes with the same budget and episode settings as example 6:

* ``vdn``   - value decomposition, recurrent DQN learner, hyper-parameters of example 6
* ``mappo`` - MAPPO with the defaults of Yu et al. 2022 (shared policy, agent-specific centralised
              critic, value normalisation, 10 epochs, 1 mini-batch, clip 0.2, rollouts of 1000 steps)
* ``ippo``  - the same PPO learner with a critic that only sees the agent's own observation

then plots training / greedy-evaluation returns and per-agent reward shares and writes tables
and GIFs to ``results/mappo_vs_vdn/<map>/``.

    python examples/08_fetch_mappo_vs_vdn.py                # open + single
    python examples/08_fetch_mappo_vs_vdn.py open           # one map
    python examples/08_fetch_mappo_vs_vdn.py --replot       # rebuild figures from saved runs
    VDN_QUICK=1 python examples/08_fetch_mappo_vs_vdn.py    # plumbing smoke test
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _common import QUICK, budget, hp  # noqa: E402

cmp = importlib.import_module("06_compare_learners")      # reuse workers, behaviour stats, plots, tables

ROOT = Path(__file__).resolve().parents[1] / "results" / "mappo_vs_vdn"
ALGOS = ["vdn", "mappo", "ippo"]
cmp.COLORS.update({"mappo": "#c0392b", "ippo": "#8e44ad"})
SEEDS = cmp.SEEDS
STEPS = budget(200_000)


def variants(steps: int):
    quick_ppo = dict(rollout_len=500, ppo_epochs=2) if QUICK else {}
    return {
        "vdn": {"agent": "vdn", "agent_kwargs": hp(steps)},
        "mappo": {"agent": "mappo", "agent_kwargs": quick_ppo},
        "ippo": {"agent": "ippo", "agent_kwargs": quick_ppo},
    }


def main(maps):
    from vdn_lab.experiment import ExperimentConfig
    jobs = []
    for m in maps:
        base = ExperimentConfig(env="fetch", env_kwargs={"layout": m}, episode_len=200, eval_episode_len=200,
                                train_steps=STEPS, eval_every=STEPS // 8, eval_episodes=3,
                                final_eval_sessions=5, save_agent=True)
        out = ROOT / m
        out.mkdir(parents=True, exist_ok=True)
        for label, ov in variants(STEPS).items():
            cfg = base.replace(**ov)
            for seed in SEEDS:
                jobs.append((m, label, seed, cfg.to_dict(), str(out / "runs")))
    results = {m: {} for m in maps}
    workers = int(os.environ.get("VDN_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        for m, label, seed, res in pool.map(cmp._train_one, jobs):
            results[m].setdefault(label, []).append(res)
        gif_jobs = []
        for m in maps:
            for label, runs in results[m].items():
                best = max(runs, key=lambda r: r["final_eval"]["mean_return"])
                gif_jobs.append((best["run_dir"], str(ROOT / m / f"{label}_best_seed.gif")))
        list(pool.map(cmp._gif_one, gif_jobs))
    for m in maps:
        report(m, results[m])


def report(m, results):
    out = ROOT / m
    results = {a: results[a] for a in ALGOS if a in results}
    cmp.plot_game(f"fetch ({m})", results, out, STEPS)
    cmp.summarise_game(f"fetch ({m})", results, out)


def replot(maps):
    for m in maps:
        results = {}
        for f in sorted((ROOT / m / "runs").glob("*/result.json")):
            res = json.loads(f.read_text())
            results.setdefault(res["config"]["name"].rsplit("_s", 1)[0], []).append(res)
        report(m, results)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    maps = args or ["open", "single"]
    replot(maps) if "--replot" in sys.argv else main(maps)
