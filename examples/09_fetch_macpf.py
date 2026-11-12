"""Example 9 - MACPF vs MAPPO, IPPO and VDN on Fetch.

Trains MACPF (multi-agent conditional policy factorisation, Wang et al. 2023) with the
hyper-parameters of the paper's appendix C.2 (SMAC easy / hard maps) on every Fetch map given (default
``open`` and ``single``), over the same three seeds, budget and episode settings as examples 6
and 8. The baselines (``vdn``, ``mappo``, ``ippo``) are taken from the saved runs of example 8
(``results/mappo_vs_vdn/<map>/runs``) when they are there, and trained otherwise.

MACPF acts with its *independent* policies at evaluation time (decentralised execution). For
reference, every MACPF run is also evaluated with its *dependent* policies (agent 1 sees agent
0's action; centralised execution), which shows how much is lost by the factorisation.

Tables, plots and GIFs go to ``results/macpf_vs_baselines/<map>/``.

    python examples/09_fetch_macpf.py                 # open + single
    python examples/09_fetch_macpf.py open            # one map
    python examples/09_fetch_macpf.py --replot        # rebuild figures from saved runs
    VDN_QUICK=1 python examples/09_fetch_macpf.py     # plumbing smoke test
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import QUICK, budget, hp  # noqa: E402

cmp = importlib.import_module("06_compare_learners")      # workers, behaviour stats, plots, tables

ROOT = Path(__file__).resolve().parents[1] / "results" / "macpf_vs_baselines"
BASELINE_ROOT = Path(__file__).resolve().parents[1] / "results" / "mappo_vs_vdn"
ALGOS = ["macpf", "macpf_a0.001", "macpf_nodetach", "mappo", "ippo", "vdn"]
MACPF = ("macpf", "macpf_a0.001", "macpf_nodetach")
cmp.COLORS.update({"mappo": "#c0392b", "ippo": "#8e44ad", "macpf": "#16a085", "macpf_a0.001": "#7fb800", "macpf_nodetach": "#e6a700"})
SEEDS = cmp.SEEDS
STEPS = budget(200_000)


def variants(steps: int):
    quick_ppo = dict(rollout_len=500, ppo_epochs=2) if QUICK else {}
    quick_macpf = dict(batch_size=2, max_train_len=50) if QUICK else {}
    return {
        "macpf": {"agent": "macpf", "agent_kwargs": quick_macpf},   # paper defaults (sec. 3, appendix C.2 / E)
        # appendix C.2's other temperature setting (super-hard SMAC maps): fixed alpha = 0.001
        "macpf_a0.001": {"agent": "macpf", "agent_kwargs": {**quick_macpf, "alpha_start": 0.001, "alpha_end": 0.001}},
        # as the released code: the dependent losses also train the shared GRU trunk
        "macpf_nodetach": {"agent": "macpf", "agent_kwargs": {**quick_macpf, "detach_dep_trunk": False}},
        "mappo": {"agent": "mappo", "agent_kwargs": quick_ppo},
        "ippo": {"agent": "ippo", "agent_kwargs": quick_ppo},
        "vdn": {"agent": "vdn", "agent_kwargs": hp(steps)},
    }


def _load_runs(run_root: Path):
    results = {}
    for f in sorted(run_root.glob("*/result.json")):
        res = json.loads(f.read_text())
        results.setdefault(res["config"]["name"].rsplit("_s", 1)[0], []).append(res)
    return results


def _dep_eval(args):
    """Greedy evaluation of a trained MACPF run with its dependent (centralised) policies."""
    run_dir, sessions = args
    import torch
    torch.set_num_threads(1)
    from vdn_lab.experiment import Experiment
    exp = Experiment.from_run_dir(run_dir, verbose=0, agent_kwargs={"eval_policy": "dep"})
    return run_dir, exp.evaluate(sessions)


def main(maps):
    from vdn_lab.experiment import ExperimentConfig
    jobs, reused = [], {m: {} for m in maps}
    for m in maps:
        base = ExperimentConfig(env="fetch", env_kwargs={"layout": m}, episode_len=200, eval_episode_len=200,
                                train_steps=STEPS, eval_every=STEPS // 8, eval_episodes=3,
                                final_eval_sessions=5, save_agent=True)
        out = ROOT / m
        out.mkdir(parents=True, exist_ok=True)
        saved = {} if QUICK else {**_load_runs(BASELINE_ROOT / m / "runs"), **_load_runs(out / "runs")}
        for label, ov in variants(STEPS).items():
            have = saved.get(label, [])
            if sorted(r["config"]["seed"] for r in have) == SEEDS \
                    and all(r["config"]["train_steps"] == STEPS for r in have):
                reused[m][label] = have
                continue
            cfg = base.replace(**ov)
            for seed in SEEDS:
                jobs.append((m, label, seed, cfg.to_dict(), str(out / "runs")))
    for m in maps:
        if reused[m]:
            print(f"[{m}] reusing saved runs for: {sorted(reused[m])}")
    results = {m: dict(reused[m]) for m in maps}
    workers = int(os.environ.get("VDN_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        for m, label, seed, res in pool.map(cmp._train_one, jobs):
            results[m].setdefault(label, []).append(res)
        dep_jobs = [(r["run_dir"], 5) for m in maps for a in MACPF for r in results[m].get(a, [])
                    if "dep_eval" not in r]
        dep = dict(pool.map(_dep_eval, dep_jobs))
        gif_jobs = []
        for m in maps:
            for r in (r for a in MACPF for r in results[m].get(a, [])):
                if r["run_dir"] in dep:
                    r["dep_eval"] = dep[r["run_dir"]]
                    Path(r["run_dir"], "result.json").write_text(json.dumps(r, indent=2))
            for label, runs in results[m].items():
                best = max(runs, key=lambda r: r["final_eval"]["mean_return"])
                if not (ROOT / m / f"{label}_best_seed.gif").exists():
                    gif_jobs.append((best["run_dir"], str(ROOT / m / f"{label}_best_seed.gif")))
        list(pool.map(cmp._gif_one, gif_jobs))
    for m in maps:
        report(m, results[m])


def report(m, results):
    out = ROOT / m
    results = {a: results[a] for a in ALGOS if a in results}
    cmp.plot_game(f"fetch ({m})", results, out, STEPS)
    rows = cmp.summarise_game(f"fetch ({m})", results, out)
    extra = ["", "MACPF greedy return, independent policies (decentralised execution) vs dependent "
             "policies (agent 1 sees agent 0's action):", "",
             "| variant | independent | dependent | independent per seed | dependent per seed |", "|---|---|---|---|---|"]
    for a in MACPF:
        runs = results.get(a, [])
        if not runs or not all("dep_eval" in r for r in runs):
            continue
        ind = np.array([r["final_eval"]["mean_return"] for r in runs])
        dep = np.array([r["dep_eval"]["mean_return"] for r in runs])
        extra.append(f"| {a} | {ind.mean():.2f} ± {ind.std():.2f} | {dep.mean():.2f} ± {dep.std():.2f} | "
                     f"{[round(float(v), 1) for v in ind]} | {[round(float(v), 1) for v in dep]} |")
        for r in rows:
            if r["variant"] == a:
                r["dep_greedy_final"], r["dep_greedy_per_seed"] = round(dep.mean(), 2), [round(float(v), 1) for v in dep]
    if len(extra) > 5:
        with open(out / "summary.md", "a") as f:
            f.write("\n".join(extra) + "\n")
        print("\n".join(extra))
        (out / "summary.json").write_text(json.dumps(rows, indent=2))


def replot(maps):
    for m in maps:
        results = _load_runs(BASELINE_ROOT / m / "runs")
        results.update(_load_runs(ROOT / m / "runs"))
        report(m, results)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    maps = args or ["open", "single"]
    if QUICK:
        ROOT = ROOT.parent / "_quick_macpf"
        shutil.rmtree(ROOT, ignore_errors=True)
    replot(maps) if "--replot" in sys.argv else main(maps)
