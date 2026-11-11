"""Example 10 - adaptive TD(lambda) (ATD(lambda)) for QMIX, QPLEX and MAPPO on Fetch.

Trains, for every Fetch map given (default ``open`` and ``single``), each base learner and its
ATD(lambda) variant over the same three seeds, budget and episode settings as examples 6, 8 and 9:

* ``qmix``  vs ``qmix_atd``   - per-pair lambda in QMIX's lambda-returns (base lambda 0.9)
* ``qplex`` vs ``qplex_atd``  - the same for QPLEX
* ``mappo`` vs ``mappo_atd``  - per-step lambda in MAPPO's GAE (base lambda 0.95)
* ``vdn``                     - reference
* ``qmix_lam07``, ``qplex_lam07`` - controls: the base learners with a *fixed* lambda of 0.7, about
                                the average lambda the ATD variants end up using, to separate the
                                effect of adapting lambda per pair from that of a lower mean lambda

QMIX and QPLEX use VDN's hyper-parameters (``_common.hp``), MAPPO the defaults of Yu et al. 2022.
The ATD(lambda) variants add only the density-ratio estimator, with the defaults in
``vdn_lab/agents/atd.py``; nothing was tuned for this task. ``vdn`` and ``mappo`` are taken from
the saved runs of example 8 (``results/mappo_vs_vdn/<map>/runs``) when they are there.

Each ATD run also records the estimator's statistics (classification accuracy between the two
buffers, mean / spread of lambda, fraction clipped) every ~10k steps in ``result.json``
(``atd_trace``). Tables, plots and GIFs go to ``results/atd_vs_baselines/<map>/``.

    python examples/10_fetch_adaptive_lambda.py              # open + single
    python examples/10_fetch_adaptive_lambda.py open         # one map
    python examples/10_fetch_adaptive_lambda.py --replot     # rebuild figures from saved runs
    VDN_QUICK=1 python examples/10_fetch_adaptive_lambda.py  # plumbing smoke test
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

ROOT = Path(__file__).resolve().parents[1] / "results" / "atd_vs_baselines"
BASELINE_ROOT = Path(__file__).resolve().parents[1] / "results" / "mappo_vs_vdn"
ALGOS = ["qmix", "qmix_lam07", "qmix_atd", "qplex", "qplex_lam07", "qplex_atd", "mappo", "mappo_atd", "vdn"]
ATD = ("qmix_atd", "qplex_atd", "mappo_atd")
cmp.COLORS.update({"qmix": "#7f8c8d", "qmix_atd": "#16a085", "qplex": "#d4a017", "qplex_atd": "#e67e22",
                   "mappo": "#c0392b", "mappo_atd": "#8e44ad",
                   "qmix_lam07": "#2c3e50", "qplex_lam07": "#a0522d"})
SEEDS = cmp.SEEDS
STEPS = budget(200_000)
TRACE_EVERY = 10_000


def variants(steps: int):
    quick_ppo = dict(rollout_len=500, ppo_epochs=2) if QUICK else {}
    quick_atd = dict(atd_warmup=5) if QUICK else {}
    return {
        "qmix": {"agent": "qmix", "agent_kwargs": hp(steps)},
        "qmix_atd": {"agent": "qmix_atd", "agent_kwargs": hp(steps, **quick_atd)},
        "qmix_lam07": {"agent": "qmix", "agent_kwargs": hp(steps, lam=0.7)},
        "qplex": {"agent": "qplex", "agent_kwargs": hp(steps)},
        "qplex_lam07": {"agent": "qplex", "agent_kwargs": hp(steps, lam=0.7)},
        "qplex_atd": {"agent": "qplex_atd", "agent_kwargs": hp(steps, **quick_atd)},
        "mappo": {"agent": "mappo", "agent_kwargs": quick_ppo},
        "mappo_atd": {"agent": "mappo_atd", "agent_kwargs": {**quick_ppo, **quick_atd}},
        "vdn": {"agent": "vdn", "agent_kwargs": hp(steps)},
    }


def _train_one(args):
    """``06_compare_learners._train_one`` plus a trace of the ATD estimator's statistics."""
    game, label, seed, cfg_dict, out_dir = args
    import torch
    torch.set_num_threads(1)
    from vdn_lab.experiment import Callback, Experiment, ExperimentConfig

    trace = []

    class ATDTrace(Callback):
        def on_episode_end(self, exp, rec):
            atd = getattr(exp.agent, "atd", None)
            if atd is not None and atd.last and rec["step"] >= (len(trace) + 1) * TRACE_EVERY:
                trace.append({"step": rec["step"], **{k: round(v, 4) for k, v in atd.last.items()}})

    cfg = ExperimentConfig.from_dict(cfg_dict).replace(name=f"{label}_s{seed}", seed=seed, out_dir=out_dir, verbose=1)
    exp = Experiment(cfg, callbacks=[ATDTrace()])
    res = exp.run()
    res["behaviour"] = cmp.behaviour_stats(exp, episodes=5)
    res["run_dir"] = str(exp.run_dir)
    if trace:
        res["atd_trace"] = trace
    Path(exp.run_dir, "result.json").write_text(json.dumps(res, indent=2))
    return game, label, seed, res


def _load_runs(run_root: Path):
    results = {}
    for f in sorted(run_root.glob("*/result.json")):
        res = json.loads(f.read_text())
        results.setdefault(res["config"]["name"].rsplit("_s", 1)[0], []).append(res)
    return results


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
    # longest jobs first so the pool does not end on one slow QPLEX run
    order = {"qplex_atd": 0, "qplex": 1, "qplex_lam07": 1, "qmix_atd": 2, "qmix": 3, "qmix_lam07": 3, "vdn": 4, "mappo_atd": 5, "mappo": 6}
    jobs.sort(key=lambda j: order.get(j[1], 9))
    results = {m: dict(reused[m]) for m in maps}
    workers = int(os.environ.get("VDN_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        for m, label, seed, res in pool.map(_train_one, jobs):
            results[m].setdefault(label, []).append(res)
        gif_jobs = []
        for m in maps:
            for label, runs in results[m].items():
                best = max(runs, key=lambda r: r["final_eval"]["mean_return"])
                if label in reused[m] or (ROOT / m / f"{label}_best_seed_000.gif").exists():
                    continue
                gif_jobs.append((best["run_dir"], str(ROOT / m / f"{label}_best_seed.gif")))
        list(pool.map(cmp._gif_one, gif_jobs))
    for m in maps:
        report(m, results[m])


def report(m, results):
    out = ROOT / m
    results = {a: sorted(results[a], key=lambda r: r["config"]["seed"]) for a in ALGOS if a in results}
    cmp.plot_game(f"fetch ({m})", results, out, STEPS)
    rows = cmp.summarise_game(f"fetch ({m})", results, out)
    plot_lambda(m, results, out)
    extra = ["", "Paired comparison, ATD(lambda) minus base, greedy return of the final policy (same seeds):", "",
             "| base | ATD - base per seed | mean | train AUC ATD - base |", "|---|---|---|---|"]
    for a in ATD:
        b = a[:-4]
        if a not in results or b not in results:
            continue
        ga = np.array([r["final_eval"]["mean_return"] for r in results[a]])
        gb = np.array([r["final_eval"]["mean_return"] for r in results[b]])
        aa = np.array([r["train_auc"] for r in results[a]])
        ab = np.array([r["train_auc"] for r in results[b]])
        extra.append(f"| {b} | {[round(float(v), 1) for v in ga - gb]} | {(ga - gb).mean():+.1f} | "
                     f"{(aa - ab).mean():+.1f} |")
        if f"{b}_lam07" in results:
            gc = np.array([r["final_eval"]["mean_return"] for r in results[f"{b}_lam07"]])
            ac = np.array([r["train_auc"] for r in results[f"{b}_lam07"]])
            extra.append(f"| {b}_lam07 (fixed lambda 0.7) | {[round(float(v), 1) for v in ga - gc]} | "
                         f"{(ga - gc).mean():+.1f} | {(aa - ac).mean():+.1f} |")
    extra += ["", "ATD(lambda) estimator over training (mean over seeds; acc = accuracy telling the small "
              "buffer from the large one, 0.5 = cannot tell them apart; lo / hi = fraction of lambdas clipped "
              "at lam_min / lam_max):", "",
              "| variant | steps | acc | lambda mean | lambda std | lo | hi |", "|---|---|---|---|---|---|---|"]
    for a in ATD:
        traces = [r.get("atd_trace", []) for r in results.get(a, [])]
        if not traces or not all(traces):
            continue
        n = min(len(t) for t in traces)
        for i in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}):
            mean = {k: np.mean([t[i][k] for t in traces]) for k in ("atd_acc", "lam_mean", "lam_std",
                                                                     "lam_lo_frac", "lam_hi_frac")}
            extra.append(f"| {a} | {traces[0][i]['step']} | {mean['atd_acc']:.2f} | {mean['lam_mean']:.3f} | "
                         f"{mean['lam_std']:.3f} | {mean['lam_lo_frac']:.2f} | {mean['lam_hi_frac']:.2f} |")
    with open(out / "summary.md", "a") as f:
        f.write("\n".join(extra) + "\n")
    print("\n".join(extra))
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


def plot_lambda(m, results, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for a in ATD:
        traces = [r.get("atd_trace", []) for r in results.get(a, [])]
        if not traces or not all(traces):
            continue
        n = min(len(t) for t in traces)
        xs = [traces[0][i]["step"] for i in range(n)]
        for ax, key in zip(axes, ("lam_mean", "atd_acc")):
            v = np.array([[t[i][key] for i in range(n)] for t in traces])
            ax.plot(xs, v.mean(0), color=cmp.COLORS[a], label=a, lw=1.8)
            ax.fill_between(xs, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=cmp.COLORS[a], alpha=0.15, lw=0)
            if key == "lam_mean":
                s = np.array([[t[i]["lam_std"] for i in range(n)] for t in traces]).mean(0)
                ax.plot(xs, v.mean(0) - s, color=cmp.COLORS[a], ls=":", lw=1)
                ax.plot(xs, v.mean(0) + s, color=cmp.COLORS[a], ls=":", lw=1)
    axes[0].set_title(f"fetch ({m}): mean λ of a training batch (dotted: ± its spread)", fontsize=10)
    axes[1].set_title("estimator accuracy, small vs large buffer (0.5 = no difference)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lambda.png", dpi=110)
    plt.close(fig)


def replot(maps):
    for m in maps:
        results = _load_runs(BASELINE_ROOT / m / "runs")
        results.update(_load_runs(ROOT / m / "runs"))
        report(m, results)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    maps = args or ["open", "single"]
    if QUICK:
        ROOT = ROOT.parent / "_quick_atd"
        shutil.rmtree(ROOT, ignore_errors=True)
    replot(maps) if "--replot" in sys.argv else main(maps)
