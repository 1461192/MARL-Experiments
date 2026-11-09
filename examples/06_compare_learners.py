"""Example 6 - VDN vs independent vs centralised learners on Fetch, Switch and Checkers.

For every game, trains the three learners over several seeds (in parallel processes), then
plots the training return (mean ± std over seeds), the periodic greedy evaluations and the
per-agent share of the reward, and records behaviour statistics + one GIF per learner.

    python examples/06_compare_learners.py                    # all games
    python examples/06_compare_learners.py fetch checkers     # a subset
    VDN_QUICK=1 python examples/06_compare_learners.py        # plumbing smoke test

Checkers gives every learner role information: VDN and the independent learners use one
shared network with a one-hot role id appended to each observation; the centralised learner
encodes each player's observation with its own (role-specific) input layer and owns the
joint action, so the role is part of its architecture.

Results go to ``results/learner_comparison/<game>/``.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _common import QUICK, budget, hp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "learner_comparison"
ALGOS = ["vdn", "independent", "centralized"]
COLORS = {"vdn": "#2a6fdb", "independent": "#e07b12", "centralized": "#3a9d5d"}
SEEDS = [0] if QUICK else [0, 1, 2]


def game_setups():
    s_fetch, s_switch, s_check = budget(200_000), budget(200_000), budget(150_000)
    return {
        "fetch": dict(
            steps=s_fetch,
            base=dict(env="fetch", env_kwargs={"layout": "open"}, agent_kwargs=hp(s_fetch)),
            variants={a: {"agent": a} for a in ALGOS},
        ),
        "switch": dict(
            steps=s_switch,
            base=dict(env="switch", env_kwargs={"layout": "single"}, agent_kwargs=hp(s_switch)),
            variants={a: {"agent": a} for a in ALGOS},
        ),
        "checkers": dict(
            steps=s_check,
            base=dict(env="checkers", env_kwargs={"layout": "default"}, agent_kwargs=hp(s_check)),
            variants={
                "vdn": {"agent": "vdn", "agent_kwargs": {"share_weights": True, "role_info": True}},
                "independent": {"agent": "independent",
                                "agent_kwargs": {"share_weights": True, "role_info": True}},
                "centralized": {"agent": "centralized"},     # per-player encoders = role-aware
            },
        ),
    }


# ----------------------------------------------------------------------------- workers
def _train_one(args):
    game, label, seed, cfg_dict, out_dir = args
    import torch
    torch.set_num_threads(1)
    from vdn_lab.experiment import Experiment, ExperimentConfig
    cfg = ExperimentConfig.from_dict(cfg_dict).replace(
        name=f"{label}_s{seed}", seed=seed, out_dir=out_dir, verbose=1)
    exp = Experiment(cfg)
    res = exp.run()
    res["behaviour"] = behaviour_stats(exp, episodes=5)
    res["run_dir"] = str(exp.run_dir)
    Path(exp.run_dir, "result.json").write_text(json.dumps(res, indent=2))
    return game, label, seed, res


def behaviour_stats(exp, episodes: int = 5):
    """Greedy sessions with per-agent action histograms, reward shares and coverage."""
    from vdn_lab.envs.base import ACTION_NAMES
    env, agent = exp.eval_env, exp.agent
    agent.set_training(False)
    n = env.n_agents
    acts = np.zeros((n, len(ACTION_NAMES)))
    moved = np.zeros(n)
    cells = [set() for _ in range(n)]
    rewards = np.zeros(n)
    stuck = np.zeros(n)            # steps where the agent did not change position or orientation
    events = np.zeros(n)           # reward events (pick-up/drop, goal, fruit)
    for _ in range(episodes):
        obs = env.reset()
        agent.reset_episode()
        done = False
        while not done:
            before = [(tuple(p), d) for p, d in zip(env.pos, env.dirs)]
            a = agent.act(obs, explore=False)
            obs, _, done, info = env.step(a)
            for i in range(n):
                acts[i, int(a[i])] += 1
                cells[i].add(tuple(env.pos[i]))
                if before[i][0] != tuple(env.pos[i]):
                    moved[i] += 1
                if before[i] == (tuple(env.pos[i]), env.dirs[i]):
                    stuck[i] += 1
                if info["agent_rewards"][i] != 0:
                    events[i] += 1
            rewards += np.asarray(info["agent_rewards"])
    steps = acts.sum(1, keepdims=True)
    return {
        "action_names": ACTION_NAMES,
        "action_freq": (acts / steps).round(3).tolist(),
        "move_rate": (moved / steps[:, 0]).round(3).tolist(),
        "idle_rate": (stuck / steps[:, 0]).round(3).tolist(),
        "cells_visited": [len(c) for c in cells],
        "agent_return": (rewards / episodes).round(2).tolist(),
        "reward_events_per_ep": (events / episodes).round(1).tolist(),
    }


def _gif_one(args):
    run_dir, out_path = args
    import torch
    torch.set_num_threads(1)
    from vdn_lab.experiment import Experiment
    exp = Experiment.from_run_dir(run_dir, verbose=0, eval_episode_len=120)
    exp.run_dir = Path(out_path).parent
    exp.cfg.visual_dir = str(Path(out_path).parent)
    exp.evaluate(1, visual="gif", label=Path(out_path).stem)


# ---------------------------------------------------------------------------- plotting
def plot_game(game, results, out: Path, steps: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    for label, runs in results.items():
        n = min(len(r["train_curve"]) for r in runs)
        ys = np.array([[x["return"] for x in r["train_curve"][:n]] for r in runs])
        xs = np.array([x["step"] for x in runs[0]["train_curve"][:n]])
        k = min(10, n)
        kern = np.ones(k) / k
        sm = np.array([np.convolve(y, kern, mode="valid") for y in ys])
        xs = xs[k - 1:]
        m, s = sm.mean(0), sm.std(0)
        ax.plot(xs, m, label=label, color=COLORS[label], lw=1.8)
        ax.fill_between(xs, m - s, m + s, alpha=0.15, color=COLORS[label], lw=0)
    ax.set_title(f"{game}: training return (ε-greedy, mean ± std over {len(SEEDS)} seeds)", fontsize=10)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("team return / 200-step episode")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    for label, runs in results.items():
        n = min(len(r["eval_curve"]) for r in runs)
        if n == 0:
            continue
        ys = np.array([[x["mean_return"] for x in r["eval_curve"][:n]] for r in runs])
        xs = np.array([x["step"] for x in runs[0]["eval_curve"][:n]])
        m, s = ys.mean(0), ys.std(0)
        ax.errorbar(xs, m, yerr=s, label=label, color=COLORS[label], marker="o", ms=4, capsize=3, lw=1.5)
    ax.set_title("greedy evaluation return during training", fontsize=10)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("team return / 200-step episode")
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = list(results)
    x = np.arange(len(labels))
    w = 0.36
    shares = np.array([[r["behaviour"]["agent_return"] for r in results[l]] for l in labels])  # (A,S,2)
    roles = game == "checkers"            # agents have fixed roles; elsewhere they are interchangeable
    if roles:
        names = ["agent 0 (apple +10 / lemon -10)", "agent 1 (apple +1 / lemon -1)"]
    else:                                 # sort per seed so "the agent doing the work" lines up
        shares = -np.sort(-shares, axis=-1)
        names = ["busier agent", "less busy agent"]
    for j, hatch in enumerate(["", "//"]):
        m, s = shares[:, :, j].mean(1), shares[:, :, j].std(1)
        ax.bar(x + (j - 0.5) * w, m, w, yerr=s, capsize=3, color=[COLORS[l] for l in labels],
               hatch=hatch, edgecolor="white", alpha=0.95 if j == 0 else 0.6, label=names[j])
    ax.set_xticks(x, labels)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("final greedy policy: reward earned by each agent" +
                 ("" if roles else " (sorted per seed)"), fontsize=10)
    ax.set_ylabel("reward / episode")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "returns.png", dpi=130)
    plt.close(fig)


def summarise_game(game, results, out: Path):
    rows = []
    for label, runs in results.items():
        auc = np.array([r["train_auc"] for r in runs])
        fin = np.array([r["train_final"] for r in runs])
        ev = np.array([r["final_eval"]["mean_return"] for r in runs])
        beh = [r["behaviour"] for r in runs]
        rows.append(dict(
            variant=label, seeds=len(runs),
            train_auc=round(auc.mean(), 2), train_auc_std=round(auc.std(), 2),
            train_final=round(fin.mean(), 2), train_final_std=round(fin.std(), 2),
            greedy_final=round(ev.mean(), 2), greedy_final_std=round(ev.std(), 2),
            greedy_per_seed=[round(float(v), 1) for v in ev],
            agent_return_per_seed=[b["agent_return"] for b in beh],
            idle_rate_per_seed=[b["idle_rate"] for b in beh],
            beam_freq_per_seed=[[round(f[6], 3) for f in b["action_freq"]] for b in beh],
            noop_freq_per_seed=[[round(f[7], 3) for f in b["action_freq"]] for b in beh],
            cells_visited_per_seed=[b["cells_visited"] for b in beh],
        ))
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = [f"# {game}", "", "| learner | train AUC | train final 10% | greedy final | greedy per seed | "
             "per-agent reward (greedy) per seed |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['train_auc']} ± {r['train_auc_std']} | {r['train_final']} ± "
                     f"{r['train_final_std']} | {r['greedy_final']} ± {r['greedy_final_std']} | "
                     f"{r['greedy_per_seed']} | {r['agent_return_per_seed']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def main(games):
    setups = game_setups()
    from vdn_lab.experiment import ExperimentConfig
    jobs = []
    for game in games:
        st = setups[game]
        steps = st["steps"]
        base = ExperimentConfig(episode_len=200, eval_episode_len=200, train_steps=steps,
                                eval_every=steps // 8, eval_episodes=3, final_eval_sessions=5,
                                save_agent=True, **st["base"])
        out = ROOT / game
        out.mkdir(parents=True, exist_ok=True)
        for label, ov in st["variants"].items():
            cfg = base.replace(**ov)
            for seed in SEEDS:
                jobs.append((game, label, seed, cfg.to_dict(), str(out / "runs")))
    results = {g: {} for g in games}
    workers = int(os.environ.get("VDN_WORKERS", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        for game, label, seed, res in pool.map(_train_one, jobs):
            results[game].setdefault(label, []).append(res)
        gif_jobs = []
        for game in games:
            for label, runs in results[game].items():
                best = max(runs, key=lambda r: r["final_eval"]["mean_return"])
                gif_jobs.append((best["run_dir"], str(ROOT / game / f"{label}_best_seed.gif")))
        list(pool.map(_gif_one, gif_jobs))
    for game in games:
        out = ROOT / game
        plot_game(game, results[game], out, setups[game]["steps"])
        summarise_game(game, results[game], out)


def replot(games):
    """Rebuild figures + tables from the saved ``runs/*/result.json`` (no training)."""
    for game in games:
        out = ROOT / game
        results = {}
        for f in sorted((out / "runs").glob("*/result.json")):
            res = json.loads(f.read_text())
            label = res["config"]["name"].rsplit("_s", 1)[0]
            results.setdefault(label, []).append(res)
        results = {a: results[a] for a in ALGOS if a in results}
        plot_game(game, results, out, 0)
        summarise_game(game, results, out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    games = args or ["fetch", "switch", "checkers"]
    replot(games) if "--replot" in sys.argv else main(games)
