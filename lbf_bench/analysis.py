"""Tables and figures for the LBF study.

    python -m lbf_bench.analysis --root results/lbf

Metrics (Papoudakis et al. 2021, sec. 4.3): **maximum returns** - the evaluation time step at which the mean
over seeds is highest, reported as mean +/- 95% confidence interval over seeds at that step; **average
returns** - the mean over all 41 evaluations of a run, mean +/- 95% CI over seeds. (The paper re-evaluates
the saved models of the best step; here the step's own 100-episode evaluation is used.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .configs import ALGOS, STUDY_ALGOS
from .env import TASKS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (all pairs)
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
FAMILIES = {"qmix": ["qmix", "qmix_atd", "qmix_l0.4"], "qplex": ["qplex", "qplex_atd", "qplex_l0.4"],
            "mappo": ["mappo", "mappo_atd", "mappo_l0.95"]}
STYLE = {0: dict(color=BLUE, ls="-"), 1: dict(color=ORANGE, ls="-"), 2: dict(color=AQUA, ls="--")}
ATD = ["qmix_atd", "qplex_atd", "mappo_atd"]
MAIN = ["qmix", "qplex", "macpf", "mappo"]


def t_crit(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return table.get(n, 1.96)


def ci95(x: np.ndarray) -> float:
    return float(t_crit(len(x)) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def load(root: Path) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {}
    for f in sorted(root.glob("*/*/seed*/result.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["task"], {}).setdefault(r["tag"], []).append(r)
    return out


def eval_matrix(runs: List[dict]):
    n = min(len(r["eval"]["mean"]) for r in runs)
    t = np.asarray(runs[0]["eval"]["t"][:n], float)
    m = np.asarray([r["eval"]["mean"][:n] for r in runs])
    return t, m


def metrics(runs: List[dict]):
    t, m = eval_matrix(runs)
    best = int(m.mean(0).argmax())
    avg = m.mean(1)
    return dict(max=(m[:, best].mean(), ci95(m[:, best]), t[best]), avg=(avg.mean(), ci95(avg)), n=len(runs))


def fmt(mu, ci):
    return f"{mu:.2f} ± {ci:.2f}"


def tables(data, algos: List[str]) -> str:
    lines = []
    for key, title in (("max", "Maximum returns"), ("avg", "Average returns")):
        lines.append(f"**{title}** (mean ± 95% CI over seeds)\n")
        lines.append("| task | " + " | ".join(ALGOS[a].label for a in algos) + " |")
        lines.append("|---|" + "---|" * len(algos))
        for task in TASKS:
            if task not in data:
                continue
            cells, vals = [], []
            for a in algos:
                runs = data[task].get(a)
                if not runs:
                    cells.append("–"), vals.append(-np.inf)
                    continue
                mt = metrics(runs)[key]
                vals.append(mt[0])
                cells.append(fmt(mt[0], mt[1]) + ("" if len(runs) >= 3 else f" ({len(runs)}s)"))
            best = max(vals)
            cells = [f"**{c}**" if v == best and np.isfinite(v) and v > 0 else c for c, v in zip(cells, vals)]
            lines.append(f"| {task} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def curve_grid(data, algos: List[str], path: Path, title: str, styles=None):
    tasks = [t for t in TASKS if t in data]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), constrained_layout=True)
    axes = axes.ravel()
    handles = {}
    for ax, task in zip(axes, tasks):
        _style(ax)
        for i, a in enumerate(algos):
            runs = data[task].get(a)
            if not runs:
                continue
            t, m = eval_matrix(runs)
            mu = m.mean(0)
            st = (styles or STYLE)[i]
            (h,) = ax.plot(t / 1e6, mu, lw=2, **st)
            if len(runs) > 1:
                ci = t_crit(len(runs)) * m.std(0, ddof=1) / np.sqrt(len(runs))
                ax.fill_between(t / 1e6, mu - ci, mu + ci, color=st["color"], alpha=0.15, lw=0)
            handles[a] = h
        ax.set_title(task, fontsize=10, color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
        ax.set_ylabel("evaluation return", fontsize=8, color=MUTED)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    n = max(len(r) for d in data.values() for k, r in d.items() if k in algos) if data else 0
    axes[-1].legend([handles[a] for a in algos if a in handles],
                    [ALGOS[a].label for a in algos if a in handles], loc="center", frameon=False, fontsize=11,
                    title=f"mean ± 95% CI over {n} seeds", title_fontsize=9)
    fig.suptitle(title, fontsize=13, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lam_matrix(runs: List[dict], col: int):
    """Per-run binned lambda statistics -> common t grid, matrix (seeds, bins)."""
    rows = [np.asarray(r["lam"]) for r in runs if r.get("lam")]
    if not rows:
        return None, None
    t = rows[0][:, 0]
    grid = np.full((len(rows), len(t)), np.nan)
    for i, a in enumerate(rows):
        grid[i] = np.interp(t, a[:, 0], a[:, col])
    return t, grid


def lambda_figure(data, path: Path):
    tasks = [t for t in TASKS if t in data]
    fig, axes = plt.subplots(len(ATD), len(tasks), figsize=(2.6 * len(tasks), 2.4 * len(ATD)),
                             constrained_layout=True, squeeze=False)
    for i, a in enumerate(ATD):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            _style(ax)
            runs = data[task].get(a, [])
            t, mean = lam_matrix(runs, 1)
            if t is not None:
                _, p10 = lam_matrix(runs, 3)
                _, p90 = lam_matrix(runs, 5)
                ax.fill_between(t / 1e6, np.nanmean(p10, 0), np.nanmean(p90, 0), color=BLUE, alpha=0.15, lw=0,
                                label="10th–90th percentile")
                for row in mean:
                    ax.plot(t / 1e6, row, color=BLUE, lw=0.6, alpha=0.45)
                ax.plot(t / 1e6, np.nanmean(mean, 0), color=BLUE, lw=2, label="mean λ")
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(task, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{ALGOS[a].label}\nλ", fontsize=9, color=INK)
            if i == len(ATD) - 1:
                ax.set_xlabel("env steps (M)", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h + [plt.Line2D([], [], color=BLUE, lw=0.6, alpha=0.45)], l + ["individual seeds"],
                   loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Adaptive λ over training (λ of the state-action pairs in each training batch)", fontsize=12,
                 color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accuracy_figure(data, path: Path):
    tasks = [t for t in TASKS if t in data]
    fig, axes = plt.subplots(1, len(ATD), figsize=(5 * len(ATD), 3.4), constrained_layout=True, squeeze=False)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # slots 1-7, fixed order
    for i, a in enumerate(ATD):
        ax = axes[0, i]
        _style(ax)
        for j, task in enumerate(tasks):
            rows = [np.asarray(r["atd"]) for r in data[task].get(a, []) if r.get("atd")]
            if not rows:
                continue
            t = rows[0][:, 0]
            acc = np.nanmean([np.interp(t, r[:, 0], r[:, 2]) for r in rows], 0)
            k = max(1, len(t) // 40)
            sm = np.convolve(acc, np.ones(k) / k, mode="same")
            ax.plot(t / 1e6, sm, color=colors[j], lw=1.6, label=task)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(ALGOS[a].label, fontsize=10, color=INK)
        ax.set_xlabel("env steps (M)", fontsize=8, color=MUTED)
        ax.set_ylabel("estimator accuracy (D_on vs D_off)", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def lambda_table(data) -> str:
    lines = ["| task | " + " | ".join(f"{ALGOS[a].label}: early / mid / late" for a in ATD) + " |",
             "|---|" + "---|" * len(ATD)]
    for task in TASKS:
        if task not in data:
            continue
        cells = []
        for a in ATD:
            t, m = lam_matrix(data[task].get(a, []), 1)
            if t is None:
                cells.append("–")
                continue
            mu = np.nanmean(m, 0)
            k = len(mu) // 10
            cells.append(f"{np.nanmean(mu[:k]):.2f} / {np.nanmean(mu[4 * k:6 * k]):.2f} / {np.nanmean(mu[-k:]):.2f}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


YELLOW = "#eda100"                                        # categorical slot 4
FAMILY_COLOR = {"qmix": BLUE, "qplex": ORANGE, "mappo": AQUA, "macpf": YELLOW}
VARIANT_LS = {"base": "-", "atd": (0, (5, 2)), "fixed": (0, (1, 1.5))}
ALL_ALGOS = [("qmix", "base"), ("qmix_atd", "atd"), ("qmix_l0.4", "fixed"),
             ("qplex", "base"), ("qplex_atd", "atd"), ("qplex_l0.4", "fixed"),
             ("mappo", "base"), ("mappo_atd", "atd"), ("mappo_l0.95", "fixed"), ("macpf", "base")]


def _family(a: str) -> str:
    return a.split("_")[0]


def curves_all(data, path: Path):
    """Every algorithm on every task in one figure. x = fraction of the algorithm's own training budget
    (2M steps off-policy, 20M on-policy), so learners with different budgets share an axis. Colour = base
    algorithm, line style = variant (solid base, dashed ATD, dotted fixed lambda). Mean over seeds only;
    the per-family figures carry the confidence bands."""
    tasks = [t for t in TASKS if t in data]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), constrained_layout=True)
    axes = axes.ravel()
    handles = {}
    for ax, task in zip(axes, tasks):
        _style(ax)
        for a, var in ALL_ALGOS:
            runs = data[task].get(a)
            if not runs:
                continue
            t, m = eval_matrix(runs)
            (h,) = ax.plot(100 * t / runs[0]["t_max"], m.mean(0), color=FAMILY_COLOR[_family(a)],
                           ls=VARIANT_LS[var], lw=1.8 if var == "base" else 1.5)
            handles[a] = h
        ax.set_title(task, fontsize=10, color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of training budget", fontsize=8, color=MUTED)
        ax.set_ylabel("evaluation return", fontsize=8, color=MUTED)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    order = [a for a, _ in ALL_ALGOS if a in handles]
    axes[-1].legend([handles[a] for a in order], [ALGOS[a].label for a in order], loc="center",
                    frameon=False, fontsize=10, ncol=1,
                    title="mean over 3 seeds\n100% of budget = 2M steps (QMIX,\nQPLEX, MACPF) or 20M (MAPPO)",
                    title_fontsize=8.5)
    fig.suptitle("All algorithms on the seven LBF tasks", fontsize=13, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def lambda_all(data, path: Path):
    """Adaptive lambda of the three ATD learners in one axes: mean over all tasks and seeds (line) and the
    range of the per-run means (band), against the fraction of each learner's training budget."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    _style(ax)
    x = np.linspace(0, 100, 201)
    for a in ATD:
        rows = []
        for task in TASKS:
            for r in data.get(task, {}).get(a, []):
                lam = np.asarray(r["lam"])
                rows.append(np.interp(x, 100 * lam[:, 0] / r["t_max"], lam[:, 1]))
        if not rows:
            continue
        rows = np.asarray(rows)
        c = FAMILY_COLOR[_family(a)]
        ax.fill_between(x, rows.min(0), rows.max(0), color=c, alpha=0.18, lw=0)
        ax.plot(x, rows.mean(0), color=c, lw=4 if a == "mappo_atd" else 1.8, zorder=5 - ATD.index(a), label=f"{ALGOS[a].label} ({len(rows)} runs)")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of training budget (2M steps for QMIX/QPLEX-ATD, 20M for MAPPO-ATD)", fontsize=9,
                  color=MUTED)
    ax.set_ylabel("mean λ in training batch", fontsize=9, color=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              title="line: mean over 7 tasks × 3 seeds\nband: min–max of per-run means", title_fontsize=8)
    ax.set_title("Adaptive λ over training, all ATD learners", fontsize=11, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/lbf")
    a = p.parse_args(argv)
    root = Path(a.root)
    data = load(root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_grid(data, ["qmix", "qplex", "macpf"], fig_dir / "curves_value_based.png",
               "QMIX, QPLEX and MACPF on LBF (2M steps)")
    for base, algos in FAMILIES.items():
        curve_grid(data, algos, fig_dir / f"curves_{base}_atd.png",
                   f"{ALGOS[base].label}: base vs adaptive TD(λ) vs fixed λ")
    lambda_figure(data, fig_dir / "lambda_over_time.png")
    accuracy_figure(data, fig_dir / "atd_estimator_accuracy.png")
    curves_all(data, fig_dir / "curves_all.png")
    lambda_all(data, fig_dir / "lambda_all.png")
    md = ["# LBF results\n", "## Main algorithms\n", tables(data, MAIN)]
    for base, algos in FAMILIES.items():
        md += [f"## {ALGOS[base].label}: adaptive vs fixed λ\n", tables(data, algos)]
    md += ["## Mean adaptive λ (first 10% / middle 20% / last 10% of training, mean over seeds)\n", lambda_table(data), ""]
    wall = {}
    for task, d in data.items():
        for k, runs in d.items():
            wall.setdefault(k, []).extend(r["wall_time"] / 60 for r in runs)
    md += ["## Wall time per run (minutes, one CPU core)\n", "| algorithm | mean | runs |", "|---|---|---|"]
    md += [f"| {ALGOS[k].label} | {np.mean(v):.0f} | {len(v)} |" for k, v in sorted(wall.items(), key=lambda x: STUDY_ALGOS.index(x[0]) if x[0] in STUDY_ALGOS else 99)]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print((root / "summary.md").read_text())


if __name__ == "__main__":
    main()
