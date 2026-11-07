"""Compare several *variants* (algorithms, hyper-parameters, game settings such as the field
of view ...) over several seeds and summarise like the paper: area under the training curve
(normalised by the best variant) and final performance."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import ExperimentConfig
from .runner import Experiment


class Benchmark:
    """
    >>> base = ExperimentConfig(env="fetch", env_kwargs={"layout": "open"}, train_steps=30_000)
    >>> bench = Benchmark(base, variants={
    ...     "vdn": {"agent": "vdn"},
    ...     "independent": {"agent": "independent"},
    ...     "vdn_fov2": {"agent": "vdn", "env_kwargs": {"view_ahead": 2}},   # any config field
    ... }, seeds=[0, 1, 2])
    >>> summary = bench.run()

    A variant is a dict of ``ExperimentConfig`` overrides (``env_kwargs``/``agent_kwargs`` are
    merged into the base values). New algorithms need no changes here: register the agent, name it.
    """

    def __init__(self, base: ExperimentConfig, variants: Dict[str, Dict[str, Any]],
                 seeds: Sequence[int] = (0,), out_dir: Optional[str] = "runs/benchmarks",
                 name: str = "benchmark"):
        self.base, self.variants, self.seeds = base, variants, list(seeds)
        self.name = name
        self.out = Path(out_dir) / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}" if out_dir else None
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def run(self) -> List[Dict[str, Any]]:
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        for label, overrides in self.variants.items():
            self.results[label] = []
            for seed in self.seeds:
                cfg = self.base.replace(**overrides).replace(
                    name=f"{label}_s{seed}", seed=seed,
                    out_dir=str(self.out / "runs") if self.out else None)
                self.results[label].append(Experiment(cfg).run())
        summary = self.summarise()
        self.print_table(summary)
        if self.out:
            self._save(summary)
        return summary

    def summarise(self) -> List[Dict[str, Any]]:
        rows = []
        for label, runs in self.results.items():
            auc = np.array([r["train_auc"] for r in runs])
            fin = np.array([r["train_final"] for r in runs])
            ev = np.array([r["final_eval"]["mean_return"] for r in runs if "final_eval" in r])
            rows.append(dict(variant=label, n_seeds=len(runs), auc_mean=auc.mean(), auc_std=auc.std(),
                             train_final_mean=fin.mean(), train_final_std=fin.std(),
                             eval_mean=float(ev.mean()) if len(ev) else float("nan"),
                             eval_std=float(ev.std()) if len(ev) else float("nan")))
        best_auc = max((abs(r["auc_mean"]) for r in rows), default=1.0) or 1.0
        best_fin = max((abs(r["train_final_mean"]) for r in rows), default=1.0) or 1.0
        for r in rows:                               # paper-style normalisation by the best variant
            r["auc_norm"] = r["auc_mean"] / best_auc
            r["final_norm"] = r["train_final_mean"] / best_fin
        return rows

    @staticmethod
    def print_table(rows: List[Dict[str, Any]]) -> None:
        w = max([len(r["variant"]) for r in rows] + [7])
        print("\n" + f"{'variant':<{w}}  seeds  {'AUC(norm)':>10}  {'train-final':>14}  {'eval':>14}")
        for r in sorted(rows, key=lambda r: -r["auc_norm"]):
            print(f"{r['variant']:<{w}}  {r['n_seeds']:>5}  {r['auc_norm']:>10.3f}  "
                  f"{r['train_final_mean']:>8.2f}±{r['train_final_std']:<5.2f}  "
                  f"{r['eval_mean']:>8.2f}±{r['eval_std']:<5.2f}")

    def _save(self, rows: List[Dict[str, Any]]) -> None:
        with open(self.out / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        (self.out / "summary.json").write_text(json.dumps(
            {"summary": rows, "base": self.base.to_dict(), "variants": self.variants,
             "seeds": self.seeds}, indent=2))
        try:
            self.plot(self.out / "curves.png")
        except ImportError:
            pass

    def plot(self, path, smooth: int = 5) -> None:
        """Training-return curves (mean ± std over seeds) for every variant."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, runs in self.results.items():
            n = min(len(r["train_curve"]) for r in runs)
            ys = np.array([[x["return"] for x in r["train_curve"][:n]] for r in runs])
            xs = np.array([x["step"] for x in runs[0]["train_curve"][:n]])
            k = min(smooth, n)
            kern = np.ones(k) / k
            sm = np.array([np.convolve(y, kern, mode="valid") for y in ys])
            xs = xs[k - 1:]
            m, s = sm.mean(0), sm.std(0)
            ax.plot(xs, m, label=label)
            ax.fill_between(xs, m - s, m + s, alpha=0.15)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("team return / episode")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
