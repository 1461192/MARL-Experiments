"""Run a grid of (algorithm, task, seed) jobs on N worker processes; finished runs are skipped.

Jobs are ordered by seed, so all of seed 0 finishes before seed 1 starts, and longest-first within a seed.

    python -m lbf_bench.sweep --seeds 0 1 2 --workers 4 --out results/lbf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .configs import ALGOS, STUDY_ALGOS
from .env import TASKS

# rough relative cost (minutes on the 15x15-4p-5f task), used only to order jobs
COST = {"macpf": 150, "mappo_atd": 120, "mappo_l0.95": 112, "mappo": 100, "qplex": 79, "qplex_atd": 75,
        "qplex_l0.4": 75, "qmix": 52, "qmix_atd": 52, "qmix_l0.4": 52}
TASK_COST = {t: (s.players * s.size) for t, s in TASKS.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="*", default=STUDY_ALGOS)
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/lbf")
    a = p.parse_args(argv)
    for x in a.algos:
        assert x in ALGOS, x
    jobs = []
    for seed in a.seeds:
        js = [(algo, task, seed) for algo in a.algos for task in a.tasks
              if not (Path(a.out) / task / algo / f"seed{seed}" / "result.json").exists()]
        started = lambda j: (Path(a.out) / j[1] / j[0] / f"seed{j[2]}" / "ckpt.pt").exists()
        js.sort(key=lambda j: (not started(j), -COST.get(j[0], 60) * TASK_COST[j[1]]))   # resume first
        jobs += js
    print(f"{len(jobs)} jobs to run", flush=True)
    logs = Path(a.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        algo, task, seed = job
        log = logs / f"{task}__{algo}__s{seed}.log"
        t0 = time.time()
        with open(log, "a") as f:
            rc = subprocess.call([sys.executable, "-m", "lbf_bench.run", "--algo", algo, "--task", task,
                                  "--seed", str(seed), "--out", a.out], stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {'ok ' if rc == 0 else f'FAIL({rc})'} {task} {algo} s{seed} "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
