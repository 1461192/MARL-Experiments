"""Can the ATD(lambda) estimator tell D_on from D_off at all? A controlled check on real LBF episodes.

    python -m lbf_bench.atd_diagnostic --task 10x10-3p-3f --steps 2000

D_off is always collected by the uniform-random policy. D_on is collected by a policy that, with probability
``p``, takes one fixed action (``load``) and otherwise acts uniformly, so ``p`` sets how far apart the two
state-action distributions are (p = 0: identical). The estimator (the same ``ATDEstimator`` the study uses:
MLP-GRU(64), Adam 1e-3, BCE with balanced batches of 32 episodes) is trained for ``--steps`` steps, and its
held-out accuracy and the spread of lambda = sigmoid(omega) are reported at several step counts. 200 steps
is about what QMIX-/QPLEX-ATD get in a whole 2M-step run (one step per 200 episodes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .algos import ATDEstimator
from .buffer import Collector, EpisodeBuffer
from .env import LBFEnv, make_envs

LOAD = 5                                  # lbforaging action: 0 noop, 1-4 move, 5 load


class BiasedRandom:
    def __init__(self, n_actions: int, p: float, seed: int):
        self.A, self.p, self.rng = n_actions, p, np.random.default_rng(seed)

    def init_hidden(self, E):
        pass

    def act(self, obs, explore, t_env):
        a = self.rng.integers(0, self.A, obs.shape[:2])
        return np.where(self.rng.random(obs.shape[:2]) < self.p, LOAD, a)


def fill(task: str, p: float, episodes: int, seed: int, env: LBFEnv) -> EpisodeBuffer:
    buf = EpisodeBuffer(episodes, env.episode_limit, env.n_agents, env.obs_dim, np.random.default_rng(seed))
    col = Collector(make_envs(task, 10, seed))
    pol = BiasedRandom(env.n_actions, p, seed)
    while buf.size < episodes:
        buf.insert(col.run(pol, explore=True, t_env=0))
    return buf


@torch.no_grad()
def evaluate(est: ATDEstimator, on, off):
    lam_on, lam_off = est.lambdas(on), est.lambdas(off)
    m_on, m_off = on["filled"] > 0, off["filled"] > 0
    lo, lf = lam_on.numpy()[m_on], lam_off.numpy()[m_off]
    acc = 0.5 * ((lo > 0.5).mean() + (lf <= 0.5).mean())
    allv = np.concatenate([lo, lf])
    return dict(acc=float(acc), lam_mean=float(allv.mean()), lam_std=float(allv.std()),
                lam_on=float(lo.mean()), lam_off=float(lf.mean()))


def run(task: str, ps, steps: int, checkpoints, seed: int):
    torch.manual_seed(seed)
    env = LBFEnv(task)
    off_tr, off_te = fill(task, 0.0, 500, seed, env), fill(task, 0.0, 200, seed + 1, env)
    out = {}
    for p in ps:
        on_tr, on_te = fill(task, p, 500, seed + 2, env), fill(task, p, 200, seed + 3, env)
        est = ATDEstimator(env.n_agents * env.obs_dim, env.n_agents, env.n_actions, 64, 1e-3)
        te_on, te_off = on_te.get(np.arange(200)), off_te.get(np.arange(200))
        rows = []
        for s in range(1, steps + 1):
            on = on_tr.get(on_tr.sample_idx(32))
            off = off_tr.get(off_tr.sample_idx(32))
            est.step(on, off, env.n_agents)
            if s in checkpoints:
                rows.append(dict(step=s, **evaluate(est, te_on, te_off)))
                print(f"p={p:.2f} step {s:5d}: " + ", ".join(f"{k} {v:.3f}" for k, v in rows[-1].items()
                                                              if k != "step"), flush=True)
        out[str(p)] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="10x10-3p-3f")
    ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.05, 0.2, 1.0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/lbf/atd_diagnostic.json")
    a = ap.parse_args(argv)
    cps = sorted({c for c in (50, 200, 500, 1000, 2000, a.steps) if c <= a.steps})
    res = dict(task=a.task, steps=a.steps, seed=a.seed, results=run(a.task, a.p, a.steps, cps, a.seed))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
