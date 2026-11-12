"""Example 10 - MACPF on the matrix game of its paper (sec. 5.1, fig. 4a).

Two agents, four actions, two optimal joint actions (A, A) and (D, D) worth 8, surrounded by
penalties. Settings follow appendix C.1 where it gives them (lr 3e-4, alpha 1 -> 0.5, batch 64,
10k steps); the game is one round per episode, so gamma = 0 and one-step targets.

For every seed it prints the greedy return of the *independent* policies (decentralised
execution) and of the *dependent* policies (agent 1 sees agent 0's action).

    python examples/10_macpf_matrix_game.py            # 8 seeds, ~2 min on 4 cores
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

from _common import QUICK  # noqa: E402

PAYOFFS = [[8, -20, -20, -20],
           [-12, 0, 0, -20],
           [-12, 0, 0, -20],
           [-12, -12, -12, 8]]
STEPS = 500 if QUICK else 10_000


def run(seed: int):
    import torch
    torch.set_num_threads(1)
    from vdn_lab.experiment import Experiment, ExperimentConfig
    kw = dict(lr=3e-4, critic_lr=3e-4, batch_size=64, alpha_start=1.0, alpha_end=0.5, alpha_anneal_steps=STEPS // 2,
              gamma=0.0, td_lambda=0.0, target_update_interval=20, eps_anneal_steps=STEPS // 5)
    cfg = ExperimentConfig(name="matrix", env="matrix", env_kwargs={"payoffs": PAYOFFS}, agent="macpf", seed=seed,
                           agent_kwargs=kw, episode_len=1, eval_episode_len=1, train_steps=STEPS, eval_every=0,
                           final_eval_sessions=1, out_dir=None, verbose=0)
    exp = Experiment(cfg)
    ind = exp.run()["final_eval"]["mean_return"]
    exp.agent.cfg.eval_policy = "dep"
    dep = exp.evaluate(1)["mean_return"]
    return seed, ind, dep


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else (1 if QUICK else 8)
    with ProcessPoolExecutor(int(os.environ.get("VDN_WORKERS", os.cpu_count() or 1))) as pool:
        res = list(pool.map(run, range(n)))
    print("| seed | independent (decentralised) | dependent |\n|---|---|---|")
    for seed, ind, dep in res:
        print(f"| {seed} | {ind:.0f} | {dep:.0f} |")
    print(f"optimal (8): independent {sum(r[1] == 8 for r in res)}/{n}, dependent {sum(r[2] == 8 for r in res)}/{n}")
