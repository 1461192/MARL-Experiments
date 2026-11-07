"""Example 1 - the paper's headline comparison on Fetch (open map).

Value-decomposition vs independent learners vs a fully centralised (joint-action) agent.
Paper (fig. 4/5): VDN > centralised, independent.

    python examples/01_fetch_vdn_vs_baselines.py        (VDN_QUICK=1 for a smoke test)

Runtime on one CPU core: ~3-5 min per run at 150k steps -> ~40 min for 3 agents x 3 seeds.
"""
from _common import QUICK, budget, hp
from vdn_lab.experiment import Benchmark, ExperimentConfig

steps = budget(150_000)
base = ExperimentConfig(
    env="fetch", env_kwargs={"layout": "open"},          # try "single" (1 corridor) / "double"
    episode_len=200, eval_episode_len=200,               # paper: 5000 / 2000
    train_steps=steps, eval_every=steps // 5, eval_episodes=3, final_eval_sessions=3,
    agent_kwargs=hp(steps),
)
Benchmark(
    base,
    variants={
        "vdn": {"agent": "vdn"},
        "independent": {"agent": "independent"},
        "centralized": {"agent": "centralized"},         # combinatorial 8^2 = 64 joint actions
    },
    seeds=[0] if QUICK else [0, 1, 2],
    name="fetch_open",
).run()          # prints an AUC / final-performance table, writes summary.csv + curves.png
