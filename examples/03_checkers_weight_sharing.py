"""Example 3 - Checkers: weight sharing vs role information (paper sec. 4.3).

The two players have very different reward scales (10x vs 1x), so one shared network that
cannot tell them apart struggles; adding a one-hot role id restores specialisation.

    vdn              separate networks per agent
    vdn+share        one shared (agent-invariant) network
    vdn+share+role   shared network + one-hot role id
"""
from _common import QUICK, budget, hp
from vdn_lab.experiment import Benchmark, ExperimentConfig

steps = budget(100_000)
base = ExperimentConfig(
    env="checkers", episode_len=200, eval_episode_len=200,
    agent="vdn", agent_kwargs=hp(steps),
    train_steps=steps, eval_every=steps // 4, eval_episodes=3, final_eval_sessions=3,
)
Benchmark(
    base,
    variants={
        "vdn": {},
        "vdn+share": {"agent_kwargs": {"share_weights": True}},
        "vdn+share+role": {"agent_kwargs": {"share_weights": True, "role_info": True}},
    },
    seeds=[0] if QUICK else [0, 1],
    name="checkers_sharing",
).run()
