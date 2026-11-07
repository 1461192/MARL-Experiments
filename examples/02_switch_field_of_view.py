"""Example 2 - game settings: how does the field of view change things in Switch?

Same algorithm (VDN, shared weights + role id), same single-corridor map, different
egocentric windows. The paper uses 4 ahead / 2 to each side (3x5x5 observations).

    "paper_4x2"    : 4 ahead, 2 side, 0 behind -> (3, 5, 5)
    "short_2x1"    : 2 ahead, 1 side           -> (3, 3, 3)  very short-sighted
    "behind_4x2+2" : 4 ahead, 2 side, 2 behind -> (3, 7, 5)  can also see backwards

Any other game kwarg (layout, rewards, ...) can be varied the same way.
"""
from _common import QUICK, budget, hp
from vdn_lab.experiment import Benchmark, ExperimentConfig

steps = budget(100_000)
base = ExperimentConfig(
    env="switch", env_kwargs={"layout": "single"},
    episode_len=200, eval_episode_len=200,
    agent="vdn", agent_kwargs=hp(steps, share_weights=True, role_info=True),
    train_steps=steps, eval_every=steps // 4, eval_episodes=3, final_eval_sessions=3,
)
Benchmark(
    base,
    variants={
        "paper_4x2": {},
        "short_2x1": {"env_kwargs": {"view_ahead": 2, "view_side": 1}},
        "behind_4x2+2": {"env_kwargs": {"view_behind": 2}},
    },
    seeds=[0] if QUICK else [0, 1],
    name="switch_fov",
).run()
