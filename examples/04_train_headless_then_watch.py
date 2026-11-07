"""Example 4 - visuals: train with NO rendering, then watch the finished agent.

Ways to look at an agent (``eval_visual`` / ``train_visual``):
    "text"    ASCII map + each agent's observation window, in the terminal
    "window"  live matplotlib window (needs a display)
    "gif"     one animated GIF per session (no display needed)

Training stays headless (train_visual="none"); set train_visual_every=N to peek at every
N-th training episode. The trained agent is saved, so you can come back later:

    python -m vdn_lab play runs/<run_dir> --sessions 3 --visual text
"""
from _common import QUICK, budget, hp
from vdn_lab.experiment import Experiment, ExperimentConfig

steps = budget(150_000)
cfg = ExperimentConfig(
    name="fetch_vdn_demo",
    env="fetch", env_kwargs={"layout": "open"}, episode_len=200, eval_episode_len=200,
    agent="vdn", agent_kwargs=hp(steps),
    train_steps=steps, eval_every=steps // 5,
    train_visual="none",                      # headless training
    final_eval_sessions=3,                    # then 3 sessions with the trained agent ...
    eval_visual="gif" if QUICK else "text",   # ... shown as ASCII (or "gif" / "window")
    visual_delay=0.05, visual_every=2,        # seconds per frame, render every 2nd step
)
Experiment(cfg).run()
