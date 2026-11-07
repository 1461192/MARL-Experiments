"""Example 5 - add a new algorithm in a few lines.

Subclass ``Agent`` (or an existing agent), register it, and it works with the CLI, the runner
and ``Benchmark`` without touching framework code. Here: a hand-written scripted baseline and
a VDN ablation with 1-step targets (lambda = 0).
"""
import numpy as np
from _common import QUICK, budget, hp
from vdn_lab import register_agent
from vdn_lab.agents import Agent, VDNAgent
from vdn_lab.experiment import Benchmark, ExperimentConfig


@register_agent("forward_turner")
class ForwardTurner(Agent):
    """Walk forward; rotate right when the square ahead is a wall (reads the raw view)."""

    def reset_episode(self):
        pass

    def act(self, obs, explore=True):
        wall = [tuple(o[:, 3, 2]) == (110, 110, 110) for o in obs]   # square directly in front
        return np.array([5 if w else 0 for w in wall])               # 5 = rotate right, 0 = forward

    def observe(self, *a, **k):
        pass

    def update(self):
        return {}


@register_agent("vdn_lam0")
class VDNOneStep(VDNAgent):
    """VDN with plain 1-step targets - an ablation of the eligibility trace."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("lam", 0.0)
        super().__init__(*args, **kwargs)


steps = budget(100_000)
base = ExperimentConfig(env="fetch", env_kwargs={"layout": "open"}, episode_len=200, eval_episode_len=200,
                        train_steps=steps, eval_every=0, final_eval_sessions=3)
Benchmark(base, variants={
    "vdn": {"agent": "vdn", "agent_kwargs": hp(steps)},
    "vdn_lam0": {"agent": "vdn_lam0", "agent_kwargs": hp(steps)},
    "forward_turner": {"agent": "forward_turner"},
}, seeds=[0], name="custom_agents").run()
