from __future__ import annotations

import numpy as np

from ..registry import register_agent
from .base import Agent


@register_agent("random")
class RandomAgent(Agent):
    """Uniformly random actions: a floor for results, a way to eyeball a game, and the smallest
    possible example of implementing ``Agent``."""

    def __init__(self, n_agents, n_actions, obs_shape, device="cpu", seed=None, **kwargs):
        super().__init__(n_agents, n_actions, obs_shape, device, seed, **kwargs)
        self.rng = np.random.default_rng(seed)

    def reset_episode(self) -> None:
        pass

    def act(self, obs, explore: bool = True):
        return self.rng.integers(self.n_actions, size=self.n_agents)

    def observe(self, obs, actions, reward, next_obs, done) -> None:
        pass

    def update(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        pass
