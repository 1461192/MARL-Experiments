"""The agent interface every algorithm in this testbed implements.

The runner only talks to this class, so adding an algorithm (QMIX, a policy-gradient
method, a scripted baseline...) means subclassing ``Agent`` and decorating it with
``@register_agent("name")``.

An agent controls *all* players of a game at once: ``act`` receives every player's local
observation and returns every player's action. Decentralised algorithms (independent, VDN)
compute each action only from that player's own observation history; centralised ones may
look at everything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Agent(ABC):
    registry_name = "agent"

    def __init__(self, n_agents: int, n_actions: int, obs_shape: Tuple[int, ...],
                 device: str = "cpu", seed: Optional[int] = None, **kwargs: Any):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_shape = tuple(obs_shape)
        self.device = device
        self.seed = seed
        self.training = True
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected arguments: {sorted(kwargs)}")

    # ------------------------------------------------------------- required
    @abstractmethod
    def reset_episode(self) -> None:
        """Called at the start of every episode (e.g. clear recurrent state)."""

    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """obs: uint8 ``(n_agents, *obs_shape)`` -> int actions ``(n_agents,)``.
        ``explore=False`` must act (near-)greedily; used for evaluation."""

    @abstractmethod
    def observe(self, obs: np.ndarray, actions: np.ndarray, reward: float,
                next_obs: np.ndarray, done: bool) -> None:
        """Receive one transition of the *team* reward. Ignored when not training."""

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Run a learning step if one is due. Return metrics (``{}`` if nothing happened)."""

    # ------------------------------------------------------------- optional
    def set_training(self, mode: bool) -> None:
        self.training = mode

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support saving")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support loading")

    def diagnostics(self) -> Dict[str, float]:
        return {}

    def describe(self) -> str:
        return type(self).__name__
