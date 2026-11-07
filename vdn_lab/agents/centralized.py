from __future__ import annotations

import numpy as np
import torch

from ..registry import register_agent
from .networks import CentralizedNet
from .recurrent_dqn import RecurrentDQNAgent


@register_agent("centralized")
class CentralizedAgent(RecurrentDQNAgent):
    """Fully centralised learner that sees every agent's observation.

    ``decompose=False`` (default): the paper's *combinatorial* agent - one Q over the joint
    action space (n_actions ** n_agents outputs), epsilon-greedy over joint actions.
    ``decompose=True``: centralised trunk, but value decomposition on top (sum of per-agent
    heads); see ``CentralizedVDNAgent``.
    """

    def __init__(self, n_agents, n_actions, obs_shape, decompose: bool = False, **kwargs):
        self.decompose = decompose
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
        self._powers = torch.tensor([n_actions ** (n_agents - 1 - i) for i in range(n_agents)],
                                    device=self.device)

    def _build_net(self):
        return CentralizedNet(self.n_agents, self.obs_dim, self.n_actions, self.cfg.hidden,
                              joint=not self.decompose)

    def _encode(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions * self._powers).sum(-1)

    def _decode(self, idx: int) -> np.ndarray:
        return np.array([(idx // int(p)) % self.n_actions for p in self._powers.tolist()])

    def _chosen_q(self, q, actions):
        if self.decompose:
            return q.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
        return q.gather(-1, self._encode(actions).unsqueeze(-1))        # (B,T,1)

    def _max_q(self, q):
        if self.decompose:
            return q.max(-1).values.sum(-1, keepdim=True)
        return q.max(-1, keepdim=True).values

    def _greedy(self, q_step: torch.Tensor) -> np.ndarray:
        if self.decompose:
            return q_step.argmax(-1).cpu().numpy()
        return self._decode(int(q_step.argmax().item()))

    def _epsilon_greedy(self, greedy: np.ndarray, eps: float) -> np.ndarray:
        if self.decompose:
            return super()._epsilon_greedy(greedy, eps)
        if self.rng.random() < eps:                                    # explore in the joint space
            return self.rng.integers(self.n_actions, size=self.n_agents)
        return greedy

    def describe(self) -> str:
        return f"CentralizedAgent(decompose={self.decompose})"


@register_agent("centralized_vdn")
class CentralizedVDNAgent(CentralizedAgent):
    """Centralised trunk + value decomposition (the paper's 'centralised with VD' agent)."""

    def __init__(self, n_agents, n_actions, obs_shape, **kwargs):
        kwargs.setdefault("decompose", True)
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
