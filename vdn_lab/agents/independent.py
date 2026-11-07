from __future__ import annotations

import numpy as np
import torch

from ..registry import register_agent
from .networks import IndependentNets
from .recurrent_dqn import RecurrentDQNAgent


class _PerAgentQ(RecurrentDQNAgent):
    """Per-agent recurrent Q-networks acting on local observations only."""

    def __init__(self, n_agents, n_actions, obs_shape, share_weights: bool = False,
                 role_info: bool = False, **kwargs):
        self.share_weights, self.role_info = share_weights, role_info
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)

    def _build_net(self):
        return IndependentNets(self.n_agents, self.obs_dim, self.n_actions, self.cfg.hidden,
                               self.share_weights, self.role_info)

    def _greedy(self, q_step: torch.Tensor) -> np.ndarray:           # q_step: (d, nA)
        return q_step.argmax(-1).cpu().numpy()

    def describe(self) -> str:
        return f"{type(self).__name__}(share_weights={self.share_weights}, role_info={self.role_info})"


@register_agent("independent")
class IndependentAgent(_PerAgentQ):
    """Independent recurrent DQN learners, each trained on the *team* reward (paper's IL baseline).

    Agent i regresses Q_i(h_i, a_i) towards  r_team + gamma * max_a' Q_i(h_i', a').
    It cannot tell teammates' (unobserved) behaviour from environment noise -> "spurious
    rewards" and the "lazy agent" problem discussed in the paper.
    """

    def _chosen_q(self, q, actions):                                  # (B,T,d,nA),(B,T,d) -> (B,T,d)
        return q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def _max_q(self, q):
        return q.max(-1).values
