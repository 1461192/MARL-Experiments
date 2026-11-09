from __future__ import annotations

from ..registry import register_agent
from .independent import _PerAgentQ


@register_agent("vdn")
class VDNAgent(_PerAgentQ):
    """Value-Decomposition Network (Sunehag et al., 2017).

        Q_team(h, a) ~= sum_i Q_i(h_i, a_i)

    Training: one TD regression of the *summed* Q towards the team-reward target
        r + gamma * sum_i max_a' Q_i'(h_i', a')
    so the team error's gradient flows through the sum into every agent's network.
    Execution: each agent acts greedily on its own Q_i, which equals maximizing the sum, so
    agents deploy fully decentralized.

    Options (paper sec. 3 / table 1): ``share_weights`` (agent-invariant network) and
    ``role_info`` (one-hot agent id concatenated with every observation).
    """

    def _chosen_q(self, q, actions):                                  # -> (B,T,1)
        return q.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)

    def _max_q(self, q):
        return q.max(-1).values.sum(-1, keepdim=True)
