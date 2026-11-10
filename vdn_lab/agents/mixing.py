"""Value-factorisation learners with a state-conditioned mixing network: QMIX and QPLEX.

Both reuse the per-agent recurrent Q-networks of VDN (so the three algorithms differ *only*
in how the per-agent utilities are combined) and the shared recurrent-DQN learner (replay,
target network, lambda-returns, epsilon-greedy).

The mixers need a global state. The testbed's agent interface only passes the agents'
observations, so the state is all observations concatenated (``(d * obs_dim,)``). In the
matrix games every agent sees the full state, so this *is* the state; in the grid worlds it
is the union of the agents' views.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..registry import register_agent
from .independent import _PerAgentQ
from .mixers import QMixer, QPLEXMixer
from .networks import IndependentNets


class MixedNets(nn.Module):
    """Per-agent nets + a mixer in one module, so the optimiser, the target network copy and
    ``state_dict`` cover both. ``forward`` is the agents' forward (used for acting)."""

    def __init__(self, agents: IndependentNets, mixer: nn.Module):
        super().__init__()
        self.agents, self.mixer = agents, mixer

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return self.agents.initial_hidden(batch, device)

    def forward(self, obs: torch.Tensor, hid: torch.Tensor):
        return self.agents(obs, hid)


class MixingAgent(_PerAgentQ):
    """Base for learners that train ``Q_tot = mixer(Q_1..Q_d, a, s)`` on the team reward.

    Training regresses ``Q_tot(tau, a, s)`` towards ``r + gamma * Q_tot^-(tau', a*, s')``
    where ``a*`` are the per-agent greedy actions (from the online network if ``double_q``,
    as in PyMARL's QPLEX, else from the target network) and ``Q_tot^-`` uses the target
    agent nets *and* the target mixer. Execution is decentralised: argmax of each ``Q_i``.
    """

    def __init__(self, n_agents, n_actions, obs_shape, double_q: bool = False, **kwargs):
        self.double_q = double_q
        self.mixer_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in self.MIXER_ARGS}
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)

    MIXER_ARGS: Tuple[str, ...] = ()

    def _make_mixer(self, state_dim: int) -> nn.Module:
        raise NotImplementedError

    def _build_net(self):
        agents = IndependentNets(self.n_agents, self.obs_dim, self.n_actions, self.cfg.hidden,
                                 self.share_weights, self.role_info)
        return MixedNets(agents, self._make_mixer(self.n_agents * self.obs_dim))

    @staticmethod
    def _state(obs: torch.Tensor) -> torch.Tensor:
        """(B, W, d, F) -> (B, W, d*F): every agent's observation, concatenated."""
        return obs.flatten(-2)

    def _td_values(self, obs, actions):
        bs, L = self.cfg.burn_in, self.cfg.seq_len
        hid0 = self.net.initial_hidden(obs.shape[0], self.device)
        state = self._state(obs)
        q, _ = self.net(obs, hid0)                                            # (B,W,d,nA)
        chosen = self.net.mixer(q[:, bs:bs + L], actions[:, bs:], state[:, bs:bs + L])
        with torch.no_grad():
            tq, _ = self.target(obs, hid0)
            nxt = tq[:, bs + 1:]
            greedy = (q[:, bs + 1:] if self.double_q else nxt).argmax(-1)
            boot = self.target.mixer(nxt, greedy, state[:, bs + 1:])
        return chosen, boot

    def describe(self) -> str:
        return (f"{type(self).__name__}(share_weights={self.share_weights}, role_info={self.role_info}, "
                f"double_q={self.double_q}, mixer={self.mixer_kwargs})")


@register_agent("qmix")
class QMIXAgent(MixingAgent):
    """QMIX (Rashid et al., 2018): monotonic mixing, ``dQ_tot/dQ_i >= 0``, with hypernetwork
    weights conditioned on the state. Option: ``mixing_embed_dim`` (paper: 32)."""

    MIXER_ARGS = ("mixing_embed_dim",)

    def _make_mixer(self, state_dim):
        return QMixer(self.n_agents, state_dim, self.mixer_kwargs.get("mixing_embed_dim", 32))


@register_agent("qplex")
class QPLEXAgent(MixingAgent):
    """QPLEX (Wang et al., 2021): duplex dueling mixing, covering every IGM-consistent joint
    value. Options: ``mixing_embed_dim``, ``n_heads`` (attention heads for lambda, default 4),
    ``attn_hidden``, ``attn_layers`` (1 or 2), ``detach_advantage`` (PyMARL's stop-gradient, default on)."""

    MIXER_ARGS = ("mixing_embed_dim", "n_heads", "attn_hidden", "attn_layers", "detach_advantage")

    def _make_mixer(self, state_dim):
        kw = dict(self.mixer_kwargs)
        if "mixing_embed_dim" in kw:
            kw["embed_dim"] = kw.pop("mixing_embed_dim")
        return QPLEXMixer(self.n_agents, self.n_actions, state_dim, **kw)
