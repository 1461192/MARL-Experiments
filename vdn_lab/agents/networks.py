"""Network architectures from section 4.1 of the paper.

Every agent network is:   Linear(32) -> ReLU -> LSTM(32) -> ReLU -> dueling layer(32)
(the dueling layer yields V(s) and A(s,a); Q = V + A - mean(A), Wang et al. 2016).

* ``IndependentNets`` - one observation (3x5x5) feeds one such net per agent (or one shared
  net, optionally with a one-hot *role id* concatenated to the observation). Used by the
  independent learners and by value-decomposition networks.
* ``CentralizedNet`` - the ``d`` observations pass through ``d`` separate linear layers,
  are concatenated (d*32 = 64 for two players), ReLU, then a single LSTM. The head is one
  dueling layer over the *joint* action space (n_actions**d), or ``d`` per-agent dueling
  layers (centralised + value decomposition).

Recurrent state is a tensor ``(B, units, 2, hidden)`` holding (h, c).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingHead(nn.Module):
    def __init__(self, in_dim: int, n_out: int):
        super().__init__()
        self.value = nn.Linear(in_dim, 1)
        self.adv = nn.Linear(in_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.adv(x)
        return self.value(x) + a - a.mean(dim=-1, keepdim=True)


def _split(hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, 2, H) -> h0, c0 each (1, N, H) as nn.LSTM expects."""
    return hidden[:, 0].unsqueeze(0).contiguous(), hidden[:, 1].unsqueeze(0).contiguous()


def _merge(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.stack([h[0], c[0]], dim=1)


class AgentQNet(nn.Module):
    """Linear -> ReLU -> LSTM -> ReLU -> dueling. Input ``(N, T, in_dim)``."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.fc = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = DuelingHead(hidden, n_actions)

    def forward(self, x: torch.Tensor, hid: torch.Tensor):
        z = F.relu(self.fc(x))
        out, (h, c) = self.lstm(z, _split(hid))
        return self.head(F.relu(out)), _merge(h, c)


class IndependentNets(nn.Module):
    """Per-agent recurrent Q-networks over *local* observations.
    forward(obs (B,T,d,F), hidden (B,d,2,H)) -> q (B,T,d,nA), hidden"""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, hidden: int = 32,
                 share_weights: bool = False, role_info: bool = False):
        super().__init__()
        self.n_agents, self.hidden = n_agents, hidden
        self.share_weights, self.role_info = share_weights, role_info
        in_dim = obs_dim + (n_agents if role_info else 0)
        n_nets = 1 if share_weights else n_agents
        self.nets = nn.ModuleList(AgentQNet(in_dim, n_actions, hidden) for _ in range(n_nets))

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.n_agents, 2, self.hidden, device=device)

    def forward(self, obs: torch.Tensor, hid: torch.Tensor):
        B, T, d, _ = obs.shape
        if self.role_info:
            roles = torch.eye(d, device=obs.device).expand(B, T, d, d)
            obs = torch.cat([obs, roles], dim=-1)
        if self.share_weights:                       # agents become extra batch items
            x = obs.permute(0, 2, 1, 3).reshape(B * d, T, -1)
            q, h = self.nets[0](x, hid.reshape(B * d, 2, self.hidden))
            return q.reshape(B, d, T, -1).permute(0, 2, 1, 3), h.reshape(B, d, 2, self.hidden)
        qs, hs = [], []
        for i, net in enumerate(self.nets):
            q, h = net(obs[:, :, i], hid[:, i])
            qs.append(q)
            hs.append(h)
        return torch.stack(qs, dim=2), torch.stack(hs, dim=1)


class CentralizedNet(nn.Module):
    """Sees every agent's observation. forward(obs (B,T,d,F), hidden (B,1,2,H)) ->
    q (B,T,nA**d) if ``joint`` else q (B,T,d,nA)."""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, hidden: int = 32, joint: bool = True):
        super().__init__()
        self.n_agents, self.hidden, self.joint = n_agents, hidden, joint
        self.encoders = nn.ModuleList(nn.Linear(obs_dim, hidden) for _ in range(n_agents))
        self.lstm = nn.LSTM(hidden * n_agents, hidden, batch_first=True)
        if joint:
            self.head = DuelingHead(hidden, n_actions ** n_agents)
        else:
            self.heads = nn.ModuleList(DuelingHead(hidden, n_actions) for _ in range(n_agents))

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, 1, 2, self.hidden, device=device)

    def forward(self, obs: torch.Tensor, hid: torch.Tensor):
        z = torch.cat([enc(obs[:, :, i]) for i, enc in enumerate(self.encoders)], dim=-1)
        out, (h, c) = self.lstm(F.relu(z), _split(hid[:, 0]))
        out = F.relu(out)
        new_hid = _merge(h, c).unsqueeze(1)
        if self.joint:
            return self.head(out), new_hid
        return torch.stack([hd(out) for hd in self.heads], dim=2), new_hid
