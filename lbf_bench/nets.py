"""Networks, following EPyMARL (the implementation of Papoudakis et al.).

* ``AgentNet``  - EPyMARL's ``RNNAgent``: ``fc1 -> ReLU -> GRU`` (``use_rnn``) or ``-> Linear -> ReLU``
  (the "FC" network type), ``-> fc2``. Parameters are shared between agents; the input is the agent's
  observation plus a one-hot agent id (``obs_agent_id=True, obs_last_action=False``).
* ``MLPCritic`` - EPyMARL's critics: ``fc1 -> ReLU -> fc2 -> ReLU -> fc3``.
* ``QMixer``    - QMIX's monotonic mixing network with 2-layer hypernetworks (embed 32, hypernet 64).
* ``LambdaNet`` - the ATD(lambda) density-ratio estimator: ``[state, one-hot joint action] -> MLP -> GRU ->
  linear``; ``sigmoid`` of the output is lambda (Deng et al., sec. 4.1, fig. 2).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_out: int, use_rnn: bool):
        super().__init__()
        self.hidden, self.use_rnn = hidden, use_rnn
        self.fc1 = nn.Linear(in_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True) if use_rnn else nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def init_hidden(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden)

    def step(self, x: torch.Tensor, h: torch.Tensor):
        """x ``(N, in)``, h ``(N, hidden)`` -> out ``(N, n_out)``, h."""
        z = F.relu(self.fc1(x))
        if self.use_rnn:
            h = self.rnn(z[:, None], h[None])[1][0]
        else:
            h = F.relu(self.rnn(z))
        return self.fc2(h), h

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """x ``(B, T, n, in)`` -> ``(B, T, n, n_out)``, hidden state from zero at t = 0."""
        B, T, n, _ = x.shape
        if not self.use_rnn:
            return self.fc2(F.relu(self.rnn(F.relu(self.fc1(x)))))
        z = F.relu(self.fc1(x)).permute(0, 2, 1, 3).reshape(B * n, T, self.hidden)
        hs = self.rnn(z)[0].reshape(B, n, T, self.hidden).permute(0, 2, 1, 3)        # (B, T, n, H)
        return self.fc2(hs)


class MLPCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_out: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMixer(nn.Module):
    """Rashid et al. 2018; pymarl/EPyMARL ``QMixer`` with ``hypernet_layers=2``."""

    def __init__(self, n_agents: int, state_dim: int, embed: int = 32, hyper_embed: int = 64):
        super().__init__()
        self.n, self.embed = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(),
                                      nn.Linear(hyper_embed, embed * n_agents))
        self.hyper_wf = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(), nn.Linear(hyper_embed, embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.V = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """qs ``(B, T, n)``, states ``(B, T, S)`` -> ``(B, T, 1)``."""
        B, T, n = qs.shape
        s = states.reshape(B * T, -1)
        q = qs.reshape(B * T, 1, n)
        w1 = torch.abs(self.hyper_w1(s)).view(-1, n, self.embed)
        b1 = self.hyper_b1(s).view(-1, 1, self.embed)
        hidden = F.elu(torch.bmm(q, w1) + b1)
        wf = torch.abs(self.hyper_wf(s)).view(-1, self.embed, 1)
        v = self.V(s).view(-1, 1, 1)
        return (torch.bmm(hidden, wf) + v).view(B, T, 1)


def _net(i: int, o: int, layers: int, hidden: int = 64) -> nn.Module:
    return nn.Linear(i, o) if layers == 1 else nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, o))


class QPLEXMixer(nn.Module):
    """QPLEX's duplex dueling mixer (Wang et al. 2021, sec. 4 eqs. 6-11, appendix B.2 / table 2).

    * transformation (eq. 7): ``Q_i(tau, a_i) = w_i(s) Q_i(tau_i, a_i) + b_i(s)``, ``V_i(tau) = w_i(s) max Q_i + b_i(s)``,
      ``w_i = |.| + 1e-10``;
    * importance weights (eq. 10): ``lambda_i = sum_k upsilon_k(s) phi_{k,i}(s) lambda_{k,i}(s, a)`` with
      ``upsilon = |.| + 1e-10`` and sigmoid ``phi``, ``lambda``;
    * with the stop-gradient of appendix B.2: ``Q_tot = sum_i Q_i(tau, a_i) + sum_i (lambda_i - 1) stopgrad(A_i(tau, a_i))``.

    Table 2, StarCraft II column: one layer in each of w, b, lambda, phi, upsilon and 4 attention heads."""

    def __init__(self, n_agents: int, n_actions: int, state_dim: int, n_heads: int = 4, layers: int = 1):
        super().__init__()
        self.n, self.A = n_agents, n_actions
        self.w = _net(state_dim, n_agents, layers)
        self.b = _net(state_dim, n_agents, layers)
        self.keys = nn.ModuleList(_net(state_dim, 1, layers) for _ in range(n_heads))
        self.agents = nn.ModuleList(_net(state_dim, n_agents, layers) for _ in range(n_heads))
        self.actions = nn.ModuleList(_net(state_dim + n_agents * n_actions, n_agents, layers) for _ in range(n_heads))

    def forward(self, q_all: torch.Tensor, acts: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q_all ``(B, T, n, A)``, acts ``(B, T, n)``, state ``(B, T, S)`` -> ``(B, T, 1)``."""
        w = self.w(state).abs() + 1e-10
        b = self.b(state)
        q_i = w * q_all.gather(-1, acts[..., None]).squeeze(-1) + b
        v_i = w * q_all.max(-1)[0] + b
        sa = torch.cat([state, F.one_hot(acts, self.A).float().flatten(-2)], -1)
        lam = 0.0
        for k, ag, ac in zip(self.keys, self.agents, self.actions):
            lam = lam + (k(state).abs() + 1e-10) * torch.sigmoid(ag(state)) * torch.sigmoid(ac(sa))
        return (q_i + (lam - 1) * (q_i - v_i).detach()).sum(-1, keepdim=True)


class LambdaNet(nn.Module):
    """omega_phi of ATD(lambda): logits per step of a ``[state, joint action]`` sequence."""

    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden: int = 64, recurrent: bool = True):
        super().__init__()
        self.n_actions, self.hidden, self.recurrent = n_actions, hidden, recurrent
        self.fc = nn.Linear(state_dim + n_agents * n_actions, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True) if recurrent else None
        self.head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """state ``(B, T, S)``, actions ``(B, T, n)`` (episodes from t = 0) -> logits ``(B, T)``."""
        onehot = F.one_hot(actions, self.n_actions).float().flatten(-2)
        z = F.relu(self.fc(torch.cat([state, onehot], -1)))
        if self.recurrent:
            z = self.gru(z)[0]
        return self.head(z).squeeze(-1)


def soft_update(target: nn.Module, src: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), src.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


def hard_update(target: nn.Module, src: nn.Module) -> None:
    target.load_state_dict(src.state_dict())


def agent_ids(B: int, T: int, n: int) -> torch.Tensor:
    return torch.eye(n).expand(B, T, n, n)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x and mask broadcastable; mean over entries where mask = 1."""
    m = mask.expand_as(x)
    return (x * m).sum() / m.sum().clamp_min(1.0)
