"""Mixing networks that turn per-agent utilities ``Q_i(tau_i, a_i)`` into a joint ``Q_tot``.

Every mixer has the same interface::

    mixer(q, actions, state) -> Q_tot
        q        (B, T, d, n_actions)  every agent's utilities for every action
        actions  (B, T, d)             the joint action to evaluate
        state    (B, T, state_dim)     global state (here: all agents' observations, concatenated)
        Q_tot    (B, T, 1)

and is built so that the per-agent greedy actions maximise ``Q_tot`` (the "IGM" property),
which is what lets the agents act decentralised after centralised training.

* ``QMixer``  - QMIX (Rashid et al., 2018), section 4.
* ``QPLEXMixer`` - QPLEX (Wang et al., 2021), section 4 / figure 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _chosen(q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """(B,T,d,nA), (B,T,d) -> (B,T,d)"""
    return q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class QMixer(nn.Module):
    """QMIX mixing network (paper sec. 4, fig. 2).

    The per-agent Q-values of the chosen actions go through a two-layer network,
    ``Q_tot = w2^T ELU(W1^T q + b1) + b2``, whose weights and biases are *produced by
    hypernetworks* that take the global state ``s``:

    * ``W1``, ``w2``: one linear layer each, followed by an absolute value, so the weights are
      non-negative and ``dQ_tot / dQ_i >= 0`` (the monotonicity constraint, eq. 5);
    * ``b1``: one linear layer (biases may be negative);
    * ``b2`` (the state value ``V(s)``): two layers with a ReLU in between.

    The mixing layer has ``embed_dim`` units (paper: 32) and uses ELU. Monotonicity means the
    greedy per-agent actions maximise ``Q_tot``, so the bootstrap ``max_a Q_tot`` is computed by
    mixing each agent's own max.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents, self.embed_dim = n_agents, embed_dim
        self.hyper_w1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = _mlp(state_dim, embed_dim, 1)

    def forward(self, q: torch.Tensor, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        qa = _chosen(q, actions).unsqueeze(-2)                               # (B,T,1,d)
        w1 = self.hyper_w1(state).abs().view(*state.shape[:-1], self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).unsqueeze(-2)                              # (B,T,1,E)
        hidden = F.elu(qa @ w1 + b1)                                         # (B,T,1,E)
        w2 = self.hyper_w2(state).abs().unsqueeze(-1)                        # (B,T,E,1)
        return (hidden @ w2).squeeze(-1) + self.hyper_b2(state)              # (B,T,1)


class QPLEXMixer(nn.Module):
    """QPLEX duplex dueling mixer (paper sec. 4, eqs. 6-10).

    1. Individual dueling: ``V_i = max_a Q_i(tau_i, a)``, ``A_i = Q_i - V_i`` (eq. 6).
    2. Transformation (eq. 7): condition on the joint history/state with a positive weight and
       a bias, ``Q_i(tau, a_i) = w_i(s) Q_i(tau_i, a_i) + b_i(s)`` and
       ``V_i(tau) = w_i(s) V_i(tau_i) + b_i(s)``, so ``A_i(tau, a_i) = w_i(s) A_i(tau_i, a_i)``;
       ``w_i = |hypernet(s)| + eps > 0``.
    3. Dueling mixing (eqs. 8-9): ``Q_tot = V_tot + A_tot``, ``V_tot = sum_i V_i(tau)``,
       ``A_tot = sum_i lambda_i(tau, a) A_i(tau, a_i)`` with ``lambda_i > 0``.
    4. ``lambda_i`` (eq. 10) from a multi-head attention module over state and *joint action*:
       ``lambda_i = sum_k  key_k(s) * agent_{k,i}(s) * action_{k,i}(s, a)`` with
       ``key_k = |.| + eps`` and sigmoid ``agent``/``action`` terms, so every head is positive.

    Since ``A_i <= 0`` with equality at each agent's greedy action and ``lambda_i > 0``,
    ``argmax_a Q_tot = (argmax_a1 Q_1, ..., argmax_ad Q_d)`` for *any* joint value function the
    network can express - the full IGM class, not just the monotonic one QMIX covers.

    ``detach_advantage=True`` (default) follows the authors' PyMARL code: the ``lambda``-term is
    written as ``sum_i (lambda_i - 1) A_i`` with ``A_i`` detached and added to
    ``sum_i Q_i(tau, a_i)`` (same value, different gradient). It learns the non-monotonic matrix
    game more reliably than the plain gradient (``False``), which tends to drive ``lambda`` at the
    optimal joint action towards 0 while the greedy action is still wrong.
    """

    def __init__(self, n_agents: int, n_actions: int, state_dim: int, embed_dim: int = 32,
                 n_heads: int = 4, attn_hidden: int = 64, attn_layers: int = 2,
                 detach_advantage: bool = True):
        super().__init__()
        self.n_agents, self.n_actions, self.n_heads = n_agents, n_actions, n_heads
        self.detach_advantage = detach_advantage
        # transformation (eq. 7)
        self.hyper_w = nn.Linear(state_dim, n_agents)
        self.hyper_b = _mlp(state_dim, embed_dim, n_agents)
        # lambda attention (eq. 10); attn_layers=1 makes each term a single linear layer
        def net(i, o):
            return _mlp(i, attn_hidden, o) if attn_layers == 2 else nn.Linear(i, o)
        self.action_in = state_dim + n_agents * n_actions
        self.keys = nn.ModuleList(net(state_dim, 1) for _ in range(n_heads))
        self.agents = nn.ModuleList(net(state_dim, n_agents) for _ in range(n_heads))
        self.actions = nn.ModuleList(net(self.action_in, n_agents) for _ in range(n_heads))

    def lambdas(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """lambda_i(tau, a) > 0, ``(B, T, d)``."""
        onehot = F.one_hot(actions, self.n_actions).float().flatten(-2)       # (B,T,d*nA)
        sa = torch.cat([state, onehot], dim=-1)
        lam = 0
        for key, ag, act in zip(self.keys, self.agents, self.actions):
            lam = lam + (key(state).abs() + 1e-10) * torch.sigmoid(ag(state)) * torch.sigmoid(act(sa))
        return lam

    def forward(self, q: torch.Tensor, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w = self.hyper_w(state).abs() + 1e-10                                # (B,T,d)
        b = self.hyper_b(state)
        q_i = w * _chosen(q, actions) + b                                    # Q_i(tau, a_i)
        v_i = w * q.max(-1).values + b                                       # V_i(tau)
        adv = q_i - v_i                                                      # A_i(tau, a_i) <= 0
        lam = self.lambdas(actions, state)
        if self.detach_advantage:
            return (q_i + (lam - 1) * adv.detach()).sum(-1, keepdim=True)
        return (v_i + lam * adv).sum(-1, keepdim=True)
