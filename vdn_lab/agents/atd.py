"""Adaptive TD(lambda), ATD(lambda), for cooperative MARL.

Deng, Wang & Zhang, *Adaptive TD-Lambda for Cooperative Multi-agent Reinforcement Learning*, IJCAI 2026
(arXiv:2605.11880). Section numbers below refer to that paper.

Idea (sec. 1, 4.2)
------------------
A fixed lambda makes the same bias / variance trade-off for every sample. ATD(lambda) gives each
state-action pair its own lambda, by how likely the pair is under the *current* policy: a long
(Monte-Carlo-like) trace through transitions the current policy would generate, a short
(bootstrapped) one through transitions of older policies, whose returns are biased.

The on-policy degree is a density ratio ``omega(s, a) = d^pi(s, a) / d^D(s, a)`` estimated without
likelihoods from **two replay buffers** (sec. 4.2, 4.3): the regular large replay buffer
(``D_off``, older trajectories) and a small buffer of the newest trajectories (``D_on``). By the
variational f-divergence bound (Nguyen et al. 2010, Theorem 1) with the Jensen-Shannon f, the
estimator ``omega_phi`` is trained with a binary cross-entropy (eq. 8, appendix A.4)::

    L(phi) = BCE_{(s,a) ~ D_off}(omega_phi(s, a), 0) + BCE_{(s,a) ~ D_on}(omega_phi(s, a), 1)

and its output, squashed into (0, 1) by a sigmoid, **is the lambda value** (sec. 4.1). The target
(eq. 10) is the lambda-return with that lambda::

    R_t = r(s, a) + gamma * [ omega(s, a) R_{t+1} + (1 - omega(s, a)) max_a' Q_tot(s', a' | theta^-) ]

Details from the paper (sec. 4.1, 4.3, appendix F/G, algorithm 1):

* **Network.** Like the agents' utility network: MLP -> GRU -> linear -> sigmoid, over a
  trajectory of [state, joint action] (fig. 2). Recurrence helped on SMAC's super-hard maps
  (fig. 12). ``atd_recurrent=False`` gives the MLP-only version.
* **Buffers.** ``D_on`` holds the newest trajectories, a 50th of ``D_off`` (fig. 6: 50x was best of
  10x / 25x / 50x / 100x). Here ``D_on`` is the newest ``buffer_size / 50`` entries of the replay.
* **Update schedule.** The estimator is updated with the same frequency as the target network,
  and the lambda of each transition is computed once, cached in the replay buffer, and the cache
  is cleared at every target-network update (sec. 4.3, appendix F "lambda value cache"), so the
  targets stay fixed between target updates.
* **Optimiser.** Adam, lr 1e-3 (table 3, ``lfiw_optim``).
* **MAPPO (sec. 4.3).** ``D_on`` is the rollout buffer; ``D_off`` keeps the history, 50x larger. The
  critic regresses onto ATD lambda-returns instead of the GAE returns; the actor update (GAE
  advantages) is unchanged.

Differences from the paper, all forced by this testbed or left open by it:

* The "state" is all agents' observations concatenated (the agent interface has no global state),
  so the GRU is not redundant here.
* The DQN learners train on fixed windows (``burn_in + seq_len + 1``) sampled from the replay; the
  estimator is trained on windows of the same length from each buffer, its GRU starting from zero
  at the window start, like the Q-networks.
* The paper does not say how many estimator steps the MAPPO variant takes per update; here
  ``atd_steps`` (default 10), each on ``atd_batch`` chunks of ``chunk_len`` steps from each buffer.
* QMIX+ATD in the paper uses lr 1e-3 against 5e-4 for QMIX and pymarl's td_lambda 0.4 for the
  baselines (table 3). The comparisons here keep the base learner's hyper-parameters unchanged.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_agent
from .mappo import MAPPOAgent
from .mixing import QMIXAgent, QPLEXAgent
from .vdn import VDNAgent


@dataclass
class ATDConfig:
    atd_hidden: int = 64              # estimator width (paper: as the utility network, rnn_hidden_dim 64)
    atd_recurrent: bool = True        # MLP -> GRU -> head (paper); False: MLP -> head
    atd_lr: float = 1e-3              # Adam (paper: lfiw_optim Adam, lfiw_optim_lr 0.001)
    atd_ratio: int = 50               # |D_off| / |D_on| (paper: 50x)
    atd_steps: Optional[int] = None   # estimator steps per update event; None -> 1 (DQN) / 10 (MAPPO)
    atd_batch: Optional[int] = None   # sequences from *each* buffer per step; None -> learner's batch size / 32
    atd_cache: bool = True            # DQN: cache lambda per transition, cleared at each target update

    @classmethod
    def pop_from(cls, kwargs: Dict[str, Any]) -> "ATDConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: kwargs.pop(k) for k in list(kwargs) if k in names})


class ATDNetwork(nn.Module):
    """``omega_phi``: [state, one-hot joint action] sequences -> lambda in (0, 1) per step (logits)."""

    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden: int, recurrent: bool):
        super().__init__()
        self.n_actions, self.hidden, self.recurrent = n_actions, hidden, recurrent
        self.fc = nn.Linear(state_dim + n_agents * n_actions, hidden)
        self.gru = nn.GRUCell(hidden, hidden) if recurrent else None
        self.head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor,
                starts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """state ``(B, T, S)``, actions ``(B, T, d)``, starts ``(B, T)`` (1 where an episode begins,
        the GRU state is reset there) -> logits ``(B, T)``."""
        onehot = F.one_hot(actions, self.n_actions).float().flatten(-2)
        z = F.relu(self.fc(torch.cat([state, onehot], -1)))
        if self.recurrent:
            h = torch.zeros(z.shape[0], self.hidden, device=z.device)
            outs = []
            for t in range(z.shape[1]):
                if starts is not None and t > 0:
                    h = h * (1.0 - starts[:, t]).unsqueeze(-1)
                h = self.gru(z[:, t], h)
                outs.append(h)
            z = torch.stack(outs, 1)
        return self.head(z).squeeze(-1)


class AdaptiveLambda:
    """The estimator, its optimiser and the BCE training step of eq. 8."""

    def __init__(self, cfg: ATDConfig, state_dim: int, n_agents: int, n_actions: int, device: str):
        self.cfg = cfg
        self.net = ATDNetwork(state_dim, n_agents, n_actions, cfg.atd_hidden, cfg.atd_recurrent).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.atd_lr)
        self.n_steps = 0
        self.last: Dict[str, float] = {}

    def train_step(self, on: Dict[str, torch.Tensor], off: Dict[str, torch.Tensor]) -> float:
        """``on`` / ``off``: dicts with ``state (B,T,S)``, ``actions (B,T,d)`` and optional ``starts (B,T)``."""
        l_on = self.net(on["state"], on["actions"], on.get("starts"))
        l_off = self.net(off["state"], off["actions"], off.get("starts"))
        loss = (F.binary_cross_entropy_with_logits(l_on, torch.ones_like(l_on))
                + F.binary_cross_entropy_with_logits(l_off, torch.zeros_like(l_off)))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.n_steps += 1
        with torch.no_grad():
            acc = 0.5 * ((l_on > 0).float().mean() + (l_off <= 0).float().mean())
        self.last.update(atd_loss=float(loss.item()), atd_acc=float(acc.item()))
        return float(loss.item())

    @torch.no_grad()
    def lambdas(self, state, actions, starts=None) -> torch.Tensor:
        return torch.sigmoid(self.net(state, actions, starts))

    def record(self, lam: torch.Tensor) -> None:
        lam = lam.flatten().float()
        self.last.update(lam_mean=float(lam.mean()), lam_std=float(lam.std()) if lam.numel() > 1 else 0.0,
                         lam_lo_frac=float((lam < 0.1).float().mean()), lam_hi_frac=float((lam > 0.9).float().mean()))

    def state_dict(self) -> Dict[str, Any]:
        return dict(net=self.net.state_dict(), opt=self.opt.state_dict(), n_steps=self.n_steps)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.n_steps = state["n_steps"]


# ----------------------------------------------------------------- value-based (replay) learners
class ATDDQNMixin:
    """ATD(lambda) for the recurrent-DQN learners (VDN / QMIX / QPLEX), algorithm 1 of the paper.

    ``D_off`` = the replay buffer, ``D_on`` = its ``buffer_size / atd_ratio`` newest entries. At every
    target-network update the estimator takes ``atd_steps`` steps on ``atd_batch`` windows from each
    buffer and the lambda cache is cleared. Lambdas of the sampled windows come from the cache where
    present, else from the estimator (and are then cached)."""

    def __init__(self, n_agents, n_actions, obs_shape, **kwargs):
        self.atd_cfg = ATDConfig.pop_from(kwargs)
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
        c = self.atd_cfg
        c.atd_steps = c.atd_steps or 1
        c.atd_batch = c.atd_batch or self.cfg.batch_size
        self.atd_small = max(self.cfg.burn_in + self.cfg.seq_len + 1, self.cfg.buffer_size // c.atd_ratio)
        self.atd = AdaptiveLambda(c, n_agents * self.obs_dim, n_agents, n_actions, self.device)
        self.lam_cache = np.full(self.cfg.buffer_size, np.nan, dtype=np.float32)

    def observe(self, obs, actions, reward, next_obs, done) -> None:
        start = self.replay.ptr
        super().observe(obs, actions, reward, next_obs, done)
        n = (self.replay.ptr - start) % self.replay.cap                    # entries just (over)written
        self.lam_cache[(start + np.arange(n)) % self.replay.cap] = np.nan

    def _windows(self, recent: Optional[int]) -> Dict[str, torch.Tensor]:
        W = self.cfg.burn_in + self.cfg.seq_len + 1
        b = self.replay.sample(self.atd_cfg.atd_batch, W, recent=recent)
        return dict(state=self._to_input(b["obs"][:, :-1]).flatten(-2),
                    actions=torch.from_numpy(b["actions"]).to(self.device))

    def _train_estimator(self) -> None:
        if len(self.replay) < self.atd_small:
            return
        for _ in range(self.atd_cfg.atd_steps):
            self.atd.train_step(self._windows(self.atd_small), self._windows(None))

    def _on_target_update(self) -> None:
        super()._on_target_update()
        self._train_estimator()
        self.lam_cache[:] = np.nan

    def _trace_lambda(self, obs, actions, idx):
        """(B, L, 1): lam[:, t] = omega(s_t, a_t) for the L learning steps of the window."""
        bs, L = self.cfg.burn_in, self.cfg.seq_len
        steps = idx[:, bs:bs + L]                                               # replay positions
        cached = self.lam_cache[steps] if self.atd_cfg.atd_cache else np.full(steps.shape, np.nan, np.float32)
        miss = np.isnan(cached)
        if miss.any():
            lam = self.atd.lambdas(obs[:, :bs + L].flatten(-2), actions[:, :bs + L])[:, bs:].cpu().numpy()
            cached = np.where(miss, lam, cached)
            if self.atd_cfg.atd_cache:
                self.lam_cache[steps[miss]] = lam[miss]
        lam_t = torch.from_numpy(cached).to(obs.device).unsqueeze(-1)
        self.atd.record(lam_t[:, :-1])                                          # the last one is never used
        return lam_t

    def _learn(self) -> Dict[str, float]:
        out = super()._learn()
        out.update(self.atd.last)
        return out

    def state_dict(self):
        return {**super().state_dict(), "atd": self.atd.state_dict(), "atd_cfg": asdict(self.atd_cfg)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "atd" in state:
            self.atd.load_state_dict(state["atd"])
        self.lam_cache[:] = np.nan

    def describe(self) -> str:
        return f"{super().describe()} + ATD(lambda)"


@register_agent("vdn_atd")
class VDNATDAgent(ATDDQNMixin, VDNAgent):
    """VDN with adaptive TD(lambda)."""


@register_agent("qmix_atd")
class QMIXATDAgent(ATDDQNMixin, QMIXAgent):
    """QMIX with adaptive TD(lambda) (the paper's main value-based instance)."""


@register_agent("qplex_atd")
class QPLEXATDAgent(ATDDQNMixin, QPLEXAgent):
    """QPLEX with adaptive TD(lambda) (paper sec. 5.3, fig. 5)."""


# ------------------------------------------------------------------------------------- MAPPO
@register_agent("mappo_atd")
class MAPPOATDAgent(MAPPOAgent):
    """MAPPO whose critic regresses onto ATD lambda-returns (paper sec. 4.3).

    ``D_on`` = the current rollout; ``D_off`` = the last ``atd_ratio`` rollouts (this one included).
    Before each PPO update the estimator takes ``atd_steps`` steps on ``atd_batch`` chunks of
    ``chunk_len`` steps from each buffer. The critic's targets are ``A^lambda_t + V_t`` with
    ``lambda_t = omega(s_t, a_t)``; the actor's advantages keep ``gae_lambda``."""

    def __init__(self, n_agents, n_actions, obs_shape, **kwargs):
        self.atd_cfg = ATDConfig.pop_from(kwargs)
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
        c = self.atd_cfg
        c.atd_steps = c.atd_steps or 10
        c.atd_batch = c.atd_batch or 32
        self.atd = AdaptiveLambda(c, n_agents * self.obs_dim, n_agents, n_actions, self.device)
        self._history: deque = deque(maxlen=max(1, c.atd_ratio))     # (obs, actions, starts) per rollout
        self._lam: Optional[np.ndarray] = None

    def _chunks(self, rollouts, n: int) -> Dict[str, torch.Tensor]:
        L, T = self.cfg.chunk_len, self.cfg.rollout_len
        r = self.rng.integers(0, len(rollouts), n)
        s = self.rng.integers(0, T // L, n) * L
        obs = np.stack([rollouts[i][0][j:j + L] for i, j in zip(r, s)])
        act = np.stack([rollouts[i][1][j:j + L] for i, j in zip(r, s)])
        st = np.stack([rollouts[i][2][j:j + L] for i, j in zip(r, s)])
        return dict(state=self._to_input(obs).flatten(-2), actions=torch.from_numpy(act).to(self.device),
                    starts=torch.from_numpy(st).to(self.device))

    def _learn(self) -> Dict[str, float]:
        b, c = self.buf, self.atd_cfg
        current = (b["obs"].copy(), b["actions"].copy(), b["starts"].copy())
        self._history.append(current)
        for _ in range(c.atd_steps):
            self.atd.train_step(self._chunks([current], c.atd_batch), self._chunks(self._history, c.atd_batch))
        # lambda over the whole rollout as one sequence (GRU reset at episode starts)
        lam = self.atd.lambdas(self._to_input(b["obs"]).flatten(-2)[None],
                               torch.from_numpy(b["actions"]).to(self.device)[None],
                               torch.from_numpy(b["starts"]).to(self.device)[None])[0]
        self.atd.record(lam)
        self._lam = lam.cpu().numpy().astype(np.float32)
        out = super()._learn()
        out.update(self.atd.last)
        return out

    def _critic_lambda(self):
        return self._lam

    def state_dict(self):
        return {**super().state_dict(), "atd": self.atd.state_dict(), "atd_cfg": asdict(self.atd_cfg)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "atd" in state:
            self.atd.load_state_dict(state["atd"])

    def describe(self) -> str:
        return f"{super().describe()} + ATD(lambda)"
