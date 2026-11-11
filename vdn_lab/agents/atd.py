"""Adaptive TD(lambda), ATD(lambda), for cooperative MARL.

Deng, Wang & Zhang, *Adaptive TD-Lambda for Cooperative Multi-agent Reinforcement Learning*,
IJCAI 2026 (arXiv:2605.11880). The paper applies it to QMIX (value-based) and MAPPO (actor-critic);
it is a plug-in for any learner that builds TD(lambda) / GAE targets, so here it is also wired into
QPLEX (and VDN).

Idea
----
A fixed lambda trades bias (small lambda: lean on the bootstrapped value) against variance and
off-policy error (large lambda: lean on the sampled rewards) in the same way for every sample.
ATD(lambda) instead picks lambda **per state-action pair**, according to how likely the pair is
under the stationary distribution of the *current* policy, ``d^pi(s, a)``, relative to the
distribution of the stored data, ``d^D(s, a)``. That ratio ``w(s, a) = d^pi / d^D`` is not
computable in a multi-agent game, so it is estimated without likelihoods from two replay buffers:

* a **large** buffer of older (off-policy) trajectories, a sample of ``d^D``;
* a **small** buffer of the newest (near on-policy) trajectories, a sample of ``d^pi``.

A discriminator ``f(s, a)`` (``DensityRatioEstimator``) is trained to tell transitions of the two
buffers apart (logistic loss, equal numbers from each). At the optimum its logit is
``log d^pi(s, a) / d^D(s, a)``, so ``w = exp(logit)``. The ratio is then **scaled into lambda**:
pairs the current policy visits more often than the data suggests get a larger lambda, pairs that
are mostly old data get a smaller one.

What is from the paper and what is not
--------------------------------------
The paper text could not be fetched from the environment this was written in (arxiv.org and
ijcai.org were blocked), so only the parts stated in its abstract and in summaries are taken
from it: per-state-action lambda; a parametric likelihood-free density-ratio estimator trained
adversarially on samples of a large and a small replay buffer; the ratio scaled to give lambda;
used with QMIX and MAPPO; buffer-size ratios of 10x and more studied. The rest follows the
likelihood-free importance-weight estimator that this construction comes from (Sinha et al.,
*Experience Replay with Likelihood-free Importance Weights*, L4DC 2022): a classifier between a
fast and a slow buffer, and batch self-normalisation of ``w`` with a temperature. The scaling
into lambda (below), the estimator architecture and every number are this implementation's
choices. Check them against the paper before relying on exact numbers.

Scaling (``ATDConfig``)
-----------------------
Within each batch of sampled state-action pairs::

    w~(s, a)    = w(s, a)^(1/T) / mean_batch(w^(1/T))          (self-normalised, mean 1)
    lambda(s,a) = clip(lambda_base * w~(s, a), lam_min, lam_max)

``lambda_base`` is the base algorithm's own lambda (0.9 for the DQN learners, 0.95 for MAPPO's
GAE), so the average lambda stays at the base value (up to clipping) and ATD(lambda) only
redistributes it between pairs. With the estimator untrained, or while both buffers hold the same
data, ``w = 1`` and the learner is exactly the base algorithm.

The joint input is the "state" used by the mixers (all agents' observations concatenated) plus
the one-hot joint action.
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
    atd_hidden: int = 64          # estimator MLP width (two hidden layers)
    atd_lr: float = 5e-4          # estimator Adam learning rate
    atd_batch: int = 128          # transitions from *each* buffer per estimator step
    atd_small: Optional[int] = None  # size of the small (on-policy) buffer in env steps; None -> learner default
    atd_large: Optional[int] = None  # size of the large buffer (MAPPO only; the DQN learners use their replay)
    atd_steps: int = 1            # estimator steps per learner update (DQN) / per rollout (MAPPO)
    atd_warmup: int = 200         # estimator steps before its lambdas are used (base lambda until then)
    atd_temperature: float = 2.0  # T in w^(1/T): >1 flattens the ratios
    lam_min: float = 0.5
    lam_max: float = 0.99
    atd_logit_clip: float = 5.0   # |log w| is clipped to this before scaling

    @classmethod
    def pop_from(cls, kwargs: Dict[str, Any], **defaults: Any) -> "ATDConfig":
        names = {f.name for f in fields(cls)}
        own = {k: kwargs.pop(k) for k in list(kwargs) if k in names}
        return cls(**{**defaults, **own})


class DensityRatioEstimator(nn.Module):
    """``f(s, a) -> log w(s, a)``: MLP over [state, one-hot joint action]."""

    def __init__(self, state_dim: int, n_agents: int, n_actions: int, hidden: int):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(nn.Linear(state_dim + n_agents * n_actions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """state ``(..., state_dim)``, actions ``(..., d)`` -> logit ``(...)``."""
        onehot = F.one_hot(actions, self.n_actions).float().flatten(-2)
        return self.net(torch.cat([state, onehot], -1)).squeeze(-1)


class AdaptiveLambda:
    """Density-ratio estimator + its optimiser + the ratio -> lambda scaling."""

    def __init__(self, cfg: ATDConfig, lam_base: float, state_dim: int, n_agents: int, n_actions: int,
                 device: str):
        self.cfg, self.lam_base = cfg, lam_base
        self.est = DensityRatioEstimator(state_dim, n_agents, n_actions, cfg.atd_hidden).to(device)
        self.opt = torch.optim.Adam(self.est.parameters(), lr=cfg.atd_lr)
        self.n_steps = 0
        self.last: Dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return self.n_steps >= self.cfg.atd_warmup

    def train_step(self, s_small, a_small, s_large, a_large) -> float:
        """One discriminator step: small buffer (label 1) vs large buffer (label 0)."""
        logits = torch.cat([self.est(s_small, a_small), self.est(s_large, a_large)])
        labels = torch.cat([torch.ones(len(s_small)), torch.zeros(len(s_large))]).to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.n_steps += 1
        with torch.no_grad():
            acc = ((logits > 0).float() == labels).float().mean()
        self.last = {"atd_loss": float(loss.item()), "atd_acc": float(acc.item())}
        return float(loss.item())

    @torch.no_grad()
    def lambdas(self, state: torch.Tensor, actions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """lambda for every pair, shape of ``state.shape[:-1]``. ``mask`` (same shape, bool) selects
        the pairs that enter the batch normalisation (default: all)."""
        c = self.cfg
        if not self.ready:
            return torch.full(state.shape[:-1], self.lam_base, device=state.device)
        log_w = self.est(state, actions).clamp(-c.atd_logit_clip, c.atd_logit_clip) / c.atd_temperature
        sel = log_w.flatten() if mask is None else log_w[mask]
        w = torch.exp(log_w - torch.logsumexp(sel, 0) + np.log(sel.numel()))     # / mean(w^(1/T))
        lam = (self.lam_base * w).clamp(c.lam_min, c.lam_max)
        lam_sel = lam.flatten() if mask is None else lam[mask]
        self.last.update(lam_mean=float(lam_sel.mean()), lam_std=float(lam_sel.std()) if lam_sel.numel() > 1 else 0.0,
                         lam_lo_frac=float((lam_sel <= c.lam_min + 1e-6).float().mean()),
                         lam_hi_frac=float((lam_sel >= c.lam_max - 1e-6).float().mean()))
        return lam

    def state_dict(self) -> Dict[str, Any]:
        return dict(est=self.est.state_dict(), opt=self.opt.state_dict(), n_steps=self.n_steps)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.est.load_state_dict(state["est"])
        self.opt.load_state_dict(state["opt"])
        self.n_steps = state["n_steps"]


# ----------------------------------------------------------------- value-based (replay) learners
class ATDDQNMixin:
    """ATD(lambda) for the recurrent-DQN learners (VDN / QMIX / QPLEX).

    Large buffer = the whole replay buffer; small buffer = its ``atd_small`` newest entries
    (default: a tenth of the replay, the smallest size ratio studied in the paper). Every learner
    update first takes ``atd_steps`` estimator steps, then uses per-pair lambdas in the
    lambda-returns of the sampled trajectories."""

    def __init__(self, n_agents, n_actions, obs_shape, **kwargs):
        self.atd_cfg = ATDConfig.pop_from(kwargs)
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
        if self.atd_cfg.atd_small is None:
            self.atd_cfg.atd_small = max(1, self.cfg.buffer_size // 10)
        self.atd = AdaptiveLambda(self.atd_cfg, self.cfg.lam, n_agents * self.obs_dim, n_agents, n_actions,
                                  self.device)

    def _learn(self) -> Dict[str, float]:
        c, R = self.atd_cfg, self.replay
        for _ in range(c.atd_steps):
            small, large = R.sample_steps(c.atd_batch, recent=c.atd_small), R.sample_steps(c.atd_batch)
            self.atd.train_step(self._to_input(small["obs"]).flatten(-2),
                                torch.from_numpy(small["actions"]).to(self.device),
                                self._to_input(large["obs"]).flatten(-2),
                                torch.from_numpy(large["actions"]).to(self.device))
        out = super()._learn()
        out.update(self.atd.last)
        return out

    def _trace_lambda(self, obs, actions):
        """(B, L, 1): lam[:, t] = lambda(s_{t+1}, a_{t+1}); the last entry is never read."""
        bs, L = self.cfg.burn_in, self.cfg.seq_len
        lam = torch.full((obs.shape[0], L, 1), self.cfg.lam, device=obs.device)
        if L > 1:
            state = obs[:, bs + 1:bs + L].flatten(-2)                           # (B, L-1, d*F)
            lam[:, :-1, 0] = self.atd.lambdas(state, actions[:, bs + 1:bs + L])
        return lam

    def state_dict(self):
        return {**super().state_dict(), "atd": self.atd.state_dict(), "atd_cfg": asdict(self.atd_cfg)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "atd" in state:
            self.atd.load_state_dict(state["atd"])

    def describe(self) -> str:
        return f"{super().describe()} + ATD(lambda)"


@register_agent("vdn_atd")
class VDNATDAgent(ATDDQNMixin, VDNAgent):
    """VDN with adaptive TD(lambda)."""


@register_agent("qmix_atd")
class QMIXATDAgent(ATDDQNMixin, QMIXAgent):
    """QMIX with adaptive TD(lambda) (the paper's value-based instance)."""


@register_agent("qplex_atd")
class QPLEXATDAgent(ATDDQNMixin, QPLEXAgent):
    """QPLEX with adaptive TD(lambda) (plug-in; QPLEX is a baseline in the paper)."""


# ------------------------------------------------------------------------------------- MAPPO
@register_agent("mappo_atd")
class MAPPOATDAgent(MAPPOAgent):
    """MAPPO with adaptive GAE(lambda).

    Small buffer = the current rollout (``rollout_len`` steps, the on-policy data). Large buffer =
    a FIFO of the last ``atd_large`` steps of earlier rollouts plus this one (default 10 rollouts).
    Before each PPO update the estimator takes ``atd_steps`` steps, then GAE uses one lambda per
    time step, ``lambda(s_{t+1}, a_{t+1})`` (shared by the agents, like the team reward)."""

    def __init__(self, n_agents, n_actions, obs_shape, **kwargs):
        rollout = kwargs.get("rollout_len", 1000)
        self.atd_cfg = ATDConfig.pop_from(kwargs, atd_steps=25, atd_warmup=50, atd_batch=256,
                                          atd_large=10 * rollout)
        super().__init__(n_agents, n_actions, obs_shape, **kwargs)
        if self.atd_cfg.atd_small is None:
            self.atd_cfg.atd_small = self.cfg.rollout_len
        self.atd = AdaptiveLambda(self.atd_cfg, self.cfg.gae_lambda, n_agents * self.obs_dim, n_agents, n_actions,
                                  self.device)
        cap = max(1, self.atd_cfg.atd_large // self.cfg.rollout_len)
        self._history: deque = deque(maxlen=cap)             # (obs, actions) of whole rollouts
        self._lam: Optional[np.ndarray] = None

    def _learn(self) -> Dict[str, float]:
        c, b, dev = self.atd_cfg, self.buf, self.device
        T = self.cfg.rollout_len
        self._history.append((b["obs"].copy(), b["actions"].copy()))
        h_obs = np.concatenate([o for o, _ in self._history])
        h_act = np.concatenate([a for _, a in self._history])
        small = min(c.atd_small, T)
        s_small = self._to_input(b["obs"][T - small:]).flatten(-2)
        a_small = torch.from_numpy(b["actions"][T - small:]).to(dev)
        for _ in range(c.atd_steps):
            i = self.rng.integers(0, small, c.atd_batch)
            j = self.rng.integers(0, len(h_obs), c.atd_batch)
            self.atd.train_step(s_small[i], a_small[i],
                                self._to_input(h_obs[j]).flatten(-2), torch.from_numpy(h_act[j]).to(dev))
        # lambda at step t belongs to (s_{t+1}, a_{t+1}); the last entry is cut by GAE anyway
        state = self._to_input(b["obs"]).flatten(-2)
        lam_next = self.atd.lambdas(state[1:], torch.from_numpy(b["actions"][1:]).to(dev),
                                    mask=~torch.from_numpy(b["dones"][:-1]).to(dev))
        self._lam = np.append(lam_next.cpu().numpy(), self.cfg.gae_lambda).astype(np.float32)
        out = super()._learn()
        out.update(self.atd.last)
        return out

    def _gae_lambda(self):
        return self.cfg.gae_lambda if self._lam is None else self._lam

    def state_dict(self):
        return {**super().state_dict(), "atd": self.atd.state_dict(), "atd_cfg": asdict(self.atd_cfg)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "atd" in state:
            self.atd.load_state_dict(state["atd"])

    def describe(self) -> str:
        return f"{super().describe()} + ATD(lambda)"
