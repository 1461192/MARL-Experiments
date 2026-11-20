"""The algorithms of Papoudakis et al. (2020) and the adaptive TD(lambda) variants, EPyMARL style.

All agents share parameters and get a one-hot id (EPyMARL default). Every learner sees the *common*
reward (sum of the agents' rewards).

* ``QLearner``   IQL / VDN / QMIX / QPLEX (+ ``td_lambda`` for fixed-lambda returns, + ``atd`` for ATD(lambda)).
                 EPyMARL ``q_learner``: episode runner, a replay of 5000 episodes, one update of 32 sampled
                 episodes after every episode, Double-Q targets, masked MSE, Adam, grad-norm clip 10.
* ``ACLearner``  IA2C (local critic), MAA2C / Central-V (centralised V critic), MAPPO (PPO clip + epochs),
                 COMA (centralised counterfactual Q critic). EPyMARL ``actor_critic_learner`` /
                 ``ppo_learner`` / ``coma_learner``: 10 parallel episodes per update, n-step critic targets
                 from a target critic, entropy bonus. ``atd`` (MAPPO): critic regresses on ATD lambda-returns.

ATD(lambda) (Deng, Wang & Zhang 2026): a density-ratio network ``omega`` is trained with a BCE to tell a
small on-policy buffer (label 1) from the large replay (label 0); ``sigmoid(omega(s_t, a_t))`` is lambda_t in

    G_t = r_t + gamma (1 - term_t) [ (1 - lambda_t) Q_target(s_{t+1}) + lambda_t G_{t+1} ]      (eq. 10)
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import Batch, EpisodeBuffer, trim
from .nets import (AgentNet, LambdaNet, MLPCritic, QMixer, QPLEXMixer, agent_ids, hard_update,
                   masked_mean, soft_update)


# --------------------------------------------------------------------------------------------- utilities
class RunningMeanStd:
    """EPyMARL's ``RunningMeanStd`` (parallel mean / variance update)."""

    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64).ravel()
        if x.size == 0:
            return
        bm, bv, bc = x.mean(), x.var(), x.size
        delta, tot = bm - self.mean, self.count + bc
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.mean, self.var, self.count = self.mean + delta * bc / tot, m2 / tot, tot

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / float(np.sqrt(self.var) + 1e-8)


def lambda_returns(r: torch.Tensor, term: torch.Tensor, mask: torch.Tensor, boot: torch.Tensor,
                   lam) -> torch.Tensor:
    """Forward-view lambda-returns over padded episodes.

    r, term, mask ``(B, T)``; boot ``(B, T, K)`` = bootstrap value of s_{t+1}; lam float or ``(B, T)``.
    G_t = r_t + gamma (1 - term_t) [(1 - l_t) boot_t + l_t G_{t+1}], with l_t := 0 at an episode's last
    step (so the last step bootstraps on truncation and not on termination). gamma is folded into boot
    and the caller passes ``gamma`` via ``lambda_returns.gamma``."""
    gamma = lambda_returns.gamma
    B, T = r.shape
    lam_t = torch.as_tensor(lam, dtype=torch.float32).expand(B, T) if not torch.is_tensor(lam) else lam
    nxt_valid = torch.cat([mask[:, 1:], torch.zeros(B, 1)], 1)
    lam_eff = (lam_t * nxt_valid).unsqueeze(-1)
    G = torch.zeros_like(boot)
    nxt = torch.zeros_like(boot[:, 0])
    for t in reversed(range(T)):
        nxt = r[:, t, None] + gamma * (1 - term[:, t, None]) * ((1 - lam_eff[:, t]) * boot[:, t] + lam_eff[:, t] * nxt)
        G[:, t] = nxt
    return G


lambda_returns.gamma = 0.99


def nstep_returns(r: torch.Tensor, term: torch.Tensor, length: torch.Tensor, vals: torch.Tensor,
                  n: int, gamma: float) -> torch.Tensor:
    """n-step returns ``sum_{k<m} g^k r_{t+k} + g^m (1 - terminal) vals[t+m]``, m = min(n, L - t).

    r, term ``(B, T)``; length ``(B,)``; vals ``(B, T+1, K)`` (value of each state, ``vals[:, L]`` = final)."""
    B, T = r.shape
    t = torch.arange(T)[None]
    end = torch.minimum(t + n, length[:, None])                                # (B, T)
    m = (end - t).clamp_min(0)
    rp = torch.cat([r, torch.zeros(B, n)], 1)
    G = torch.zeros(B, T)
    for k in range(n):
        G = G + (gamma ** k) * rp[:, k:k + T] * (k < m).float()
    term_ep = term.sum(1).clamp(max=1.0)
    notterm = 1.0 - (end == length[:, None]).float() * term_ep[:, None]
    K = vals.shape[-1]
    v_end = vals.gather(1, end.clamp(max=vals.shape[1] - 1)[..., None].expand(B, T, K))
    return G[..., None] + (torch.pow(gamma, m.float()) * notterm)[..., None] * v_end


def t_(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


class Algo:
    off_policy = True
    action_dist = "q"           # "q" (epsilon-greedy), "pi" (sample the policy), "det" (MADDPG)

    def __init__(self, n_agents: int, n_actions: int, obs_dim: int, T: int, seed: int):
        self.n, self.A, self.od, self.T = n_agents, n_actions, obs_dim, T
        self.S = n_agents * obs_dim
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.train_steps = 0
        self.lam_log: List[tuple] = []     # (t_env, mean, std, p10, p50, p90, frac<0.1, frac>0.9)
        self.atd_log: List[tuple] = []     # (t_env, bce loss, accuracy)
        self.h: Optional[torch.Tensor] = None

    # ----------------------------------------------------------- acting
    def _inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs ``(..., n, od)`` -> ``(..., n, od + n)`` (agent id appended)."""
        ids = torch.eye(self.n).expand(*obs.shape[:-2], self.n, self.n)
        return torch.cat([obs, ids], -1)

    def init_hidden(self, E: int) -> None:
        self.h = self.actor.init_hidden(E * self.n)

    @torch.no_grad()
    def _logits(self, obs: np.ndarray) -> torch.Tensor:
        E = obs.shape[0]
        x = self._inputs(t_(obs)).reshape(E * self.n, -1)
        out, self.h = self.actor.step(x, self.h)
        return out.reshape(E, self.n, -1)

    def act(self, obs: np.ndarray, explore: bool, t_env: int) -> np.ndarray:
        raise NotImplementedError

    def train(self, batch: Batch, t_env: int) -> Dict[str, float]:
        raise NotImplementedError

    # ----------------------------------------------------------- ATD bookkeeping
    def _record_lambda(self, t_env: int, lam: torch.Tensor) -> None:
        lam = lam.flatten()
        if lam.numel() == 0:
            return
        q = torch.quantile(lam, torch.tensor([0.1, 0.5, 0.9])).tolist()
        self.lam_log.append((t_env, float(lam.mean()), float(lam.std()) if lam.numel() > 1 else 0.0, *q,
                             float((lam < 0.1).float().mean()), float((lam > 0.9).float().mean())))


class ATDEstimator:
    """omega_phi with its Adam optimiser (paper: lfiw_optim Adam, lr 1e-3) and the BCE step of eq. 8."""

    def __init__(self, state_dim: int, n: int, A: int, hidden: int = 64, lr: float = 1e-3):
        self.net = LambdaNet(state_dim, n, A, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def step(self, on: Batch, off: Batch, n: int) -> tuple:
        def bce(b: Batch, label: float):
            st = t_(b["obs"][:, :-1]).flatten(-2)
            logits = self.net(st, t_(b["actions"][:, :-1], torch.long))
            m = t_(b["filled"])
            loss = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label), reduction="none")
            acc = ((logits > 0).float() == label).float()
            return masked_mean(loss, m), masked_mean(acc, m)
        l_on, a_on = bce(on, 1.0)
        l_off, a_off = bce(off, 0.0)
        loss = l_on + l_off
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.detach()), float(0.5 * (a_on + a_off))

    @torch.no_grad()
    def lambdas(self, b: Batch) -> torch.Tensor:
        """``(B, T)`` lambda for every step of the (trimmed) batch."""
        return torch.sigmoid(self.net(t_(b["obs"][:, :-1]).flatten(-2), t_(b["actions"][:, :-1], torch.long)))


# ------------------------------------------------------------------------------------------- Q-learning
@dataclass
class QConfig:
    hidden: int = 64
    lr: float = 5e-4
    optim: str = "adam"                  # "adam" (EPyMARL) | "rmsprop" (PyMARL: alpha 0.99, eps 1e-5)
    use_rnn: bool = True
    mixer: Optional[str] = None          # None (IQL) | "vdn" | "qmix" | "qplex"
    qplex_heads: int = 4
    qplex_layers: int = 1                # layers of the w, b, lambda, phi, upsilon networks (QPLEX table 2, SMAC: 1)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal: int = 50_000
    eval_eps: float = 0.0
    target: float = 200                  # > 1: hard copy every N episodes; <= 1: soft update (tau) per gradient step
    std_rewards: bool = True
    buffer: int = 5000                   # episodes
    batch: int = 32                      # episodes
    updates_per_episode: int = 1         # gradient steps after each collected episode (QPLEX paper: 2)
    gamma: float = 0.99
    grad_clip: float = 10.0
    double_q: bool = True
    td_lambda: float = 0.0               # fixed-lambda returns (0 = 1-step targets)
    atd: bool = False                    # adaptive TD(lambda)
    atd_ratio: int = 50                  # |D_off| / |D_on|
    atd_period: int = 200                # episodes between estimator steps / lambda-cache clears (= target period)
    atd_steps: int = 1
    atd_lr: float = 1e-3
    atd_hidden: int = 64


def make_optim(params, cfg) -> torch.optim.Optimizer:
    if cfg.optim == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, alpha=0.99, eps=1e-5)
    return torch.optim.Adam(params, lr=cfg.lr)


class QLearner(Algo):
    def __init__(self, n_agents, n_actions, obs_dim, T, seed, **kw):
        super().__init__(n_agents, n_actions, obs_dim, T, seed)
        self.cfg = c = QConfig(**kw)
        self.actor = AgentNet(obs_dim + n_agents, c.hidden, n_actions, c.use_rnn)
        self.target_actor = AgentNet(obs_dim + n_agents, c.hidden, n_actions, c.use_rnn)
        hard_update(self.target_actor, self.actor)
        params = list(self.actor.parameters())
        self.mixer = self.target_mixer = None
        if c.mixer in ("qmix", "qplex"):
            make = (lambda: QMixer(n_agents, self.S)) if c.mixer == "qmix" else \
                (lambda: QPLEXMixer(n_agents, n_actions, self.S, c.qplex_heads, c.qplex_layers))
            self.mixer, self.target_mixer = make(), make()
            hard_update(self.target_mixer, self.mixer)
            params += list(self.mixer.parameters())
        self.params = params
        self.opt = make_optim(params, c)
        self.buffer = EpisodeBuffer(c.buffer, T, n_agents, obs_dim, self.rng)
        self.rew_ms = RunningMeanStd()
        self.episodes = 0
        self.last_hard = 0
        self.last_atd = 0
        if c.atd:
            self.atd = ATDEstimator(self.S, n_agents, n_actions, c.atd_hidden, c.atd_lr)
            self.lam_cache = np.full((c.buffer, T), np.nan, np.float32)
            self.atd_small = max(c.batch, c.buffer // c.atd_ratio)

    def epsilon(self, t_env: int) -> float:
        c = self.cfg
        return max(c.eps_end, c.eps_start - (c.eps_start - c.eps_end) * t_env / c.eps_anneal)

    def act(self, obs, explore, t_env):
        q = self._logits(obs)
        greedy = q.argmax(-1).numpy()
        eps = self.epsilon(t_env) if explore else self.cfg.eval_eps
        if eps <= 0:
            return greedy
        rand = self.rng.integers(0, self.A, greedy.shape)
        return np.where(self.rng.random(greedy.shape) < eps, rand, greedy)

    def _mix(self, mixer, q_all, acts, states):
        """q_all ``(B, T, n, A)``, acts ``(B, T, n)`` -> ``(B, T, K)`` (K = 1, or n for IQL)."""
        qs = q_all.gather(-1, acts[..., None]).squeeze(-1)
        if self.cfg.mixer == "vdn":
            return qs.sum(-1, keepdim=True)
        if self.cfg.mixer == "qmix":
            return mixer(qs, states)
        if self.cfg.mixer == "qplex":
            return mixer(q_all, acts, states)
        return qs                                            # IQL: one target per agent

    def _atd_lambda(self, idx: np.ndarray, b: Batch, t_env: int) -> torch.Tensor:
        Tm = b["reward"].shape[1]
        cached = self.lam_cache[idx, :Tm]
        valid = b["filled"] > 0
        if np.isnan(cached[valid]).any():
            lam = self.atd.lambdas(b).numpy()
            cached = np.where(np.isnan(cached), lam, cached)
            self.lam_cache[idx, :Tm] = np.where(valid, cached, np.nan)
        lam = torch.from_numpy(np.nan_to_num(cached, nan=0.0))
        used = valid.copy()
        used[np.arange(len(idx)), b["length"] - 1] = False   # lambda at the last step is never used
        self._record_lambda(t_env, lam[torch.from_numpy(used)])
        return lam

    def _atd_update(self, t_env: int) -> None:
        c = self.cfg
        if not self.buffer.can_sample(self.atd_small + c.batch):
            return
        for _ in range(c.atd_steps):     # D_on = newest atd_small episodes, D_off = the rest of the replay
            on = self.buffer.get(self.buffer.sample_idx(c.batch, recent=self.atd_small))
            off = self.buffer.get(self.buffer.sample_idx(c.batch, skip_recent=self.atd_small))
            loss, acc = self.atd.step(on, off, self.n)
            self.atd_log.append((t_env, loss, acc))
        self.lam_cache[:] = np.nan

    def train(self, batch: Batch, t_env: int) -> Dict[str, float]:
        c = self.cfg
        idx_new = self.buffer.insert(batch)
        self.episodes += len(batch["length"])
        if c.atd:
            self.lam_cache[idx_new] = np.nan
        out: Dict[str, float] = {}
        if self.buffer.can_sample(c.batch):
            for _ in range(c.updates_per_episode * len(batch["length"])):
                out = self._update(t_env)
        if c.target > 1.0 and self.episodes - self.last_hard >= c.target:
            hard_update(self.target_actor, self.actor)
            if self.mixer is not None:
                hard_update(self.target_mixer, self.mixer)
            self.last_hard = self.episodes
        if c.atd and self.episodes - self.last_atd >= c.atd_period:
            self._atd_update(t_env)
            self.last_atd = self.episodes
        return out

    def _update(self, t_env: int) -> Dict[str, float]:
        c = self.cfg
        idx = self.buffer.sample_idx(c.batch)
        b = self.buffer.get(idx)
        obs, act = t_(b["obs"]), t_(b["actions"], torch.long)
        r, term, mask = t_(b["reward"]), t_(b["term"]), t_(b["filled"])
        Tm = r.shape[1]
        if c.std_rewards:
            self.rew_ms.update(b["reward"][b["filled"] > 0])
            r = self.rew_ms.normalise(r) * mask
        x = self._inputs(obs)
        states = obs.flatten(-2)
        q = self.actor.forward_seq(x)                                    # (B, Tm+1, n, A)
        chosen = self._mix(self.mixer, q[:, :Tm], act[:, :Tm], states[:, :Tm])
        with torch.no_grad():
            tq = self.target_actor.forward_seq(x)[:, 1:]
            a_star = (q[:, 1:] if c.double_q else tq).argmax(-1)
            boot = self._mix(self.target_mixer, tq, a_star, states[:, 1:])
            lam = self._atd_lambda(idx, b, t_env) if c.atd else c.td_lambda
            lambda_returns.gamma = c.gamma
            G = lambda_returns(r, term, mask, boot, lam)
        loss = masked_mean((chosen - G) ** 2, mask[..., None])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, c.grad_clip)
        self.opt.step()
        self.train_steps += 1
        if c.target <= 1.0:
            soft_update(self.target_actor, self.actor, c.target)
            if self.mixer is not None:
                soft_update(self.target_mixer, self.mixer, c.target)
        return {"loss": float(loss.detach()), "q": float(masked_mean(chosen.detach(), mask[..., None]))}


# --------------------------------------------------------------------------------------- actor-critic
@dataclass
class ACConfig:
    hidden: int = 128
    lr: float = 5e-4
    use_rnn: bool = True
    critic: str = "central"              # "local" (IA2C) | "central" (MAA2C, MAPPO) | "coma"
    entropy: float = 0.01
    nstep: int = 5
    target: float = 0.01                 # <= 1: soft tau per update; > 1: hard copy every N updates
    std_rewards: bool = True
    epochs: int = 1                      # PPO epochs (MAPPO: 4)
    ppo_clip: Optional[float] = None     # None -> A2C loss; float -> PPO clipped surrogate
    gamma: float = 0.99
    grad_clip: float = 10.0
    critic_lambda: Optional[float] = None  # fixed-lambda control: critic regresses on lambda-returns with this lambda
    atd: bool = False                    # MAPPO-ATD: critic regresses on ATD lambda-returns
    atd_ratio: int = 50                  # D_off = the last 50 batches, D_on = the current batch
    atd_steps: int = 1                   # estimator steps per update: the target critic is updated every update
    atd_lr: float = 1e-3
    atd_hidden: int = 64


class ACLearner(Algo):
    off_policy = False
    action_dist = "pi"

    def __init__(self, n_agents, n_actions, obs_dim, T, seed, **kw):
        super().__init__(n_agents, n_actions, obs_dim, T, seed)
        self.cfg = c = ACConfig(**kw)
        self.actor = AgentNet(obs_dim + n_agents, c.hidden, n_actions, c.use_rnn)
        if c.critic == "local":
            cin, cout = obs_dim + n_agents, 1
        elif c.critic == "central":
            cin, cout = self.S + n_agents, 1
        elif c.critic == "coma":
            cin, cout = self.S + n_agents * n_actions + n_agents, n_actions
        else:
            raise ValueError(c.critic)
        self.critic, self.target_critic = MLPCritic(cin, c.hidden, cout), MLPCritic(cin, c.hidden, cout)
        hard_update(self.target_critic, self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.rew_ms = RunningMeanStd()
        self.last_hard = 0
        if c.atd:
            self.atd = ATDEstimator(self.S, n_agents, n_actions, c.atd_hidden, c.atd_lr)
            self.history: deque = deque(maxlen=c.atd_ratio)

    def act(self, obs, explore, t_env):
        probs = F.softmax(self._logits(obs), -1).numpy().astype(np.float64)
        cum = probs.cumsum(-1)
        u = self.rng.random(probs.shape[:-1] + (1,))
        return np.minimum((u > cum).sum(-1), self.A - 1)

    def _critic_in(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        B, T1, n, _ = obs.shape
        ids = agent_ids(B, T1, n)
        if self.cfg.critic == "local":
            return torch.cat([obs, ids], -1)
        state = obs.flatten(-2)[:, :, None].expand(B, T1, n, self.S)
        if self.cfg.critic == "central":
            return torch.cat([state, ids], -1)
        onehot = F.one_hot(act, self.A).float()                                # (B, T1, n, A)
        others = onehot.flatten(-2)[:, :, None].expand(B, T1, n, n * self.A)
        own = (1 - torch.eye(n)).repeat_interleave(self.A, -1)                  # (n, n*A): hide own action
        return torch.cat([state, others * own, ids], -1)

    def _values(self, net, cin, act) -> torch.Tensor:
        out = net(cin)
        if self.cfg.critic == "coma":
            return out                                                          # (B, T1, n, A)
        return out.squeeze(-1)                                                  # (B, T1, n)

    def _atd_lambda(self, b: Batch, t_env: int) -> torch.Tensor:
        """D_on = this batch; D_off = the previous ``atd_ratio`` batches. Before D_off has data the estimator
        is not trained (lambda = sigmoid of an untrained net, about 0.5)."""
        c = self.cfg
        if self.history:
            for _ in range(c.atd_steps):
                pick = self.rng.integers(0, len(self.history), b["length"].shape[0])
                ep = np.array([self.rng.integers(0, self.history[i]["length"].shape[0]) for i in pick])
                off = trim(_stack_eps([self.history[i] for i in pick], ep, self.T))
                loss, acc = self.atd.step(b, off, self.n)
            self.atd_log.append((t_env, loss, acc))
        self.history.append(b)
        lam = self.atd.lambdas(b)
        used = b["filled"] > 0
        used[np.arange(len(used)), b["length"] - 1] = False
        self._record_lambda(t_env, lam[torch.from_numpy(used)])
        return lam

    def train(self, batch: Batch, t_env: int) -> Dict[str, float]:
        c = self.cfg
        b = trim(batch)
        obs, act = t_(b["obs"]), t_(b["actions"], torch.long)
        r, term, mask = t_(b["reward"]), t_(b["term"]), t_(b["filled"])
        length = t_(b["length"], torch.long)
        Tm = r.shape[1]
        if c.std_rewards:
            self.rew_ms.update(b["reward"][b["filled"] > 0])
            r = self.rew_ms.normalise(r) * mask
        x = self._inputs(obs)
        cin = self._critic_in(obs, act)
        lam = self._atd_lambda(b, t_env) if c.atd else None          # trains the estimator (needs grad)
        with torch.no_grad():
            tv = self._values(self.target_critic, cin, act)
            if c.critic == "coma":
                tv = tv.gather(-1, act[..., None]).squeeze(-1)                   # Q_target(s, a) (B, T1, n)
            G = nstep_returns(r, term, length, tv, c.nstep, c.gamma)             # (B, Tm, n)
            G_critic = G
            if c.atd or c.critic_lambda is not None:
                lambda_returns.gamma = c.gamma
                G_critic = lambda_returns(r, term, mask, tv[:, 1:], lam if c.atd else c.critic_lambda)
            old_logp = None
            if c.ppo_clip is not None:
                old_pi = F.softmax(self.actor.forward_seq(x)[:, :Tm], -1)
                old_logp = torch.log(old_pi.gather(-1, act[:, :Tm, :, None]).squeeze(-1) + 1e-10)
        m = mask[..., None]
        for _ in range(c.epochs):
            v = self._values(self.critic, cin, act)[:, :Tm]
            if c.critic == "coma":
                v_taken = v.gather(-1, act[:, :Tm, :, None]).squeeze(-1)
                critic_loss = masked_mean((G_critic - v_taken) ** 2, m)
            else:
                critic_loss = masked_mean((G_critic - v) ** 2, m)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
            self.critic_opt.step()

            pi = F.softmax(self.actor.forward_seq(x)[:, :Tm], -1)               # (B, Tm, n, A)
            logp = torch.log(pi.gather(-1, act[:, :Tm, :, None]).squeeze(-1) + 1e-10)
            entropy = -(pi * torch.log(pi + 1e-10)).sum(-1)
            if c.critic == "coma":
                qv = v.detach()
                adv = qv.gather(-1, act[:, :Tm, :, None]).squeeze(-1) - (pi.detach() * qv).sum(-1)
            else:
                adv = (G - v).detach()
            if c.ppo_clip is None:
                obj = adv * logp
            else:
                ratio = torch.exp(logp - old_logp)
                obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - c.ppo_clip, 1 + c.ppo_clip) * adv)
            actor_loss = -masked_mean(obj + c.entropy * entropy, m)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
            self.actor_opt.step()
        self.train_steps += 1
        if c.target <= 1.0:
            soft_update(self.target_critic, self.critic, c.target)
        elif self.train_steps - self.last_hard >= c.target:
            hard_update(self.target_critic, self.critic)
            self.last_hard = self.train_steps
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                "entropy": float(masked_mean(entropy.detach(), m))}


def _stack_eps(batches: List[Batch], ep: np.ndarray, T: int) -> Batch:
    """One episode from each of ``batches`` (episode ``ep[i]`` of ``batches[i]``), re-padded to ``T``."""
    out = {}
    for k in ("obs", "actions", "reward", "term", "filled"):
        rows = []
        for bt, e in zip(batches, ep):
            v = bt[k][e]
            full = T + 1 if k in ("obs", "actions") else T
            pad = [(0, full - v.shape[0])] + [(0, 0)] * (v.ndim - 1)
            rows.append(np.pad(v, pad))
        out[k] = np.stack(rows)
    out["length"] = np.array([bt["length"][e] for bt, e in zip(batches, ep)])
    return out
