"""MACPF: Multi-Agent Conditional Policy Factorization (Wang, Ye & Lu, ICLR 2023, arXiv:2209.12681).

Maximum-entropy actor-critic (FOP-style) whose joint policy during training is *dependent*,
``pi_jt(a|s) = prod_i pi_i^dep(a_i | s, a_<i)``: agent i sees the actions already chosen by agents 0..i-1
(sec. 3.1). For decentralised execution each agent also learns an *independent* policy
``pi_i^ind(a_i | tau_i)``, with ``pi_i^dep = pi_i^ind + b_i^dep`` and ``Q_i^dep = Q_i^ind + c_i^dep`` (logits /
values; stop-gradient on the independent part, sec. 3.2). Two joint critics are trained, each with its own
mixer (eqs. 14-17), and four losses: soft TD(lambda) for each critic and ``E_pi[alpha log pi_i - Q_i]`` for each
policy (eqs. 18-19). The dependent joint policy collects the data; only ``pi^ind`` is used at execution.

Settings from the paper (appendix C.2, SMAC): lr 5e-4; critic and policy networks
``FC(64) -> ReLU -> GRU(64) -> FC``; correction networks ``FC(64) -> ELU -> FC``; targets updated every 200
training episodes; temperatures alpha annealed from 0.5 to 0.05 over 200k steps (easy / hard maps). The mixer
is a QMIX-style hypernetwork mixer *without* the ELU non-linearity (appendix C.1 / E).
Not given in the paper, taken from the authors' PyMARL-based code (``dfop.yaml``) as in ``vdn_lab``:
RMSprop (alpha 0.99, eps 1e-5), grad clip 10, TD(lambda) 0.8, batches of 32 episodes from a 5000-episode
buffer, one update per episode, exploration by mixing the behaviour policy with uniform (epsilon 1 -> 0.05
over 50k steps), last action in the agents' inputs, greedy evaluation of ``pi^ind``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algos import Algo, lambda_returns, masked_mean, t_
from .buffer import Batch, EpisodeBuffer
from .nets import hard_update


@dataclass
class MACPFConfig:
    hidden: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    td_lambda: float = 0.8
    grad_clip: float = 10.0
    batch: int = 32
    buffer: int = 5000
    target: int = 200                    # episodes
    alpha_start: float = 0.5
    alpha_end: float = 0.05
    alpha_anneal: int = 200_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal: int = 50_000
    mixing_embed: int = 64
    hypernet_embed: int = 64
    dep_embed: int = 64


class DependentNet(nn.Module):
    """Shared by all agents: ``fc1 -> ReLU -> GRU -> fc2`` gives the independent output; the dependent output
    adds a correction from the GRU state and an embedding of the preceding agents' [id, action]."""

    def __init__(self, in_dim: int, n: int, A: int, hidden: int, dep_embed: int):
        super().__init__()
        self.n, self.hidden = n, hidden
        self.fc1 = nn.Linear(in_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, A)
        self.dep_fc1 = nn.Linear(n + A, dep_embed)
        self.dep_fc2 = nn.Linear(dep_embed + hidden, A)

    def trunk_seq(self, x: torch.Tensor) -> torch.Tensor:
        """x ``(B, T, n, in)`` -> GRU states ``(B, T, n, H)`` from zeros."""
        B, T, n, _ = x.shape
        z = F.relu(self.fc1(x)).permute(0, 2, 1, 3).reshape(B * n, T, -1)
        return self.rnn(z)[0].reshape(B, n, T, -1).permute(0, 2, 1, 3)

    def trunk_step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x ``(N, in)``, h ``(N, H)`` -> new h."""
        return self.rnn(F.relu(self.fc1(x))[:, None], h[None])[1][0]

    def dep_embedding(self, prev: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
        """prev ``(..., n, n+A)`` one-hot [id, action]; prec[i, j] = 1 iff j < i -> ``(..., n, E)``."""
        return torch.matmul(prec, F.elu(self.dep_fc1(prev))) / self.n

    def dep_head(self, h, ind, emb):
        return ind.detach() + self.dep_fc2(torch.cat([h, emb], -1))


class HyperMixer(nn.Module):
    """QMIX-style monotonic hypernetwork mixer without the hidden non-linearity (MACPF appendix C)."""

    def __init__(self, n: int, state_dim: int, embed: int, hyper_embed: int):
        super().__init__()
        self.n, self.embed = n, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(), nn.Linear(hyper_embed, embed * n))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(), nn.Linear(hyper_embed, embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.V = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """q ``(B, T, n)``, state ``(B, T, S)`` -> ``(B, T, 1)``."""
        w1 = self.hyper_w1(state).abs().view(*state.shape[:-1], self.n, self.embed)
        hidden = (q.unsqueeze(-2) @ w1).squeeze(-2) + self.hyper_b1(state)
        return ((hidden * self.hyper_w2(state).abs()).sum(-1) + self.V(state).squeeze(-1)).unsqueeze(-1)


class MACPF(Algo):
    action_dist = "pi"

    def __init__(self, n_agents, n_actions, obs_dim, T, seed, **kw):
        super().__init__(n_agents, n_actions, obs_dim, T, seed)
        self.cfg = c = MACPFConfig(**kw)
        n, A = n_agents, n_actions
        in_dim = obs_dim + A + n                                      # obs, last action, agent id
        self.actor = DependentNet(in_dim, n, A, c.hidden, c.dep_embed)
        self.critic = DependentNet(in_dim, n, A, c.hidden, c.dep_embed)
        self.target_critic = DependentNet(in_dim, n, A, c.hidden, c.dep_embed)
        mk = lambda: HyperMixer(n, self.S, c.mixing_embed, c.hypernet_embed)
        self.mixer = {"ind": mk(), "dep": mk()}
        self.target_mixer = {"ind": mk(), "dep": mk()}
        for k in ("ind", "dep"):
            hard_update(self.target_mixer[k], self.mixer[k])
        hard_update(self.target_critic, self.critic)
        self.critic_params = list(self.critic.parameters()) + [p for m in self.mixer.values() for p in m.parameters()]
        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr=c.lr, alpha=0.99, eps=1e-5)
        self.critic_opt = torch.optim.RMSprop(self.critic_params, lr=c.lr, alpha=0.99, eps=1e-5)
        self.eye = torch.eye(n)
        self.prec = torch.tril(torch.ones(n, n), -1)
        self.buffer = EpisodeBuffer(c.buffer, T, n, obs_dim, self.rng)
        self.episodes = 0
        self.last_hard = 0
        self.t_env = 0
        self.last_act = None

    # ------------------------------------------------------------------ schedules / helpers
    def _sched(self, start, end, horizon, t):
        return max(end, start - (start - end) * t / horizon) if start >= end else min(end, start + (end - start) * t / horizon)

    def epsilon(self, t):
        return self._sched(self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_anneal, t)

    def alpha(self, t):
        return self._sched(self.cfg.alpha_start, self.cfg.alpha_end, self.cfg.alpha_anneal, t)

    def _x(self, obs: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """obs ``(..., n, od)``, last actions ``(..., n)`` (-1 = none) -> inputs ``(..., n, in)``."""
        la = F.one_hot(last.clamp(min=0), self.A).float() * (last >= 0).unsqueeze(-1).float()
        return torch.cat([obs, la, self.eye.expand(*obs.shape[:-1], self.n)], -1)

    def _joint_onehot(self, acts: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.eye.expand(*acts.shape, self.n), F.one_hot(acts, self.A).float()], -1)

    def _mixeps(self, logits, eps):
        return (1 - eps) * torch.softmax(logits, -1) + eps / self.A

    def _policy(self, h: torch.Tensor, dep: bool, eps: float, greedy: bool = False):
        """h ``(..., n, H)`` -> probs ``(..., n, A)``, actions ``(..., n)``; dependent mode is sequential."""
        ind = self.actor.fc2(h)
        if not dep:
            p = self._mixeps(ind, eps)
            a = p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
            return p, a
        probs, acts = [], []
        prev = torch.zeros(*h.shape[:-2], self.n, self.n + self.A)
        for i in range(self.n):
            emb = self.actor.dep_embedding(prev, self.prec)[..., i, :]
            p = self._mixeps(self.actor.dep_head(h[..., i, :], ind[..., i, :], emb), eps)
            a = p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
            probs.append(p)
            acts.append(a)
            prev = prev.clone()
            prev[..., i, :] = torch.cat([self.eye[i].expand(*a.shape, self.n), F.one_hot(a, self.A).float()], -1)
        return torch.stack(probs, -2), torch.stack(acts, -1)

    def _critic_q(self, net: DependentNet, h, acts, dep: bool):
        q = net.fc2(h)
        if not dep:
            return q
        return net.dep_head(h, q, net.dep_embedding(self._joint_onehot(acts), self.prec))

    # ------------------------------------------------------------------ acting
    def init_hidden(self, E: int) -> None:
        self.h = torch.zeros(E * self.n, self.cfg.hidden)
        self.last_act = torch.full((E, self.n), -1, dtype=torch.long)

    @torch.no_grad()
    def act(self, obs, explore, t_env):
        E = obs.shape[0]
        x = self._x(t_(obs), self.last_act).reshape(E * self.n, -1)
        self.h = self.actor.trunk_step(x, self.h)
        h = self.h.reshape(E, self.n, -1)
        if explore:                                      # behaviour: dependent joint policy, eps-mixed
            _, a = self._policy(h, dep=True, eps=self.epsilon(t_env))
        else:                                            # execution: independent policies, greedy
            _, a = self._policy(h, dep=False, eps=0.0, greedy=True)
        self.last_act = a
        return a.numpy()

    # ------------------------------------------------------------------ learning
    def train(self, batch: Batch, t_env: int) -> Dict[str, float]:
        c = self.cfg
        self.buffer.insert(batch)
        self.episodes += len(batch["length"])
        out: Dict[str, float] = {}
        if self.buffer.can_sample(c.batch):
            for _ in range(len(batch["length"])):
                out = self._update(t_env)
        if self.episodes - self.last_hard >= c.target:
            hard_update(self.target_critic, self.critic)
            for k in ("ind", "dep"):
                hard_update(self.target_mixer[k], self.mixer[k])
            self.last_hard = self.episodes
        return out

    def _update(self, t_env: int) -> Dict[str, float]:
        c = self.cfg
        b = self.buffer.get(self.buffer.sample_idx(c.batch))
        obs, act = t_(b["obs"]), t_(b["actions"], torch.long)          # (B, T+1, n, od), (B, T+1, n)
        r, term, mask = t_(b["reward"]), t_(b["term"]), t_(b["filled"])
        Tm = r.shape[1]
        last = torch.cat([torch.full_like(act[:, :1], -1), act[:, :-1]], 1)
        x = self._x(obs, last)                                          # (B, T+1, n, in)
        state = obs.flatten(-2)
        alpha, eps = self.alpha(t_env), self.epsilon(t_env)
        m = mask[..., None]
        stats: Dict[str, float] = {}
        with torch.no_grad():
            h_crit = self.critic.trunk_seq(x)
        # ---- policies: E_pi [alpha log pi_i - Q_i]  (ind and dep)
        h_act = self.actor.trunk_seq(x)
        actor_loss = 0.0
        for dep in (False, True):
            probs, acts = self._policy(h_act[:, :Tm], dep, eps)
            with torch.no_grad():
                q = self._critic_q(self.critic, h_crit[:, :Tm], acts, dep)
            logp = torch.log(probs.clamp(min=1e-10))
            per = (probs * (alpha * logp - q)).sum(-1)
            actor_loss = actor_loss + masked_mean(per, m)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
        self.actor_opt.step()
        # ---- critics: soft TD(lambda) for Q^ind and Q^dep
        h_crit = self.critic.trunk_seq(x)
        with torch.no_grad():
            h_act = self.actor.trunk_seq(x)[:, 1:]
            h_tgt = self.target_critic.trunk_seq(x)[:, 1:]
        critic_loss = 0.0
        for dep in (False, True):
            k = "dep" if dep else "ind"
            with torch.no_grad():
                probs, a2 = self._policy(h_act, dep, eps)               # a' ~ pi at s_{t+1}
                tq = self._critic_q(self.target_critic, h_tgt, a2, dep).gather(-1, a2[..., None]).squeeze(-1)
                logp2 = torch.log(probs.gather(-1, a2[..., None]).squeeze(-1).clamp(min=1e-10)).sum(-1, keepdim=True)
                boot = self.target_mixer[k](tq, state[:, 1:]) - alpha * logp2
                lambda_returns.gamma = c.gamma
                G = lambda_returns(r, term, mask, boot, c.td_lambda)
            q = self._critic_q(self.critic, h_crit[:, :Tm], act[:, :Tm], dep).gather(-1, act[:, :Tm, :, None]).squeeze(-1)
            q_jt = self.mixer[k](q, state[:, :Tm])
            critic_loss = critic_loss + masked_mean((q_jt - G) ** 2, m)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, c.grad_clip)
        self.critic_opt.step()
        self.train_steps += 1
        stats.update(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()), alpha=alpha)
        return stats
