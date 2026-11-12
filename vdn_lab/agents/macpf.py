"""Multi-Agent Conditional Policy Factorization (MACPF).

Wang, Ye & Lu, "More Centralized Training, Still Decentralized Execution: Multi-Agent
Conditional Policy Factorization", ICLR 2023 (arXiv:2209.12681). Section numbers and equations
below refer to that paper. Where it leaves a detail open, the authors' code is followed
(github.com/PKU-RL/FOP-DMAC-MACPF, ``dfop_*`` modules).

**Conditional factorised soft policy iteration (sec. 3.1).** Maximum-entropy MARL in which
agent ``i``'s policy also takes the actions of the agents before it (eqs. 3-5),

    pi_i(a_i | s, a_<i) = exp((Q_i(s, a_<i, a_i) - V_i(s, a_<i)) / alpha_i),
    pi_jt(a | s) = prod_i pi_i(a_i | s, a_<i),

so the joint policy is no longer restricted by FOP's IGO condition, and alternating joint soft
policy evaluation with individual conditional soft policy improvement converges to the optimal
joint policy (Theorem 1).

**Independent joint policy (sec. 3.2).** For any dependent joint policy there is an independent
one with the same value (Theorem 2), so the independent one can be used for decentralised
execution.

**Framework (sec. 3.3, fig. 3).** Each agent has four modules: an independent policy
``pi_i^ind(a_i | tau_i)``, a dependency policy correction ``b_i^dep(a_i | tau_i, a_<i)``, an
independent critic ``Q_i^ind(tau_i, a_i)`` and a dependency critic correction
``c_i^dep(tau_i, a_<i, a_i)``:

    logits of pi_i^dep = logits of pi_i^ind + b_i^dep                (eq. 10, footnote 1)
    Q_i^dep = Q_i^ind + c_i^dep                                     (eq. 13)
    Q_jt^dep = Mixer([Q_i^dep], s),   Q_jt^ind = Mixer([Q_i^ind], s)  (eqs. 14-15, *one* Mixer)

The dependent joint policy is the behaviour policy. The same replay batch trains both critics
by soft TD (eqs. 16-17, ``a'`` sampled from the current dependent / independent policy) and
both policies by the KL / soft-actor-critic objective (eqs. 18-19; for ``J^dep`` the preceding
agents' actions are sampled from their current dependent policies). "The gradient of L^dep only
updates [c_i^dep]" and "the gradient of J^dep only updates b_i^dep": the dependent heads see a
stop-gradient copy of the independent output *and* of the GRU state (the "Detach" arrows in
fig. 3), so ``L^ind`` / ``J^ind`` alone train the shared trunk. The Mixer is trained by both
critic losses. At execution time only ``pi_i^ind`` is used.

**Mixer (appendix E).** A hypernetwork *without* non-linearity, i.e. a weighted sum of the
``Q_i`` with non-negative weights that depend only on the state (``mixer="linear"``, default;
two-layer hypernetworks, embed 64, as in the code). The paper's ``MACPF_ATT`` ablation, a
FOP/QPLEX-style attention mixer whose weights also depend on the joint action, is available as
``mixer="att"``. The testbed has no global state, so the mixer's state is all agents'
observations concatenated.

**Networks (appendix C.2).** Policy and critic, shared by all agents: ``Linear(64) -> ReLU ->
GRU(64) -> Linear`` on ``[obs, last action, agent id]``. Correction networks: ``Linear(64) ->
ELU -> Linear``. The code embeds each preceding agent's one-hot ``[id, action]`` with the first
layer, sums them / ``n_agents`` and concatenates the GRU state before the second layer.

**Hyper-parameters (appendix C.2, SMAC, plus the code's ``dfop.yaml`` where C.2 is silent).**
lr 5e-4; RMSprop (alpha 0.99, eps 1e-5); grad-norm clip 10; gamma 0.99; batches of 32
episodes; buffer of 5000 episodes; one training step per episode; target networks updated every
200 training episodes; temperature ``alpha`` annealed 0.5 -> 0.05 over 200k steps (C.2: easy
and hard maps; super-hard maps use a fixed 0.001, MPE anneals over 50k steps). The code uses
TD(lambda = 0.8) targets for eqs. 16-17 and mixes the policies with a uniform distribution with
weight epsilon, 1.0 -> 0.05 over 50k steps (used in the losses too).

Episodes in the grid worlds end on a time limit only, so the last state bootstraps from the
target critic (as pymarl does for ``episode_limit``).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_agent
from .base import Agent


@dataclass
class MACPFConfig:
    gamma: float = 0.99
    td_lambda: float = 0.8
    lr: float = 5e-4                    # actor
    critic_lr: float = 5e-4
    optim_alpha: float = 0.99           # RMSprop smoothing
    optim_eps: float = 1e-5
    grad_norm_clip: float = 10.0
    batch_size: int = 32                # episodes per training step
    buffer_size: int = 5000             # episodes
    train_every: int = 1                # training steps after every `train_every` episodes
    target_update_interval: int = 200   # episodes
    alpha_start: float = 0.5            # entropy temperature, annealed linearly (appendix C.2: easy / hard maps)
    alpha_end: float = 0.05
    alpha_anneal_steps: int = 200_000
    hidden: int = 64                    # GRU size
    mixing_embed_dim: int = 64
    hypernet_embed: int = 64
    dep_embed_dim: int = 64
    eps_start: float = 1.0
    eps_finish: float = 0.05
    eps_anneal_steps: int = 50_000
    obs_last_action: bool = True
    obs_agent_id: bool = True
    mixer: str = "linear"               # "linear": hypernetwork without non-linearity (appendix E) | "att": MACPF_ATT ablation
    n_head: int = 4                     # attention heads of the "att" mixer (both attention modules)
    attend_reg_coef: float = 0.001      # penalty on the "att" mixer's attention logits
    shared_mixer: bool = True           # one Mixer for Q_jt^dep and Q_jt^ind (sec. 3.3)
    detach_dep_trunk: bool = True       # L^dep / J^dep only update the correction networks (sec. 3.3)
    eval_policy: str = "ind"            # act(explore=False) with "ind" (decentralised) or "dep" (needs a_<i)
    max_train_len: Optional[int] = None  # train on random windows of this many steps (None: whole episodes)


# ------------------------------------------------------------------------------- networks
class DependentNet(nn.Module):
    """Shared by all agents. ``trunk`` gives the GRU states; ``ind_head`` the independent output
    (logits or soft Q-values); ``dep_head`` the dependent one = stopgrad(ind) + correction."""

    def __init__(self, in_dim: int, n_agents: int, n_actions: int, hidden: int, dep_embed: int):
        super().__init__()
        self.n_agents, self.hidden = n_agents, hidden
        self.fc1 = nn.Linear(in_dim, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, n_actions)
        self.dep_fc1 = nn.Linear(n_agents + n_actions, dep_embed)
        self.dep_fc2 = nn.Linear(dep_embed + hidden, n_actions)
        self.dep_embed = dep_embed

    def trunk(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """x ``(N, T, in)`` -> GRU states ``(N, T, H)``, last state ``(1, N, H)``."""
        z = F.relu(self.fc1(x))
        return self.rnn(z, h0)

    def ind_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(h)

    def dep_embedding(self, prev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """prev ``(..., d, d+nA)`` one-hot [id, action] of every agent, mask ``(d, d)`` with
        ``mask[i, j] = 1`` iff agent ``i`` conditions on agent ``j`` -> ``(..., d, E)``."""
        e = F.elu(self.dep_fc1(prev))                                   # (..., d, E)
        return torch.matmul(mask, e) / self.n_agents

    def dep_head(self, h: torch.Tensor, ind: torch.Tensor, dep_h: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """stopgrad(ind) + correction([h, dep_h]); with ``detach`` the GRU state is a stop-gradient
        input too, so only the correction network learns from the dependent losses."""
        return ind.detach() + self.dep_fc2(torch.cat([h.detach() if detach else h, dep_h], -1))


class HyperMixer(nn.Module):
    """MACPF's mixer (appendix E; reference ``DFOPMixer``): two-layer hypernetworks of the state
    with absolute values, and *no* non-linearity between the mixing layers, so
    ``Q_jt = sum_i w_i(s) Q_i + b(s)`` with ``w_i(s) >= 0``: a weighted sum whose weights depend on
    the state only (eq. 6 with ``alpha / alpha_i = w_i``).
    q ``(B, T, d)``, state ``(B, T, S)`` -> ``(B, T)``."""

    def __init__(self, n_agents: int, state_dim: int, embed: int, hyper_embed: int):
        super().__init__()
        self.n_agents, self.embed = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(),
                                      nn.Linear(hyper_embed, embed * n_agents))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(), nn.Linear(hyper_embed, embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.V = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w1 = self.hyper_w1(state).abs().view(*state.shape[:-1], self.n_agents, self.embed)
        hidden = (q.unsqueeze(-2) @ w1).squeeze(-2) + self.hyper_b1(state)  # (B,T,E); no nonlinearity, as in the reference
        w2 = self.hyper_w2(state).abs()
        return (hidden * w2).sum(-1) + self.V(state).squeeze(-1)


class FOPMixer(nn.Module):
    """FOP's attention mixer (Zhang et al. 2021; reference ``FOPMixer``). MACPF does *not* use it:
    it is the ``MACPF_ATT`` ablation of appendix E (``mixer="att"``), where the joint-action
    dependent weights conflict with the explicit dependency in ``Q_i^dep``.

        Q_jt = sum_i [w_i(s) Q_i + V(s)/d]  +  sum_i (lambda_i(s, a) - 1) * stopgrad(w_i(s) (Q_i - V_i)),

    i.e. ``Q_jt = V_jt + sum_i lambda_i w_i (Q_i - V_i)`` with ``V_jt = sum_i w_i V_i + V(s)``, where
    ``V_i = alpha logsumexp(Q_i / alpha)`` is agent i's soft value, ``w_i > 0`` are Qatten attention
    weights (state selector vs. per-agent "unit" keys, softmax over agents, summed over heads) and
    ``lambda_i > 0`` are QPLEX-style weights of state and joint action. The per-agent soft
    advantages ``Q_i - V_i`` then add up to the joint one, so each agent's Boltzmann policy of
    its own Q is its factor of the joint Boltzmann policy. The unit features are each agent's
    observation (the "state" is all observations concatenated)."""

    def __init__(self, n_agents: int, n_actions: int, unit_dim: int, embed: int, hyper_embed: int, n_head: int,
                 attend_reg_coef: float):
        super().__init__()
        d = n_agents
        self.n_agents, self.unit_dim, self.embed, self.reg = d, unit_dim, embed, attend_reg_coef
        state_dim = d * unit_dim
        self.selectors = nn.ModuleList(nn.Sequential(nn.Linear(state_dim, hyper_embed), nn.ReLU(),
                                                     nn.Linear(hyper_embed, embed, bias=False)) for _ in range(n_head))
        self.key_ext = nn.ModuleList(nn.Linear(unit_dim, embed, bias=False) for _ in range(n_head))
        self.lam_keys = nn.ModuleList(nn.Linear(state_dim, 1) for _ in range(n_head))
        self.lam_agents = nn.ModuleList(nn.Linear(state_dim, d) for _ in range(n_head))
        self.lam_actions = nn.ModuleList(nn.Linear(state_dim + d * n_actions, d) for _ in range(n_head))
        self.V = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))
        self.last_reg = torch.zeros(())

    def forward(self, q: torch.Tensor, state: torch.Tensor, actions_onehot: torch.Tensor,
                vs: torch.Tensor) -> torch.Tensor:
        """q, vs ``(B, T, d)``; state ``(B, T, S)``; actions_onehot ``(B, T, d, nA)`` -> ``(B, T)``."""
        d = self.n_agents
        units = state.view(*state.shape[:-1], d, self.unit_dim)
        w, reg = 0.0, 0.0
        for sel, kx in zip(self.selectors, self.key_ext):
            logits = (kx(units) @ sel(state).unsqueeze(-1)).squeeze(-1)          # (B, T, d)
            w = w + torch.softmax(logits / np.sqrt(self.embed), -1)
            reg = reg + (logits ** 2).mean()
        self.last_reg = self.reg * reg
        w = w + 1e-10
        sa = torch.cat([state, actions_onehot.flatten(-2)], -1)
        lam = 0.0
        for k, ag, ac in zip(self.lam_keys, self.lam_agents, self.lam_actions):
            lam = lam + (k(state).abs() + 1e-10) * torch.sigmoid(ag(state)) * torch.sigmoid(ac(sa))
        adv = (w * (q - vs)).detach()
        v_tot = (w * q).sum(-1) + self.V(state).squeeze(-1)                     # sum_i (w_i Q_i + V/d)
        return v_tot + (adv * (lam - 1)).sum(-1)


def td_lambda_targets(rewards: torch.Tensor, mask: torch.Tensor, target_q: torch.Tensor,
                      gamma: float, lam: float) -> torch.Tensor:
    """Forward-view lambda-returns (pymarl ``build_td_lambda_targets``, no terminal states).

    rewards / mask ``(B, T)``, target_q ``(B, T+1)`` = Q_jt'(tau_t, a'_t). ``mask[b, t]`` is 1 for
    real steps; padded steps after an episode's end have mask 0, and the bootstrap value of a
    shorter episode sits at its own final index (copied into all later columns by the caller)."""
    B, T = rewards.shape
    ret = torch.zeros_like(target_q)
    ret[:, -1] = target_q[:, -1]
    for t in range(T - 1, -1, -1):
        ret[:, t] = (lam * gamma * ret[:, t + 1] + mask[:, t] * (rewards[:, t] + (1 - lam) * gamma * target_q[:, t + 1])
                     ) * mask[:, t] + (1 - mask[:, t]) * target_q[:, t]
    return ret[:, :-1]


# --------------------------------------------------------------------------------- agent
@register_agent("macpf")
class MACPFAgent(Agent):
    """MACPF: dependent (conditional) policies for centralised training, distilled independent
    policies for decentralised execution. Agent kwargs are the ``MACPFConfig`` fields."""

    def __init__(self, n_agents: int, n_actions: int, obs_shape: Tuple[int, ...],
                 device: str = "cpu", seed: Optional[int] = None, **kwargs: Any):
        super().__init__(n_agents, n_actions, obs_shape, device, seed)
        self.cfg = c = MACPFConfig(**kwargs)
        if c.eval_policy not in ("ind", "dep"):
            raise ValueError("eval_policy must be 'ind' or 'dep'")
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        d, nA = n_agents, n_actions
        self.obs_dim = int(np.prod(obs_shape))
        in_dim = self.obs_dim + (nA if c.obs_last_action else 0) + (d if c.obs_agent_id else 0)
        self.actor = DependentNet(in_dim, d, nA, c.hidden, c.dep_embed_dim).to(device)
        self.critic = DependentNet(in_dim, d, nA, c.hidden, c.dep_embed_dim).to(device)
        state_dim = d * self.obs_dim
        if c.mixer not in ("linear", "att"):
            raise ValueError("mixer must be 'linear' or 'att'")

        def make_mixer() -> nn.Module:
            if c.mixer == "att":
                return FOPMixer(d, nA, self.obs_dim, c.mixing_embed_dim, c.hypernet_embed, c.n_head,
                                c.attend_reg_coef).to(device)
            return HyperMixer(d, state_dim, c.mixing_embed_dim, c.hypernet_embed).to(device)

        self.mixer_ind = make_mixer()
        self.mixer_dep = self.mixer_ind if c.shared_mixer else make_mixer()
        self.target_critic = copy.deepcopy(self.critic)
        self.target_mixer_ind = copy.deepcopy(self.mixer_ind)
        self.target_mixer_dep = self.target_mixer_ind if c.shared_mixer else copy.deepcopy(self.mixer_dep)
        for net in (self.target_critic, self.target_mixer_ind, self.target_mixer_dep):
            net.requires_grad_(False)
        critic_params = list(self.critic.parameters()) + list(self.mixer_ind.parameters())
        if not c.shared_mixer:
            critic_params += list(self.mixer_dep.parameters())
        self.critic_params = critic_params
        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr=c.lr, alpha=c.optim_alpha, eps=c.optim_eps)
        self.critic_opt = torch.optim.RMSprop(critic_params, lr=c.critic_lr, alpha=c.optim_alpha, eps=c.optim_eps)

        self._eye = torch.eye(d, device=device)
        self._prec = torch.tril(torch.ones(d, d, device=device), -1)     # agent i conditions on j < i

        self.buffer: List[Dict[str, np.ndarray]] = []
        self.buf_ptr = 0
        self._ep: Dict[str, list] = {"obs": [], "actions": [], "rewards": []}
        self._h: Optional[torch.Tensor] = None
        self._last_actions = np.full(d, -1, np.int64)
        self.env_steps = 0
        self.episodes = 0
        self.n_updates = 0
        self._last_target_update = 0
        self._train_due = False
        self._stats: Dict[str, float] = {}

    # ---------------------------------------------------------------------- helpers
    @property
    def alpha(self) -> float:
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c.alpha_anneal_steps))
        return c.alpha_start + frac * (c.alpha_end - c.alpha_start)

    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c.eps_anneal_steps))
        return c.eps_start + frac * (c.eps_finish - c.eps_start)

    def _inputs(self, obs: torch.Tensor, last_actions: torch.Tensor) -> torch.Tensor:
        """obs uint8 ``(..., d, *obs_shape)``, last actions ``(..., d)`` (-1 = none) -> ``(..., d, in)``."""
        x = [obs.float().div(255.0).reshape(*obs.shape[:-len(self.obs_shape)], -1)]
        if self.cfg.obs_last_action:
            x.append(F.one_hot(last_actions.clamp(min=0), self.n_actions).float() * (last_actions >= 0).unsqueeze(-1))
        if self.cfg.obs_agent_id:
            x.append(self._eye.expand(*x[0].shape[:-1], self.n_agents))
        return torch.cat(x, -1)

    def _onehot_joint(self, actions: torch.Tensor) -> torch.Tensor:
        """actions ``(..., d)`` -> ``(..., d, d + nA)`` one-hot [agent id, action]."""
        return torch.cat([self._eye.expand(*actions.shape, self.n_agents),
                          F.one_hot(actions, self.n_actions).float()], -1)

    def _mix_eps(self, logits: torch.Tensor, eps: float) -> torch.Tensor:
        return (1 - eps) * torch.softmax(logits, -1) + eps / self.n_actions

    def _sequential_policy(self, h: torch.Tensor, dep: bool, eps: float, greedy: bool = False):
        """Run the policy agent by agent. h ``(..., d, H)`` -> probs ``(..., d, nA)``, actions ``(..., d)``.

        In dependent mode agent ``i``'s logits use the actions just chosen for agents ``< i``."""
        ind = self.actor.ind_head(h)
        if not dep:
            probs = self._mix_eps(ind, eps)
            acts = probs.argmax(-1) if greedy else torch.distributions.Categorical(probs=probs).sample()
            return probs, acts
        d = self.n_agents
        probs, acts = [], []
        prev = torch.zeros(*h.shape[:-2], d, d + self.n_actions, device=h.device)
        for i in range(d):
            dep_h = self.actor.dep_embedding(prev, self._prec)[..., i, :]
            logit = self.actor.dep_head(h[..., i, :], ind[..., i, :], dep_h, self.cfg.detach_dep_trunk)
            p = self._mix_eps(logit, eps)
            a = p.argmax(-1) if greedy else torch.distributions.Categorical(probs=p).sample()
            probs.append(p)
            acts.append(a)
            prev = prev.clone()
            prev[..., i, :] = torch.cat([self._eye[i].expand(*a.shape, d), F.one_hot(a, self.n_actions).float()], -1)
        return torch.stack(probs, -2), torch.stack(acts, -1)

    def _critic_q(self, net: DependentNet, h: torch.Tensor, actions: torch.Tensor, dep: bool) -> torch.Tensor:
        """Per-agent soft Q for every own action. h ``(..., d, H)``, actions ``(..., d)`` (only
        the preceding agents' actions are read, in dependent mode) -> ``(..., d, nA)``."""
        q = net.ind_head(h)
        if not dep:
            return q
        dep_h = net.dep_embedding(self._onehot_joint(actions), self._prec)
        return net.dep_head(h, q, dep_h, self.cfg.detach_dep_trunk)

    # ------------------------------------------------------------ Agent interface
    def reset_episode(self) -> None:
        self._h = None
        self._last_actions = np.full(self.n_agents, -1, np.int64)
        self._ep = {"obs": [], "actions": [], "rewards": []}

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        o = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
        x = self._inputs(o, torch.from_numpy(self._last_actions).to(self.device))   # (d, in)
        h_seq, self._h = self.actor.trunk(x.unsqueeze(1), self._h)                  # (d, 1, H)
        h = h_seq[:, 0]
        if explore:
            # behaviour policy: the dependent joint policy, epsilon-mixed with uniform
            _, acts = self._sequential_policy(h, dep=True, eps=self.epsilon)
        else:
            _, acts = self._sequential_policy(h, dep=self.cfg.eval_policy == "dep", eps=0.0, greedy=True)
        acts = acts.cpu().numpy().astype(np.int64)
        self._last_actions = acts
        return acts

    def observe(self, obs, actions, reward, next_obs, done) -> None:
        if not self.training:
            return
        self._ep["obs"].append(np.asarray(obs, np.uint8))
        self._ep["actions"].append(np.asarray(actions, np.int64))
        self._ep["rewards"].append(float(reward))
        self.env_steps += 1
        if done:
            ep = dict(obs=np.stack(self._ep["obs"] + [np.asarray(next_obs, np.uint8)]),
                      actions=np.stack(self._ep["actions"]), rewards=np.asarray(self._ep["rewards"], np.float32))
            if len(self.buffer) < self.cfg.buffer_size:
                self.buffer.append(ep)
            else:
                self.buffer[self.buf_ptr] = ep
            self.buf_ptr = (self.buf_ptr + 1) % self.cfg.buffer_size
            self.episodes += 1
            self._ep = {"obs": [], "actions": [], "rewards": []}
            self._train_due = self.episodes % self.cfg.train_every == 0

    def update(self) -> Dict[str, float]:
        if not (self.training and self._train_due):
            return {}
        self._train_due = False
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        return self.train_step(self._sample())

    # ------------------------------------------------------------------- learning
    def _sample(self) -> Dict[str, torch.Tensor]:
        c, dev = self.cfg, self.device
        idx = self.rng.integers(0, len(self.buffer), c.batch_size)
        eps = []
        for i in idx:
            e = self.buffer[i]
            n, s0 = len(e["rewards"]), 0
            if c.max_train_len and n > c.max_train_len:          # random window, GRU starts from zeros
                s0 = int(self.rng.integers(0, n - c.max_train_len + 1))
                n = c.max_train_len
            eps.append((e, s0, n))
        T = max(n for _, _, n in eps)
        B, d = len(eps), self.n_agents
        obs = np.zeros((B, T + 1, d, *self.obs_shape), np.uint8)
        acts = np.zeros((B, T + 1, d), np.int64)
        last = np.full((B, T + 1, d), -1, np.int64)
        rew = np.zeros((B, T), np.float32)
        mask = np.zeros((B, T), np.float32)
        for b, (e, s0, n) in enumerate(eps):
            obs[b, :n + 1] = e["obs"][s0:s0 + n + 1]
            obs[b, n + 1:] = obs[b, n]
            acts[b, :n] = e["actions"][s0:s0 + n]
            rew[b, :n], mask[b, :n] = e["rewards"][s0:s0 + n], 1.0
            if s0 > 0:
                last[b, 0] = e["actions"][s0 - 1]
            last[b, 1:n + 1] = acts[b, :n]
            last[b, n + 1:] = acts[b, n - 1]
        t = lambda a: torch.from_numpy(a).to(dev)
        return dict(obs=t(obs), actions=t(acts), last=t(last), rewards=t(rew), mask=t(mask))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        x = self._inputs(batch["obs"], batch["last"])                # (B, T+1, d, in)
        for dep in (False, True):
            stats.update(self._train_actor(batch, x, dep))
        for dep in (False, True):
            stats.update(self._train_critic(batch, x, dep))
        self.n_updates += 1
        if self.episodes - self._last_target_update >= self.cfg.target_update_interval:
            self._update_targets()
            self._last_target_update = self.episodes
        stats["epsilon"] = self.epsilon
        stats["alpha"] = self.alpha
        stats["loss"] = stats["critic_loss_ind"] + stats["critic_loss_dep"]
        self._stats = stats
        return stats

    def _gru(self, net: DependentNet, x: torch.Tensor) -> torch.Tensor:
        """x ``(B, T, d, in)`` -> GRU states ``(B, T, d, H)``, each agent's sequence from zeros."""
        B, T, d, _ = x.shape
        h, _ = net.trunk(x.permute(0, 2, 1, 3).reshape(B * d, T, -1))
        return h.reshape(B, d, T, -1).permute(0, 2, 1, 3)

    def _train_actor(self, batch, x, dep: bool) -> Dict[str, float]:
        c, sfx = self.cfg, "dep" if dep else "ind"
        mask = batch["mask"]                                          # (B, T)
        h = self._gru(self.actor, x[:, :-1])
        probs, acts = self._sequential_policy(h, dep, self.epsilon)
        with torch.no_grad():
            q = self._critic_q(self.critic, self._gru(self.critic, x[:, :-1]), acts, dep)
        log_p = torch.log(probs.clamp(min=1e-10))
        per = (probs * (self.alpha * log_p - q)).sum(-1)                # (B, T, d)
        m = mask.unsqueeze(-1).expand_as(per)
        loss = (per * m).sum() / m.sum()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_norm_clip)
        self.actor_opt.step()
        ent = (-(probs * log_p).sum(-1) * m).sum() / m.sum()
        return {f"policy_loss_{sfx}": loss.item(), f"entropy_{sfx}": ent.item(), f"actor_grad_{sfx}": float(gn)}

    def _train_critic(self, batch, x, dep: bool) -> Dict[str, float]:
        c, sfx = self.cfg, "dep" if dep else "ind"
        mask, rewards, actions = batch["mask"], batch["rewards"], batch["actions"]
        B, T1, d = actions.shape
        state = batch["obs"].float().div(255.0).reshape(B, T1, -1)
        mixer, target_mixer = (self.mixer_dep, self.target_mixer_dep) if dep else (self.mixer_ind, self.target_mixer_ind)
        with torch.no_grad():
            probs, next_a = self._sequential_policy(self._gru(self.actor, x), dep, self.epsilon)   # (B, T+1, d, ...)
            tq = self._critic_q(self.target_critic, self._gru(self.target_critic, x), next_a, dep)
            target_q = self._mix(target_mixer, tq, next_a, state)                              # (B, T+1)
            # bootstrap of a shorter (padded) episode: its own final step's value
            lengths = mask.sum(1).long()
            ar = torch.arange(T1, device=x.device)
            final = target_q.gather(1, lengths.unsqueeze(1))
            target_q = torch.where(ar.unsqueeze(0) > lengths.unsqueeze(1), final, target_q)
            targets = td_lambda_targets(rewards, mask, target_q, c.gamma, c.td_lambda)
            logp_next = torch.log(probs.gather(-1, next_a.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10))
            targets = targets - self.alpha * logp_next[:, 1:].mean(-1)
        q = self._critic_q(self.critic, self._gru(self.critic, x[:, :-1]), actions[:, :-1], dep)
        q_jt = self._mix(mixer, q, actions[:, :-1], state[:, :-1])
        td = (q_jt - targets) * mask
        loss = (td ** 2).sum() / mask.sum()
        self.critic_opt.zero_grad(set_to_none=True)
        (loss + getattr(mixer, "last_reg", 0.0)).backward()
        gn = nn.utils.clip_grad_norm_(self.critic_params, c.grad_norm_clip)
        self.critic_opt.step()
        return {f"critic_loss_{sfx}": loss.item(), f"q_jt_{sfx}": (q_jt * mask).sum().item() / mask.sum().item(),
                f"critic_grad_{sfx}": float(gn)}

    def _mix(self, mixer: nn.Module, q: torch.Tensor, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Joint soft Q of ``actions`` from the per-agent ``q`` ``(B, T, d, nA)``."""
        q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if isinstance(mixer, FOPMixer):
            a = self.alpha
            vs = a * torch.logsumexp(q / a, -1)
            return mixer(q_taken, state, F.one_hot(actions, self.n_actions).float(), vs)
        return mixer(q_taken, state)

    def _update_targets(self) -> None:
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer_ind.load_state_dict(self.mixer_ind.state_dict())
        if not self.cfg.shared_mixer:
            self.target_mixer_dep.load_state_dict(self.mixer_dep.state_dict())

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    mixer_ind=self.mixer_ind.state_dict(), mixer_dep=self.mixer_dep.state_dict(),
                    target_critic=self.target_critic.state_dict(),
                    target_mixer_ind=self.target_mixer_ind.state_dict(),
                    target_mixer_dep=self.target_mixer_dep.state_dict(),
                    actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    env_steps=self.env_steps, episodes=self.episodes, n_updates=self.n_updates,
                    cfg=asdict(self.cfg))

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for k in ("actor", "critic", "mixer_ind", "mixer_dep", "target_critic", "target_mixer_ind",
                  "target_mixer_dep", "actor_opt", "critic_opt"):
            getattr(self, k).load_state_dict(state[k])
        self.env_steps, self.episodes, self.n_updates = state["env_steps"], state["episodes"], state["n_updates"]

    def diagnostics(self) -> Dict[str, float]:
        n = sum(p.numel() for p in self.actor.parameters())
        out = {"n_params": float(n + sum(p.numel() for p in self.critic_params)), "n_actor_params": float(n),
               "env_steps": float(self.env_steps), "updates": float(self.n_updates), "epsilon": self.epsilon,
               "buffer_episodes": float(len(self.buffer)), "alpha": self.alpha}
        out.update({k: v for k, v in self._stats.items() if k.startswith(("entropy", "q_jt"))})
        return out

    def describe(self) -> str:
        c = self.cfg
        return (f"MACPFAgent(batch={c.batch_size} eps, td_lambda={c.td_lambda}, alpha={c.alpha_start}->{c.alpha_end}, "
                f"mixer={c.mixer}, shared_mixer={c.shared_mixer}, eval={c.eval_policy})")
