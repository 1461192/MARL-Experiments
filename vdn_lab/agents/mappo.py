"""Multi-Agent PPO (MAPPO) and independent PPO (IPPO).

Learner: PPO with the clipped surrogate objective, Schulman et al. 2017 ("Proximal Policy
Optimization Algorithms", arXiv:1707.06347), eq. 7 plus an entropy bonus (eq. 9) and GAE
advantages (eq. 11-12), trained for K epochs on each freshly collected batch.

Multi-agent setup and implementation choices: Yu et al. 2022 ("The Surprising Effectiveness of
PPO in Cooperative Multi-Agent Games", arXiv:2103.01955), especially section 5:

* **Centralised value function, decentralised policies** (CTDE). Each agent's policy sees only
  its own observation history; the critic sees more (``critic_input``) and is only used for
  training. ``critic_input="local"`` gives IPPO (registered as ``ippo``).
* **5.1 value normalisation.** The critic regresses onto running-mean/std normalised returns
  (``ValueNorm``: debiased EMA, as in the authors' code); values are de-normalised for GAE.
* **5.2 input representation of the value function.** The testbed's agents only receive
  observations, so the "global state" is the union of the agents' views. The default is the
  paper's *agent-specific* input: the agent's own observation first, then the other agents'
  observations, then a one-hot agent id (``"agent_specific"``). ``"concat"`` is the paper's CL
  (the same concatenation for every agent).
* **5.3 training data usage.** Few epochs (default 10; the paper uses 15 on easy, 5-10 on hard
  tasks) and no mini-batch splitting (``num_mini_batch=1``).
* **5.4 PPO clipping.** ``clip=0.2`` (the paper recommends staying at or below 0.2); the value
  prediction is clipped with the same range.
* **5.5 batch size.** One update per ``rollout_len`` environment steps (default 1000 steps =
  5 of the 200-step episodes used in the examples; x2 agents = 2000 samples per update).

Other details follow the paper's appendix / code: parameter sharing across agents
(``share_weights``), GRU policy and critic trained on chunks of ``chunk_len`` = 10 steps whose
initial hidden state is the one recorded during collection, input LayerNorm, orthogonal
initialisation, Huber value loss (delta 10), advantage normalisation, Adam (lr 5e-4, eps 1e-5),
gradient-norm clipping 10, entropy coefficient 0.01, gamma 0.99, GAE lambda 0.95.

Episodes in the grid worlds end on a time limit only, so by default the last step of an
episode bootstraps from ``V(next_obs)`` (``bootstrap_on_done``) instead of treating it as
terminal. Evaluation (``act(explore=False)``) takes the most likely action of each policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_agent
from .base import Agent


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 5e-4                  # actor learning rate
    critic_lr: Optional[float] = None  # None -> same as lr
    adam_eps: float = 1e-5
    hidden: int = 64
    recurrent: bool = True            # GRU policy / critic (else MLP only)
    chunk_len: int = 10               # BPTT data-chunk length (paper: 10)
    rollout_len: int = 1000           # environment steps per PPO update
    ppo_epochs: int = 10
    num_mini_batch: int = 1
    clip: float = 0.2
    value_clip: Optional[float] = 0.2  # None -> no value clipping
    value_norm: bool = True
    huber_delta: Optional[float] = 10.0  # None -> MSE value loss
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 10.0
    normalize_advantages: bool = True
    layer_norm: bool = True           # LayerNorm on the network inputs and after the GRU
    share_weights: bool = True        # one policy (and one critic) for all agents
    role_info: bool = False           # one-hot agent id appended to the *policy* input
    critic_input: str = "agent_specific"  # "agent_specific" | "concat" | "local"
    bootstrap_on_done: bool = True    # episodes end on a time limit, not in a terminal state
    deterministic_eval: bool = True   # act(explore=False) -> argmax of the policy


# ------------------------------------------------------------------------------- networks
def _init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class RecurrentNet(nn.Module):
    """[LayerNorm] -> Linear -> ReLU -> Linear -> ReLU -> [GRU -> LayerNorm] -> Linear(out).

    ``forward(x (T, N, in), h0 (N, H), starts (T, N))``: the hidden state is zeroed before
    step t wherever ``starts[t]`` is set (an episode began there) except at t = 0, where
    ``h0`` is already the state to use."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, recurrent: bool, layer_norm: bool, out_gain: float):
        super().__init__()
        g = np.sqrt(2)
        self.hidden, self.recurrent = hidden, recurrent
        self.in_norm = nn.LayerNorm(in_dim) if layer_norm else nn.Identity()
        self.mlp = nn.Sequential(_init(nn.Linear(in_dim, hidden), g), nn.ReLU(),
                                 _init(nn.Linear(hidden, hidden), g), nn.ReLU())
        if recurrent:
            self.gru = nn.GRUCell(hidden, hidden)
            for name, p in self.gru.named_parameters():
                nn.init.orthogonal_(p) if "weight" in name else nn.init.zeros_(p)
            self.out_norm = nn.LayerNorm(hidden) if layer_norm else nn.Identity()
        self.head = _init(nn.Linear(hidden, out_dim), out_gain)

    def forward(self, x: torch.Tensor, h: torch.Tensor, starts: Optional[torch.Tensor] = None):
        z = self.mlp(self.in_norm(x))                                   # (T, N, H)
        if not self.recurrent:
            return self.head(z), h
        outs = []
        for t in range(z.shape[0]):
            if t > 0 and starts is not None:
                h = h * (1.0 - starts[t]).unsqueeze(-1)
            h = self.gru(z[t], h)
            outs.append(h)
        return self.head(self.out_norm(torch.stack(outs))), h


class AgentNets(nn.Module):
    """One ``RecurrentNet`` per agent, or one shared by all agents (agents become batch items).
    ``forward(x (T, B, d, in), h (B, d, H), starts (T, B))`` -> out (T, B, d, out), h."""

    def __init__(self, n_agents: int, in_dim: int, out_dim: int, cfg: PPOConfig, out_gain: float):
        super().__init__()
        self.n_agents, self.hidden, self.share = n_agents, cfg.hidden, cfg.share_weights
        n = 1 if self.share else n_agents
        self.nets = nn.ModuleList(RecurrentNet(in_dim, out_dim, cfg.hidden, cfg.recurrent, cfg.layer_norm, out_gain)
                                  for _ in range(n))

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.n_agents, self.hidden, device=device)

    def forward(self, x: torch.Tensor, h: torch.Tensor, starts: Optional[torch.Tensor] = None):
        T, B, d, _ = x.shape
        if self.share:
            s = None if starts is None else starts.repeat_interleave(d, dim=1)
            out, h = self.nets[0](x.reshape(T, B * d, -1), h.reshape(B * d, -1), s)
            return out.reshape(T, B, d, -1), h.reshape(B, d, -1)
        outs, hs = [], []
        for i, net in enumerate(self.nets):
            o, hi = net(x[:, :, i], h[:, i], starts)
            outs.append(o)
            hs.append(hi)
        return torch.stack(outs, 2), torch.stack(hs, 1)


class ValueNorm(nn.Module):
    """Running mean / std of the value targets (debiased EMA, as ``ValueNorm`` in the MAPPO code)."""

    def __init__(self, beta: float = 0.99999, eps: float = 1e-5):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("mean_sq", torch.zeros(()))
        self.register_buffer("debias", torch.zeros(()))

    def stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.debias.clamp(min=self.eps)
        mean, mean_sq = self.mean / d, self.mean_sq / d
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2).sqrt()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        w = self.beta ** x.numel()                       # one EMA step per element
        self.mean.mul_(w).add_(x.mean() * (1 - w))
        self.mean_sq.mul_(w).add_((x ** 2).mean() * (1 - w))
        self.debias.mul_(w).add_(1 - w)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self.stats()
        return (x - m) / s

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self.stats()
        return x * s + m


def gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, dones: np.ndarray,
        gamma: float, lam: float) -> np.ndarray:
    """Generalised advantage estimation over a rollout (PPO paper eq. 11-12).

    rewards ``(T,)``, values ``(T, d)``: V(o_t). ``next_values[t]`` is V(o_{t+1}) and is only read
    where the episode ended at t (``dones[t]``) or at the last step of the rollout; elsewhere
    V(o_{t+1}) = ``values[t+1]``. The advantage does not run on across an episode end."""
    T = len(rewards)
    adv = np.zeros_like(values)
    last = np.zeros(values.shape[1:], dtype=values.dtype)
    for t in reversed(range(T)):
        cut = dones[t] or t == T - 1
        nv = next_values[t] if cut else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + (0.0 if cut else gamma * lam) * last
        adv[t] = last
    return adv


# --------------------------------------------------------------------------------- agent
@register_agent("mappo")
class MAPPOAgent(Agent):
    """MAPPO: PPO policies per agent (decentralised execution) + a centralised critic.

    Agent kwargs are the ``PPOConfig`` fields (``critic_input``, ``share_weights``,
    ``role_info``, ``ppo_epochs``, ``rollout_len``, ``clip``, ...)."""

    DEFAULTS: Dict[str, Any] = {}

    def __init__(self, n_agents: int, n_actions: int, obs_shape: Tuple[int, ...],
                 device: str = "cpu", seed: Optional[int] = None, **kwargs: Any):
        super().__init__(n_agents, n_actions, obs_shape, device, seed)
        self.cfg = c = PPOConfig(**{**self.DEFAULTS, **kwargs})
        if c.critic_input not in ("agent_specific", "concat", "local"):
            raise ValueError(f"critic_input must be agent_specific | concat | local, got {c.critic_input!r}")
        if c.rollout_len % c.chunk_len:
            raise ValueError("rollout_len must be a multiple of chunk_len")
        if not 1 <= c.num_mini_batch <= c.rollout_len // c.chunk_len:
            raise ValueError("num_mini_batch must be between 1 and rollout_len / chunk_len")
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = int(np.prod(obs_shape))
        d = n_agents
        pol_in = self.obs_dim + (d if c.role_info else 0)
        crit_in = {"agent_specific": d * self.obs_dim + d, "concat": d * self.obs_dim,
                   "local": self.obs_dim + (d if c.role_info else 0)}[c.critic_input]
        self.actor = AgentNets(d, pol_in, n_actions, c, out_gain=0.01).to(device)
        self.critic = AgentNets(d, crit_in, 1, c, out_gain=1.0).to(device)
        self.value_norm = ValueNorm().to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.lr, eps=c.adam_eps)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr or c.lr, eps=c.adam_eps)
        # orderings for the agent-specific critic input: agent i sees [o_i, o_others..., id_i]
        self._order = torch.tensor([[i] + [j for j in range(d) if j != i] for i in range(d)], device=device)
        self._eye = torch.eye(d, device=device)

        T, H = c.rollout_len, c.hidden
        self.buf = dict(
            obs=np.zeros((T, d, *obs_shape), np.uint8), actions=np.zeros((T, d), np.int64),
            logp=np.zeros((T, d), np.float32), values=np.zeros((T, d), np.float32),
            next_values=np.zeros((T, d), np.float32), rewards=np.zeros(T, np.float32),
            dones=np.zeros(T, bool), starts=np.zeros(T, np.float32),
            actor_h=np.zeros((T, d, H), np.float32), critic_h=np.zeros((T, d, H), np.float32))
        self.ptr = 0
        self.actor_h: Optional[torch.Tensor] = None
        self.critic_h: Optional[torch.Tensor] = None
        self._episode_start = True
        self._pending: Optional[Dict[str, np.ndarray]] = None
        self.env_steps = 0
        self.n_updates = 0
        self._last_entropy = float("nan")

    # ---------------------------------------------------------------------- inputs
    def _to_input(self, obs: np.ndarray) -> torch.Tensor:
        """uint8 ``(..., d, *obs_shape)`` -> float ``(..., d, obs_dim)`` in [0, 1]."""
        x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device).float().div_(255.0)
        return x.reshape(*x.shape[:x.dim() - len(self.obs_shape)], -1)

    def _with_id(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self._eye.expand(*x.shape[:-1], self.n_agents)], -1)

    def _policy_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._with_id(x) if self.cfg.role_info else x

    def _critic_input(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., d, F)`` -> ``(..., d, critic_in)``."""
        mode, d = self.cfg.critic_input, self.n_agents
        if mode == "local":
            return self._policy_input(x)
        if mode == "concat":
            return x.flatten(-2).unsqueeze(-2).expand(*x.shape[:-2], d, d * x.shape[-1])
        per_agent = x[..., self._order, :]                               # (..., d, d, F): own obs first
        return self._with_id(per_agent.flatten(-2))

    # ------------------------------------------------------------ Agent interface
    def reset_episode(self) -> None:
        self.actor_h = self.actor.initial_hidden(1, self.device)
        self.critic_h = self.critic.initial_hidden(1, self.device)
        self._episode_start = True
        self._pending = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if self.actor_h is None:
            self.reset_episode()
        x = self._to_input(obs[None, None])                              # (1, 1, d, F)
        ah0 = self.actor_h
        logits, self.actor_h = self.actor(self._policy_input(x), ah0)
        logits = logits[0, 0]                                            # (d, nA)
        if explore or not self.cfg.deterministic_eval:
            actions = torch.distributions.Categorical(logits=logits).sample()
        else:
            actions = logits.argmax(-1)
        if self.training and explore:
            ch0 = self.critic_h
            v, self.critic_h = self.critic(self._critic_input(x), ch0)
            self._pending = dict(obs=obs, actions=actions.cpu().numpy(),
                                 logp=torch.log_softmax(logits, -1).gather(-1, actions[:, None])[:, 0].cpu().numpy(),
                                 values=v[0, 0, :, 0].cpu().numpy(),
                                 actor_h=ah0[0].cpu().numpy(), critic_h=ch0[0].cpu().numpy())
        return actions.cpu().numpy()

    @torch.no_grad()
    def _bootstrap(self, next_obs: np.ndarray) -> np.ndarray:
        """De-normalised V(next_obs), continuing the critic's recurrent state."""
        v, _ = self.critic(self._critic_input(self._to_input(next_obs[None, None])), self.critic_h)
        return self._denorm(v[0, 0, :, 0]).cpu().numpy()

    def _denorm(self, v: torch.Tensor) -> torch.Tensor:
        return self.value_norm.denormalize(v) if self.cfg.value_norm else v

    def observe(self, obs, actions, reward, next_obs, done) -> None:
        if not self.training or self._pending is None:
            return
        p, b, t = self._pending, self.buf, self.ptr
        if not np.array_equal(np.asarray(actions), p["actions"]):
            raise RuntimeError("MAPPO.observe got actions that differ from the ones act() sampled")
        for k in ("obs", "actions", "logp", "values", "actor_h", "critic_h"):
            b[k][t] = p[k]
        b["rewards"][t], b["dones"][t], b["starts"][t] = reward, done, float(self._episode_start)
        if done and not self.cfg.bootstrap_on_done:
            b["next_values"][t] = 0.0
        elif done or t == self.cfg.rollout_len - 1:
            b["next_values"][t] = self._bootstrap(next_obs)
        self._pending = None
        self._episode_start = False
        self.ptr += 1
        self.env_steps += 1

    def update(self) -> Dict[str, float]:
        if not self.training or self.ptr < self.cfg.rollout_len:
            return {}
        metrics = self._learn()
        self.ptr = 0
        return metrics

    # ------------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        c, b, dev = self.cfg, self.buf, self.device
        T, L, d = c.rollout_len, c.chunk_len, self.n_agents
        vn = self.value_norm
        # value predictions were stored on the normalised scale of collection time
        old_v = self._denorm(torch.from_numpy(b["values"]).to(dev))
        adv = gae(b["rewards"], old_v.cpu().numpy(), b["next_values"], b["dones"], c.gamma, c.gae_lambda)
        adv_t = torch.from_numpy(adv).to(dev)
        returns = adv_t + old_v
        if c.value_norm:
            vn.update(returns)
            targets, old_v_n = vn.normalize(returns), vn.normalize(old_v)
        else:
            targets, old_v_n = returns, old_v
        if c.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-5)

        # chunk the rollout: (T, ...) -> (L, n_chunks, ...)
        nc = T // L

        def chunk(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(nc, L, *x.shape[1:]).transpose(0, 1)

        x = self._to_input(b["obs"])                                     # (T, d, F)
        pol_x, crit_x = chunk(self._policy_input(x)), chunk(self._critic_input(x))
        acts = chunk(torch.from_numpy(b["actions"]).to(dev))
        old_logp = chunk(torch.from_numpy(b["logp"]).to(dev))
        adv_c, tgt_c, oldv_c = chunk(adv_t), chunk(targets), chunk(old_v_n)
        starts = chunk(torch.from_numpy(b["starts"]).to(dev))
        ah0 = torch.from_numpy(b["actor_h"][::L]).to(dev)                # (nc, d, H)
        ch0 = torch.from_numpy(b["critic_h"][::L]).to(dev)

        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
        for _ in range(c.ppo_epochs):
            for mb in np.array_split(self.rng.permutation(nc), c.num_mini_batch):
                mb = torch.from_numpy(mb).to(dev)
                logits, _ = self.actor(pol_x[:, mb], ah0[mb], starts[:, mb])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(-1, acts[:, mb].unsqueeze(-1)).squeeze(-1)
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                log_ratio = logp - old_logp[:, mb]
                ratio = log_ratio.exp()
                a = adv_c[:, mb]
                surr = torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a)
                policy_loss = -surr.mean()
                self.actor_opt.zero_grad(set_to_none=True)
                (policy_loss - c.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.actor_opt.step()

                v, _ = self.critic(crit_x[:, mb], ch0[mb], starts[:, mb])
                v = v.squeeze(-1)
                value_loss = self._value_loss(v, oldv_c[:, mb], tgt_c[:, mb])
                self.critic_opt.zero_grad(set_to_none=True)
                (c.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > c.clip).float().mean().item())
        self.n_updates += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        self._last_entropy = out["entropy"]
        out["loss"] = out["policy_loss"] + c.value_loss_coef * out["value_loss"]
        out["return_mean"] = float(returns.mean().item())
        return out

    def _value_loss(self, v: torch.Tensor, old_v: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = self.cfg

        def err(pred):
            if c.huber_delta:
                return F.huber_loss(pred, target, reduction="none", delta=c.huber_delta)
            return (target - pred) ** 2

        loss = err(v)
        if c.value_clip is not None:
            v_clipped = old_v + (v - old_v).clamp(-c.value_clip, c.value_clip)
            loss = torch.max(loss, err(v_clipped))
        return loss.mean()

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    value_norm=self.value_norm.state_dict(), actor_opt=self.actor_opt.state_dict(),
                    critic_opt=self.critic_opt.state_dict(), env_steps=self.env_steps,
                    n_updates=self.n_updates, cfg=asdict(self.cfg))

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.value_norm.load_state_dict(state["value_norm"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.env_steps, self.n_updates = state["env_steps"], state["n_updates"]
        self.ptr = 0

    def diagnostics(self) -> Dict[str, float]:
        n = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        return {"n_params": float(n), "n_actor_params": float(sum(p.numel() for p in self.actor.parameters())),
                "env_steps": float(self.env_steps), "updates": float(self.n_updates), "entropy": self._last_entropy}

    def describe(self) -> str:
        c = self.cfg
        return (f"{type(self).__name__}(critic={c.critic_input}, share={c.share_weights}, role={c.role_info}, "
                f"epochs={c.ppo_epochs}, rollout={c.rollout_len}, clip={c.clip})")


@register_agent("ippo")
class IPPOAgent(MAPPOAgent):
    """Independent PPO: same learner, but each agent's critic sees only its own observation
    history (``critic_input="local"``), as the IPPO baseline in Yu et al. 2022."""

    DEFAULTS = {"critic_input": "local"}
