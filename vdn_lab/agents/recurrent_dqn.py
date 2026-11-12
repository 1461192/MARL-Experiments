"""Shared learner for all DQN-style agents in the paper (sec. 2.2 and 4.1).

From the paper: experience replay, target network, LSTM Q-network, trajectories of length 8
for both BPTT and the multi-step forward view, eligibility-trace lambda = 0.9, dueling heads,
Adam with lr 1e-4. Everything the paper leaves unspecified (replay size, epsilon schedule,
batch size, target-update period, loss) lives in ``DQNConfig``.

Subclasses only decide *how per-agent / joint Q-values are combined* (``_build_net``,
``_chosen_q``, ``_max_q``, ``_greedy``) - exactly where independent learners, VDN and
centralised agents differ.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent
from .replay import SequenceReplay


def lambda_returns(rewards: torch.Tensor, boot: torch.Tensor, gamma: float,
                   lam: Union[float, torch.Tensor]) -> torch.Tensor:
    """Forward-view lambda-returns over a trajectory (no off-policy correction).

    rewards: ``(B, L, 1)``  r_t
    boot:    ``(B, L, K)``  max_a' Q_target(o_{t+1}, a')  (bootstrap value *after* step t)
    lam:     a float, or a tensor ``(B, L, 1)`` of per-step values (adaptive TD(lambda): ``lam[:, t]`` is
             ``omega(s_t, a_t)``, ATD paper eq. 10). ``lam[:, L-1]`` is never read.
    G_t = r_t + gamma * ((1 - lam_t) * boot_t + lam_t * G_{t+1}),   G_L := boot_{L-1}
    lam=0 -> 1-step TD target; lam=1 -> discounted n-step return bootstrapped at the end.
    """
    G = torch.empty_like(boot)
    nxt = boot[:, -1]
    per_step = torch.is_tensor(lam)
    for t in reversed(range(boot.shape[1])):
        l = lam[:, t] if per_step else lam
        nxt = rewards[:, t] + gamma * ((1 - l) * boot[:, t] + l * nxt)
        G[:, t] = nxt
    return G


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4              # paper
    hidden: int = 32              # paper: all layers have 32 units
    seq_len: int = 8              # paper: trajectory length (forward view and BPTT)
    lam: float = 0.9              # paper: eligibility-trace parameter
    burn_in: int = 0              # extra leading steps used only to warm up the LSTM state
    batch_size: int = 32
    buffer_size: int = 100_000    # in environment steps
    warmup_steps: int = 2_000
    train_every: int = 4          # environment steps between gradient steps
    target_update_every: int = 200  # gradient steps between hard target updates
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    eval_epsilon: float = 0.0     # epsilon used when act(explore=False)
    loss: str = "huber"           # "huber" | "mse"


class RecurrentDQNAgent(Agent):
    """Abstract multi-agent recurrent DQN with lambda-returns."""

    def __init__(self, n_agents: int, n_actions: int, obs_shape: Tuple[int, ...],
                 device: str = "cpu", seed: Optional[int] = None, **kwargs: Any):
        super().__init__(n_agents, n_actions, obs_shape, device, seed)
        self.cfg = DQNConfig(**kwargs)
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = int(np.prod(obs_shape))
        self.net: nn.Module = self._build_net().to(device)
        self.target: nn.Module = copy.deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.replay = SequenceReplay(self.cfg.buffer_size, n_agents, obs_shape, seed)
        self.hidden: Optional[torch.Tensor] = None
        self.env_steps = 0
        self.n_updates = 0

    # ------------------------------------------------------- subclass hooks
    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def _chosen_q(self, q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q-value(s) of the actions taken -> ``(B, T, K)`` (K independent regression targets)."""
        raise NotImplementedError

    def _max_q(self, q: torch.Tensor) -> torch.Tensor:
        """Greedy bootstrap value(s) -> ``(B, T, K)``."""
        raise NotImplementedError

    def _greedy(self, q_step: torch.Tensor) -> np.ndarray:
        """Greedy joint action from the network output of one time step."""
        raise NotImplementedError

    def _epsilon_greedy(self, greedy: np.ndarray, eps: float) -> np.ndarray:
        explore = self.rng.random(self.n_agents) < eps
        random = self.rng.integers(self.n_actions, size=self.n_agents)
        return np.where(explore, random, greedy)

    # ------------------------------------------------------------- helpers
    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def _to_input(self, obs: np.ndarray) -> torch.Tensor:
        """uint8 ``(..., d, C, H, W)`` -> float ``(..., d, C*H*W)`` in [0, 1]."""
        x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device).float().div_(255.0)
        return x.reshape(*x.shape[:-3], -1)

    # ------------------------------------------------------- Agent interface
    def reset_episode(self) -> None:
        self.hidden = self.net.initial_hidden(1, self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if self.hidden is None:
            self.reset_episode()
        q, self.hidden = self.net(self._to_input(obs[None, None]), self.hidden)
        greedy = self._greedy(q[0, 0])
        eps = self.epsilon if explore else self.cfg.eval_epsilon
        return self._epsilon_greedy(greedy, eps) if eps > 0 else greedy

    def observe(self, obs, actions, reward, next_obs, done) -> None:
        if not self.training:
            return
        self.replay.add(obs, actions, reward)
        self.env_steps += 1
        if done:
            self.replay.end_episode(next_obs)

    def update(self) -> Dict[str, float]:
        c = self.cfg
        if (not self.training or self.env_steps < c.warmup_steps
                or self.env_steps % c.train_every != 0
                or len(self.replay) < 2 * (c.burn_in + c.seq_len + 1)):
            return {}
        return self._learn()

    # ---------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        c = self.cfg
        L, bs = c.seq_len, c.burn_in
        batch = self.replay.sample(c.batch_size, bs + L + 1)
        obs = self._to_input(batch["obs"])                                  # (B, W, d, F)
        actions = torch.from_numpy(batch["actions"]).to(self.device)        # (B, W-1, d)
        rew = torch.from_numpy(batch["rewards"]).to(self.device)[:, bs:].unsqueeze(-1)  # (B, L, 1)
        chosen, boot = self._td_values(obs, actions)                         # (B, L, K) each
        with torch.no_grad():
            lam = self._trace_lambda(obs, actions, batch["idx"])
            G = lambda_returns(rew, boot, c.gamma, lam)
        loss_fn = F.smooth_l1_loss if c.loss == "huber" else F.mse_loss
        loss = loss_fn(chosen, G)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
        self.opt.step()
        self.n_updates += 1
        if self.n_updates % c.target_update_every == 0:
            self.target.load_state_dict(self.net.state_dict())
            self._on_target_update()
        return {"loss": float(loss.item()), "q_mean": float(chosen.mean().item()), "epsilon": self.epsilon}

    def _trace_lambda(self, obs: torch.Tensor, actions: torch.Tensor, idx: np.ndarray) -> Union[float, torch.Tensor]:
        """lambda for the returns of a sampled batch (``idx``: replay positions of the window): the fixed
        ``cfg.lam``, or (adaptive TD(lambda), ``atd.py``) a ``(B, L, 1)`` tensor as in ``lambda_returns``."""
        return self.cfg.lam

    def _on_target_update(self) -> None:
        """Called right after every hard target-network update."""

    def _td_values(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs ``(B, W, d, F)``, actions ``(B, W-1, d)`` -> (Q of the taken actions on the L
        learning steps, greedy bootstrap values after each of them), both ``(B, L, K)``.
        Agents whose joint value needs more than the per-step Q's (mixing networks) override this."""
        bs, L = self.cfg.burn_in, self.cfg.seq_len
        hid0 = self.net.initial_hidden(obs.shape[0], self.device)
        q, _ = self.net(obs, hid0)
        chosen = self._chosen_q(q[:, bs:bs + L], actions[:, bs:])
        with torch.no_grad():
            tq, _ = self.target(obs, hid0)
            boot = self._max_q(tq[:, bs + 1:])
        return chosen, boot

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return dict(net=self.net.state_dict(), target=self.target.state_dict(), opt=self.opt.state_dict(),
                    env_steps=self.env_steps, n_updates=self.n_updates, cfg=asdict(self.cfg))

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.target.load_state_dict(state["target"])
        self.opt.load_state_dict(state["opt"])
        self.env_steps, self.n_updates = state["env_steps"], state["n_updates"]

    def diagnostics(self) -> Dict[str, float]:
        return {"epsilon": self.epsilon,
                "n_params": float(sum(p.numel() for p in self.net.parameters())),
                "env_steps": float(self.env_steps), "updates": float(self.n_updates)}
