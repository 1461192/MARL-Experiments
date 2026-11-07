"""Ring-buffer replay of *consecutive* steps, so fixed-length trajectories (paper: 8) can be
sampled for recurrent / multi-step learning."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class SequenceReplay:
    """Stores per-step ``(obs, actions, team_reward)``. When an episode ends the final
    observation is stored as an extra entry, so a sampled window of ``W`` entries yields
    ``W-1`` full transitions plus a bootstrap observation."""

    def __init__(self, capacity: int, n_agents: int, obs_shape: Tuple[int, ...], seed: Optional[int] = None):
        self.cap = capacity
        self.obs = np.zeros((capacity, n_agents, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.ep = np.full(capacity, -1, dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self.ep_id = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _push(self, obs, act, rew) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.ep[i] = obs, act, rew, self.ep_id
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add(self, obs: np.ndarray, actions: np.ndarray, reward: float) -> None:
        self._push(obs, actions, reward)

    def end_episode(self, final_obs: np.ndarray) -> None:
        self._push(final_obs, 0, 0.0)
        self.ep_id += 1

    def sample(self, batch: int, window: int) -> Dict[str, np.ndarray]:
        """obs ``(B, W, n, *obs)``, actions ``(B, W-1, n)``, rewards ``(B, W-1)``."""
        hi = self.size - window + 1 if self.size < self.cap else self.cap
        if hi <= 0:
            raise RuntimeError("not enough data to sample")
        chosen = []
        for _ in range(100):
            s = self.rng.integers(0, hi, size=batch * 2)
            idx = (s[:, None] + np.arange(window)[None, :]) % self.cap
            ep = self.ep[idx]
            ok = (ep == ep[:, :1]).all(1) & (ep[:, 0] >= 0)
            if self.size == self.cap:                      # don't straddle newest -> oldest
                off = (self.ptr - s) % self.cap
                ok &= ~((off > 0) & (off < window))
            chosen.extend(idx[ok])
            if len(chosen) >= batch:
                break
        else:
            raise RuntimeError("could not sample valid windows (episodes shorter than the window?)")
        idx = np.stack(chosen[:batch])
        return dict(obs=self.obs[idx], actions=self.act[idx[:, :-1]], rewards=self.rew[idx[:, :-1]])
