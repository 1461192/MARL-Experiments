"""Episode storage and the rollout collector (EPyMARL's episode / parallel runners).

An episode batch is a dict of arrays, padded to the time limit ``T``:

* ``obs``     ``(E, T+1, n, obs_dim)``  observation before step t; ``obs[:, L]`` is the final one
* ``actions`` ``(E, T+1, n)``           action at step t; ``actions[:, L]`` is sampled at the final
                                        observation but never executed (COMA bootstraps with it)
* ``reward``  ``(E, T)``  common reward;  ``term`` ``(E, T)`` 1 at a terminal (not time-limit) step
* ``filled``  ``(E, T)``  1 for real transitions;  ``length`` ``(E,)``
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .env import LBFEnv

Batch = Dict[str, np.ndarray]


def empty_batch(E: int, T: int, n: int, od: int) -> Batch:
    return dict(obs=np.zeros((E, T + 1, n, od), np.float32), actions=np.zeros((E, T + 1, n), np.int64),
                reward=np.zeros((E, T), np.float32), term=np.zeros((E, T), np.float32),
                filled=np.zeros((E, T), np.float32), length=np.zeros(E, np.int64))


def trim(batch: Batch) -> Batch:
    """Cut the padding to the longest episode in the batch."""
    L = int(batch["length"].max())
    return dict(obs=batch["obs"][:, :L + 1], actions=batch["actions"][:, :L + 1], reward=batch["reward"][:, :L],
                term=batch["term"][:, :L], filled=batch["filled"][:, :L], length=batch["length"])


class Collector:
    """Runs one episode in each of its environments, all agents acting through ``algo.act``."""

    def __init__(self, envs: List[LBFEnv]):
        self.envs = envs
        e = envs[0]
        self.n, self.od, self.T = e.n_agents, e.obs_dim, e.episode_limit

    def run(self, algo, explore: bool, t_env: int) -> Batch:
        E, T = len(self.envs), self.T
        b = empty_batch(E, T, self.n, self.od)
        cur = np.stack([env.reset() for env in self.envs])
        b["obs"][:, 0] = cur
        algo.init_hidden(E)
        active = np.ones(E, bool)
        final_pending = np.zeros(E, bool)
        t = 0
        while active.any() or final_pending.any():
            acts = algo.act(cur, explore=explore, t_env=t_env)
            for e in np.flatnonzero(final_pending):
                b["actions"][e, b["length"][e]] = acts[e]
                final_pending[e] = False
            for e in np.flatnonzero(active):
                b["actions"][e, t] = acts[e]
                nxt, r, term, trunc = self.envs[e].step(acts[e])
                b["reward"][e, t], b["term"][e, t], b["filled"][e, t] = r, float(term), 1.0
                b["obs"][e, t + 1] = nxt
                cur[e] = nxt
                if term or trunc:
                    active[e] = False
                    final_pending[e] = True
                    b["length"][e] = t + 1
            t += 1
        return b


class EpisodeBuffer:
    """Ring buffer of whole episodes (EPyMARL ``ReplayBuffer``, ``buffer_size`` in episodes)."""

    def __init__(self, capacity: int, T: int, n: int, od: int, rng: np.random.Generator):
        self.cap, self.T = capacity, T
        self.data = empty_batch(capacity, T, n, od)
        self.ptr = 0
        self.size = 0
        self.rng = rng
        self.version = np.zeros(capacity, np.int64)     # bumped whenever a slot is overwritten

    def insert(self, batch: Batch) -> np.ndarray:
        E = batch["length"].shape[0]
        idx = (self.ptr + np.arange(E)) % self.cap
        for k, v in batch.items():
            self.data[k][idx] = v
        self.version[idx] += 1
        self.ptr = (self.ptr + E) % self.cap
        self.size = min(self.size + E, self.cap)
        return idx

    def can_sample(self, n: int) -> bool:
        return self.size >= n

    def sample_idx(self, n: int, recent: Optional[int] = None, skip_recent: int = 0) -> np.ndarray:
        """Uniform over the stored episodes: all of them, only the ``recent`` newest ones, or all but the
        ``skip_recent`` newest ones."""
        if recent is not None:
            k, off = min(recent, self.size), 0
        else:
            k, off = self.size - skip_recent, skip_recent
        if k <= 0:
            raise RuntimeError("not enough episodes to sample")
        back = self.rng.choice(k, n, replace=False) if n <= k else self.rng.integers(0, k, n)
        return (self.ptr - 1 - off - back) % self.cap

    def get(self, idx: np.ndarray) -> Batch:
        return trim({k: v[idx] for k, v in self.data.items()})
