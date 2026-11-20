"""Level-Based Foraging tasks (Papoudakis et al. 2021) on top of ``lbforaging``.

The environment itself is the reference implementation, ``lbforaging`` (semitable/lb-foraging,
``pip install lbforaging``). This module only

* builds the tasks by name (``TASKS``),
* returns observations as float32 arrays ``(n_agents, obs_dim)`` plus a global state (all agents'
  observations concatenated, as EPyMARL's gym wrapper does for LBF),
* turns the per-agent rewards into one **common reward** (their sum, EPyMARL's default
  ``common_reward=True, reward_scalarisation="sum"``); the rewards are normalised by lbforaging so the
  team return of an episode in which every item is collected is 1,
* restores the item-level rule of lbforaging v1 (the version the benchmark used, task ids ``Foraging-...-v1``,
  see ``V1ItemLevels``),
* separates *termination* (every item collected) from *truncation* (time limit). lbforaging reports both
  as ``done``; the learners bootstrap on truncation but not on termination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from lbforaging.foraging.environment import ForagingEnv


class V1ItemLevels(ForagingEnv):
    """``ForagingEnv`` (lbforaging 2.0) with the item levels of lbforaging 1.0.x, which the benchmark used.

    Both versions draw agent levels from {1, 2} and bound an item's level by S, the sum of the three lowest
    agent levels. v1 draws a non-cooperative item's level with ``randint(1, S)`` - numpy's exclusive upper
    bound, so 1..S-1 - while v2 draws it from 1..S inclusive. With three agents that takes the share of items
    that need *all* agents from about 2% (v1) to about 25% (v2). Cooperative (``force_coop``) items are S in
    both versions. Everything else (placement, dynamics, rewards, observations) is lbforaging 2.0's code.
    """

    def spawn_food(self, max_num_food, min_levels, max_levels):
        if not self.force_coop and self.max_food_level is None:
            max_levels = np.maximum(np.asarray(max_levels) - 1, np.asarray(min_levels))
        return super().spawn_food(max_num_food, min_levels, max_levels)


@dataclass(frozen=True)
class TaskSpec:
    size: int
    players: int
    foods: int
    coop: bool = False
    sight: Optional[int] = None          # None -> full observability (sight = grid size)
    max_steps: int = 50

    def kwargs(self) -> dict:
        return dict(players=self.players, min_player_level=1, max_player_level=2,
                    min_food_level=1, max_food_level=None, field_size=(self.size, self.size),
                    max_num_food=self.foods, sight=self.sight or self.size,
                    max_episode_steps=self.max_steps, force_coop=self.coop,
                    normalize_reward=True, grid_observation=False, penalty=0.0)


# The seven LBF tasks of Papoudakis et al. (NeurIPS 2021 Datasets & Benchmarks, "Benchmarking Multi-Agent Deep
# Reinforcement Learning Algorithms in Cooperative Tasks", sec. 3.4 / table 3): "-2s" = sight 2 (5x5 view),
# "-c" = cooperative (every item needs all agents). Time limit 50 steps (appendix A: env_args.time_limit=50).
TASKS: Dict[str, TaskSpec] = {
    "8x8-2p-2f-c": TaskSpec(8, 2, 2, coop=True),
    "8x8-2p-2f-2s-c": TaskSpec(8, 2, 2, coop=True, sight=2),
    "10x10-3p-3f": TaskSpec(10, 3, 3),
    "10x10-3p-3f-2s": TaskSpec(10, 3, 3, sight=2),
    "15x15-3p-5f": TaskSpec(15, 3, 5),
    "15x15-4p-3f": TaskSpec(15, 4, 3),
    "15x15-4p-5f": TaskSpec(15, 4, 5),
}


class LBFEnv:
    """One LBF environment with a common reward and a terminated / truncated split."""

    def __init__(self, task: str | TaskSpec, seed: Optional[int] = None):
        self.spec = TASKS[task] if isinstance(task, str) else task
        self.env = V1ItemLevels(**self.spec.kwargs())
        self.n_agents = self.spec.players
        self.n_actions = 6
        self.obs_dim = int(self.env.observation_space[0].shape[0])
        self.state_dim = self.n_agents * self.obs_dim
        self.episode_limit = self.spec.max_steps
        self._seed = seed
        self._first = True

    def reset(self) -> np.ndarray:
        if self._first:
            obs, _ = self.env.reset(seed=self._seed)
            self._first = False
        else:
            obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool]:
        obs, rewards, done, _, _ = self.env.step([int(a) for a in actions])
        terminated = bool(self.env.field.sum() == 0)
        truncated = bool(done and not terminated)
        return np.asarray(obs, dtype=np.float32), float(sum(rewards)), terminated, truncated

    @staticmethod
    def state(obs: np.ndarray) -> np.ndarray:
        return obs.reshape(*obs.shape[:-2], -1)


def make_envs(task: str, n: int, seed: int) -> List[LBFEnv]:
    return [LBFEnv(task, seed=seed * 1000 + i) for i in range(n)]
