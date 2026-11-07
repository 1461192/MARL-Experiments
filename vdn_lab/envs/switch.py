"""Switch (paper sec. 4.2).

Two agents start at opposite ends of a map and must each reach their own goal at the
other end. If they meet in a narrow corridor one has to back out and let the other pass.
Team reward: +1 whenever a player reaches its goal. When both have reached their goals
the whole environment resets.

Legend: ``#`` wall, ``A``/``B`` start of agent 0/1, ``a``/``b`` goal of agent 0/1.
A goal is only visible to its own agent (as in the paper) -> strong observation aliasing.
"""
from __future__ import annotations

from typing import List, Tuple

from ..registry import register_env
from .base import GridWorld, Pos, Color, C_GREEN

LAYOUTS = {
    "open": [
        "###########",
        "#.........#",
        "#A.......B#",
        "#.........#",
        "#b.......a#",
        "###########",
    ],
    # two rooms joined by a single one-square-wide corridor
    "single": [
        "#############",
        "#b...###...a#",
        "#A.........B#",
        "#....###....#",
        "#############",
    ],
    # two parallel corridors (agents can pass each other if they coordinate)
    "double": [
        "#############",
        "#b.........a#",
        "#A...#####.B#",
        "#...........#",
        "#############",
    ],
}


@register_env("switch")
class Switch(GridWorld):
    LAYOUTS = LAYOUTS
    default_layout_name = "single"

    def __init__(self, goal_reward: float = 1.0, freeze_at_goal: bool = True, **kwargs):
        self.goal_reward = goal_reward
        self.freeze_at_goal = freeze_at_goal
        self.goals: List[Pos] = [None, None]  # type: ignore[list-item]
        super().__init__(**kwargs)

    def _parse_cell(self, ch: str, pos: Pos) -> None:
        if ch in "ab":
            self.goals["ab".index(ch)] = pos
        else:
            super()._parse_cell(ch, pos)

    def _post_parse(self) -> None:
        if any(g is None for g in self.goals):
            raise ValueError("Switch layout needs goals 'a' and 'b'")

    def _reset_task(self) -> None:
        self.reached = [False, False]

    def _frozen(self, i: int) -> bool:
        return self.freeze_at_goal and self.reached[i]

    def _apply_rules(self) -> List[float]:
        out = [0.0, 0.0]
        for i in range(2):
            if not self.reached[i] and self.pos[i] == self.goals[i]:
                self.reached[i] = True
                out[i] = self.goal_reward
        if all(self.reached):                      # full reset, task repeats
            self._reset_positions()
            self.reached = [False, False]
        return out

    def _object_overlays(self, i: int) -> List[Tuple[Pos, Color]]:
        return [(self.goals[i], C_GREEN)]

    def _global_overlays(self) -> List[Tuple[Pos, Color]]:
        return [(self.goals[0], C_GREEN), (self.goals[1], (0, 120, 60))]

    def _global_chars(self):
        return {self.goals[0]: "a", self.goals[1]: "b"}
