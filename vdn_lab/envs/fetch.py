"""Fetch (paper sec. 4.2).

Both players start on the same side; a single pick-up point is on the opposite side and a
drop-off point is next to the start. Picking the item up scores +3 for the team, dropping
it at the base +5; afterwards the pick-up is available again to either player. Optimal:
cycle so that when one player reaches the pick-up the other returns to base.

Legend: ``#`` wall, ``A``/``B`` starts, ``P`` pick-up, ``D`` drop-off.
The other player is drawn in *lighter blue* (paper). With ``show_carrying`` the agent's
own square turns cyan while it holds the item (set False for strictly pixel-only
observations, leaving this to the LSTM).
"""
from __future__ import annotations

from typing import List, Tuple

from ..registry import register_env
from .base import GridWorld, Pos, Color, C_MAGENTA, C_GREEN, C_LIGHT_BLUE, C_SELF, C_CYAN

# Pick-up (P) and drop-off (D) are in the interior so that circling the perimeter does not
# solve the task by accident.
LAYOUTS = {
    "open": [
        "###########",
        "#A........#",
        "#.D....P..#",
        "#B........#",
        "###########",
    ],
    # two rooms joined by one corridor: the agents must take turns going through it
    "single": [
        "#############",
        "#...#####...#",
        "#A..#####...#",
        "#.D.......P.#",
        "#B..#####...#",
        "#...#####...#",
        "#############",
    ],
    # two parallel corridors (top and bottom)
    "double": [
        "#############",
        "#...........#",
        "#A..#####...#",
        "#.D.#####.P.#",
        "#B..#####...#",
        "#...........#",
        "#############",
    ],
}


@register_env("fetch")
class Fetch(GridWorld):
    LAYOUTS = LAYOUTS
    default_layout_name = "single"

    def __init__(self, pickup_reward: float = 3.0, dropoff_reward: float = 5.0,
                 show_carrying: bool = True, **kwargs):
        self.pickup_reward = pickup_reward
        self.dropoff_reward = dropoff_reward
        self.show_carrying = show_carrying
        self.pickup: Pos = None  # type: ignore[assignment]
        self.drop: Pos = None    # type: ignore[assignment]
        super().__init__(**kwargs)

    def _parse_cell(self, ch: str, pos: Pos) -> None:
        if ch == "P":
            self.pickup = pos
        elif ch == "D":
            self.drop = pos
        else:
            super()._parse_cell(ch, pos)

    def _post_parse(self) -> None:
        if self.pickup is None or self.drop is None:
            raise ValueError("Fetch layout needs 'P' (pick-up) and 'D' (drop-off)")

    def _reset_task(self) -> None:
        self.item_available = True
        self.carrying = [False, False]

    def _apply_rules(self) -> List[float]:
        out = [0.0, 0.0]
        for i in range(2):                                  # drop-offs first
            if self.carrying[i] and self.pos[i] == self.drop:
                self.carrying[i] = False
                self.item_available = True
                out[i] += self.dropoff_reward
        for i in self.rng.permutation(2):                   # then pick-ups
            if self.item_available and not self.carrying[i] and self.pos[i] == self.pickup:
                self.carrying[i] = True
                self.item_available = False
                out[i] += self.pickup_reward
        return out

    def _self_color(self, i: int) -> Color:
        return C_CYAN if (self.show_carrying and self.carrying[i]) else C_SELF

    def _other_color(self, viewer: int, other: int) -> Color:
        return C_LIGHT_BLUE

    def _agent_global_color(self, i: int) -> Color:
        return C_CYAN if self.carrying[i] else (C_SELF if i == 0 else C_LIGHT_BLUE)

    def _object_overlays(self, i: int) -> List[Tuple[Pos, Color]]:
        out = [(self.drop, C_GREEN)]
        if self.item_available:
            out.append((self.pickup, C_MAGENTA))
        return out

    def _global_overlays(self) -> List[Tuple[Pos, Color]]:
        return self._object_overlays(0)

    def scene_items(self):
        out = [(self.drop, C_GREEN, "D")]
        if self.item_available:
            out.append((self.pickup, C_MAGENTA, "P"))
        else:
            out.append((self.pickup, (235, 200, 235), "p"))
        return out

    def scene_legend(self):
        return [(C_MAGENTA, f"P pick-up (+{self.pickup_reward:g})"),
                (C_GREEN, f"D drop-off (+{self.dropoff_reward:g})"),
                ((235, 200, 235), "p pick-up taken")]

    def agent_status(self, i: int) -> str:
        return "carrying item" if self.carrying[i] else "empty-handed"

    def scene_carrying(self, i: int) -> bool:
        return bool(self.carrying[i])

    def _global_chars(self):
        d = {self.drop: "D"}
        if self.item_available:
            d[self.pickup] = "P"
        return d
