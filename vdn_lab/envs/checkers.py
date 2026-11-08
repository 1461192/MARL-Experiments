"""Checkers (paper sec. 4.2).

The map holds apples and lemons. Player 0 is very sensitive: +10 for an apple and -10 for
a lemon (for the team). Player 1 is less sensitive: +1 / -1. A wall of lemons separates the
players from the apples. Items disappear when collected; the environment resets when all
apples are eaten. Cooperative optimum: player 1 eats the obstructing lemons and leaves the
apples to player 0.

Legend: ``#`` wall, ``A``/``B`` starts, ``p`` apple, ``l`` lemon.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..registry import register_env
from .base import GridWorld, Pos, Color, C_GREEN, C_ORANGE

LAYOUTS = {
    "default": [
        "#############",
        "#A...l...pp.#",
        "#....l...pp.#",
        "#B...l...pp.#",
        "#############",
    ],
    # lemons and apples interleaved in the apple field (harder)
    "mixed": [
        "#############",
        "#A...l.p.pl.#",
        "#....l.l.p..#",
        "#B...l.p.pl.#",
        "#############",
    ],
}


@register_env("checkers")
class Checkers(GridWorld):
    LAYOUTS = LAYOUTS
    default_layout_name = "default"

    def __init__(self, values: Sequence[float] = (10.0, 1.0), **kwargs):
        """values[i] is the magnitude agent i earns/loses for apples/lemons."""
        self.values = tuple(values)
        self._initial_items: Dict[Pos, str] = {}
        super().__init__(**kwargs)

    def _parse_cell(self, ch: str, pos: Pos) -> None:
        if ch == "p":
            self._initial_items[pos] = "apple"
        elif ch == "l":
            self._initial_items[pos] = "lemon"
        else:
            super()._parse_cell(ch, pos)

    def _post_parse(self) -> None:
        if "apple" not in self._initial_items.values():
            raise ValueError("Checkers layout needs at least one apple 'p'")

    def _reset_task(self) -> None:
        self.items: Dict[Pos, str] = dict(self._initial_items)

    def _apply_rules(self) -> List[float]:
        out = [0.0, 0.0]
        for i in range(2):
            kind = self.items.pop(self.pos[i], None)
            if kind == "apple":
                out[i] += self.values[i]
            elif kind == "lemon":
                out[i] -= self.values[i]
        if "apple" not in self.items.values():     # all apples eaten -> full reset
            self._reset_positions()
            self.items = dict(self._initial_items)
        return out

    def _item_overlays(self):
        return [(p, C_GREEN if k == "apple" else C_ORANGE) for p, k in self.items.items()]

    def _object_overlays(self, i: int) -> List[Tuple[Pos, Color]]:
        return self._item_overlays()

    def _global_overlays(self) -> List[Tuple[Pos, Color]]:
        return self._item_overlays()

    def _global_chars(self):
        return {p: ("p" if k == "apple" else "l") for p, k in self.items.items()}

    def scene_items(self):
        return [(p, C_GREEN if k == "apple" else C_ORANGE, "A" if k == "apple" else "L")
                for p, k in self.items.items()]

    def scene_legend(self):
        return [(C_GREEN, f"A apple (agent 0: +{self.values[0]:g}, agent 1: +{self.values[1]:g})"),
                (C_ORANGE, f"L lemon (agent 0: -{self.values[0]:g}, agent 1: -{self.values[1]:g})")]

    def scene_summary(self) -> str:
        n_apples = sum(k == "apple" for k in self.items.values())
        return f"apples left: {n_apples}   lemons left: {len(self.items) - n_apples}"
