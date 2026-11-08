"""Common machinery for the two-player grid worlds of Sunehag et al. (2017), sec. 4.2.

Shared properties (paper defaults; *configurable* where marked):

* Each agent sees an egocentric RGB window.  Paper: 3x5x5 = 4 squares ahead and 2 on
  each side.  *Configurable* via ``view_ahead``, ``view_side``, ``view_behind``.
* 8 actions: step forward / backward / left / right, rotate left / right, use beam,
  stand still.  The beam only lights up squares straight ahead (yellow).
* One *team* reward per step; agents never see individual rewards
  (``info["agent_rewards"]`` is provided for diagnostics only).
* Tasks repeat when solved, so an "episode" is just ``max_steps`` long
  (paper: 5,000 training / 2,000 testing steps; *configurable*).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ACTION_NAMES = ["forward", "backward", "left", "right", "rot_left", "rot_right", "beam", "noop"]
N_ACTIONS = len(ACTION_NAMES)
FWD, BACK, LEFT, RIGHT, ROT_L, ROT_R, BEAM, NOOP = range(N_ACTIONS)

# orientation index -> (d_row, d_col); 0=north, 1=east, 2=south, 3=west
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = "^>v<"

Pos = Tuple[int, int]
Color = Tuple[int, int, int]

C_FLOOR: Color = (0, 0, 0)
C_WALL: Color = (110, 110, 110)
C_SELF: Color = (0, 0, 255)
C_RED: Color = (255, 0, 0)
C_LIGHT_BLUE: Color = (120, 170, 255)
C_GREEN: Color = (0, 220, 0)
C_ORANGE: Color = (255, 140, 0)
C_MAGENTA: Color = (220, 0, 220)
C_YELLOW: Color = (255, 255, 0)
C_CYAN: Color = (0, 200, 200)

COLOR_CHARS: Dict[Color, str] = {
    C_FLOOR: " ", C_WALL: "#", C_SELF: "@", C_RED: "R", C_LIGHT_BLUE: "o",
    C_GREEN: "G", C_ORANGE: "L", C_MAGENTA: "P", C_YELLOW: "*", C_CYAN: "@",
}


def _right_of(f: Pos) -> Pos:
    """Unit vector to the agent's right when it faces ``f`` (row axis points down)."""
    return (f[1], -f[0])


class GridWorld(ABC):
    """Abstract two-player grid world with egocentric partial observations."""

    n_actions = N_ACTIONS
    LAYOUTS: Dict[str, Sequence[str]] = {}
    default_layout_name = ""
    agent_chars = "AB"                       # start-cell markers in ascii maps

    def __init__(
        self,
        layout: Optional[str | Sequence[str]] = None,
        view_ahead: int = 4,
        view_side: int = 2,
        view_behind: int = 0,
        max_steps: int = 5000,
        seed: Optional[int] = None,
        random_orientation: bool = False,
    ):
        """
        Args:
            layout: a key of ``LAYOUTS`` or a custom ascii map (list of equal-length strings).
            view_ahead / view_side / view_behind: field of view in squares. Observation
                shape is ``(3, view_ahead+view_behind+1, 2*view_side+1)``.
            max_steps: episode length (time limit; the task itself is continuing).
            random_orientation: spawn facing a random direction (default: towards map centre).
        """
        if layout is None:
            layout = self.default_layout_name
        self.layout_name = layout if isinstance(layout, str) else "custom"
        rows = list(self.LAYOUTS[layout]) if isinstance(layout, str) else list(layout)
        if len({len(r) for r in rows}) != 1:
            raise ValueError("all rows of a layout must have equal length")
        self.rows = rows
        self.height, self.width = len(rows), len(rows[0])

        self.view_ahead, self.view_side, self.view_behind = view_ahead, view_side, view_behind
        self.view_h = view_ahead + view_behind + 1
        self.view_w = 2 * view_side + 1
        self.obs_shape = (3, self.view_h, self.view_w)

        self.max_steps = max_steps
        self.random_orientation = random_orientation
        self.rng = np.random.default_rng(seed)

        self.walls = np.zeros((self.height, self.width), dtype=bool)
        self.starts: List[Optional[Pos]] = [None] * len(self.agent_chars)
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch in self.agent_chars:
                    self.starts[self.agent_chars.index(ch)] = (r, c)
                elif ch != ".":
                    self._parse_cell(ch, (r, c))
        if any(s is None for s in self.starts):
            raise ValueError(f"layout must contain start markers {list(self.agent_chars)}")
        self.n_agents = len(self.starts)
        self._post_parse()

        self.pos: List[Pos] = []
        self.dirs: List[int] = []
        self.beams: List[List[Pos]] = []
        self.t = 0
        self.last_rewards = [0.0] * self.n_agents
        self.reset()

    # ------------------------------------------------------------------ hooks
    def _parse_cell(self, ch: str, pos: Pos) -> None:
        raise ValueError(f"unknown map character {ch!r} at {pos}")

    def _post_parse(self) -> None:
        pass

    @abstractmethod
    def _reset_task(self) -> None:
        """Reset task-specific state. Positions are reset by the base class."""

    @abstractmethod
    def _apply_rules(self) -> List[float]:
        """Called after all agents moved. Returns per-agent contributions to the team reward."""

    @abstractmethod
    def _object_overlays(self, i: int) -> List[Tuple[Pos, Color]]:
        """Objects visible to agent ``i``."""

    @abstractmethod
    def _global_overlays(self) -> List[Tuple[Pos, Color]]:
        """All objects, for the global (debug) rendering."""

    def _other_color(self, viewer: int, other: int) -> Color:
        return C_RED

    def _self_color(self, i: int) -> Color:
        return C_SELF

    def _frozen(self, i: int) -> bool:
        return False

    # ------------------------------------------------------------- core API
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._reset_positions()
        self._reset_task()
        self.beams = [[] for _ in range(self.n_agents)]
        self.last_rewards = [0.0] * self.n_agents
        return self._observe()

    def _reset_positions(self) -> None:
        self.pos = [tuple(s) for s in self.starts]  # type: ignore[misc]
        self.dirs = []
        for (_, c) in self.pos:
            if self.random_orientation:
                self.dirs.append(int(self.rng.integers(4)))
            else:
                self.dirs.append(1 if c < self.width / 2 else 3)

    def step(self, actions: Sequence[int]):
        """Simultaneous move. Returns ``(obs, team_reward, done, info)``.
        obs: uint8 ``(n_agents, 3, H, W)``. Square conflicts are resolved in random order."""
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions")
        self.beams = [[] for _ in range(self.n_agents)]
        for i in self.rng.permutation(self.n_agents):
            if not self._frozen(i):
                self._do_action(int(i), int(actions[i]))
        contributions = self._apply_rules()
        self.last_rewards = list(contributions)
        self.t += 1
        done = self.t >= self.max_steps
        info = {"agent_rewards": list(contributions), "t": self.t}
        return self._observe(), float(sum(contributions)), done, info

    # -------------------------------------------------------------- movement
    def _free(self, p: Pos, ignore: int) -> bool:
        r, c = p
        if not (0 <= r < self.height and 0 <= c < self.width) or self.walls[r, c]:
            return False
        return all(j == ignore or self.pos[j] != p for j in range(self.n_agents))

    def _do_action(self, i: int, a: int) -> None:
        f = DIRS[self.dirs[i]]
        rt = _right_of(f)
        move = {FWD: f, BACK: (-f[0], -f[1]), LEFT: (-rt[0], -rt[1]), RIGHT: rt}.get(a)
        if move is not None:
            new = (self.pos[i][0] + move[0], self.pos[i][1] + move[1])
            if self._free(new, ignore=i):
                self.pos[i] = new
        elif a == ROT_L:
            self.dirs[i] = (self.dirs[i] - 1) % 4
        elif a == ROT_R:
            self.dirs[i] = (self.dirs[i] + 1) % 4
        elif a == BEAM:
            p = self.pos[i]
            while True:
                p = (p[0] + f[0], p[1] + f[1])
                if not (0 <= p[0] < self.height and 0 <= p[1] < self.width) or self.walls[p]:
                    break
                self.beams[i].append(p)

    # ---------------------------------------------------------- observations
    def _observe(self) -> np.ndarray:
        return np.stack([self._view(i) for i in range(self.n_agents)])

    def _view(self, i: int) -> np.ndarray:
        """Egocentric ``(3, H, W)`` uint8 window: row 0 is furthest ahead; the agent sits at
        row ``view_ahead``, column ``view_side``."""
        f = DIRS[self.dirs[i]]
        rt = _right_of(f)
        r0, c0 = self.pos[i]
        ahead = (self.view_ahead - np.arange(self.view_h))[:, None]
        side = (np.arange(self.view_w) - self.view_side)[None, :]
        rr = r0 + ahead * f[0] + side * rt[0]
        cc = c0 + ahead * f[1] + side * rt[1]
        inside = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
        wall = np.ones_like(inside)
        wall[inside] = self.walls[rr[inside], cc[inside]]
        img = np.zeros((self.view_h, self.view_w, 3), dtype=np.uint8)
        img[wall] = C_WALL

        def to_view(p: Pos):
            d0, d1 = p[0] - r0, p[1] - c0
            a = d0 * f[0] + d1 * f[1]
            s = d0 * rt[0] + d1 * rt[1]
            return self.view_ahead - a, s + self.view_side

        def put(p: Pos, color: Color) -> None:
            row, col = to_view(p)
            if 0 <= row < self.view_h and 0 <= col < self.view_w and not wall[row, col]:
                img[row, col] = color

        for p, col in self._object_overlays(i):
            put(p, col)
        for j in range(self.n_agents):                      # beams light up *empty* squares
            for p in self.beams[j]:
                row, col = to_view(p)
                if 0 <= row < self.view_h and 0 <= col < self.view_w and not img[row, col].any():
                    img[row, col] = C_YELLOW
        for j in range(self.n_agents):
            if j != i:
                put(self.pos[j], self._other_color(i, j))
        put(self.pos[i], self._self_color(i))
        return img.transpose(2, 0, 1).copy()

    def visible_cells(self, i: int) -> List[Pos]:
        f = DIRS[self.dirs[i]]
        rt = _right_of(f)
        r0, c0 = self.pos[i]
        out = []
        for a in range(-self.view_behind, self.view_ahead + 1):
            for s in range(-self.view_side, self.view_side + 1):
                p = (r0 + a * f[0] + s * rt[0], c0 + a * f[1] + s * rt[1])
                if 0 <= p[0] < self.height and 0 <= p[1] < self.width:
                    out.append(p)
        return out

    # ------------------------------------------------------------ rendering
    # ------------------------------------------------- full-scene (human) view hooks
    def scene_items(self) -> List[Tuple[Pos, Color, str]]:
        """Objects for the full-environment view: ``(position, colour, short label)``.
        Override in a game to label its objects; the default reuses ``_global_overlays``."""
        return [(p, col, "") for p, col in self._global_overlays()]

    def scene_legend(self) -> List[Tuple[Color, str]]:
        """``(colour, description)`` pairs explaining ``scene_items``."""
        return []

    def scene_summary(self) -> str:
        """One line of game-wide state for the full-environment view (e.g. items left)."""
        return ""

    def agent_status(self, i: int) -> str:
        """Short game-specific status of agent ``i`` (e.g. "carrying item")."""
        return ""

    def _agent_global_color(self, i: int) -> Color:
        return self._self_color(i) if i == 0 else self._other_color(0, i)

    def render_rgb(self, cell: int = 20, show_views: bool = True, show_fov: bool = True) -> np.ndarray:
        """Full-map RGB image (uint8); optionally tints the FOV and appends each agent's own window."""
        grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        grid[self.walls] = C_WALL
        if show_fov:
            tint = np.zeros((self.height, self.width), dtype=bool)
            for i in range(self.n_agents):
                for p in self.visible_cells(i):
                    tint[p] = True
            grid[tint & ~self.walls] = 28
        for p, col in self._global_overlays():
            grid[p] = col
        for i in range(self.n_agents):
            for p in self.beams[i]:
                if not grid[p].any() or (grid[p] == 28).all():
                    grid[p] = C_YELLOW
        img = np.kron(grid, np.ones((cell, cell, 1), dtype=np.uint8))
        m = max(2, cell // 4)                                  # facing marker size
        for i in range(self.n_agents):
            r, c = self.pos[i]
            y0, x0 = r * cell, c * cell
            img[y0:y0 + cell, x0:x0 + cell] = self._agent_global_color(i)
            d = DIRS[self.dirs[i]]
            my = y0 + (cell - m) // 2 + d[0] * ((cell - m) // 2)
            mx = x0 + (cell - m) // 2 + d[1] * ((cell - m) // 2)
            img[my:my + m, mx:mx + m] = 255
        if not show_views:
            return img
        vc = max(4, cell // 2)
        panels = []
        for i in range(self.n_agents):
            v = np.kron(self._view(i).transpose(1, 2, 0), np.ones((vc, vc, 1), dtype=np.uint8))
            panel = np.full((v.shape[0] + 12, v.shape[1] + 12, 3), 40, dtype=np.uint8)
            panel[6:6 + v.shape[0], 6:6 + v.shape[1]] = v
            panels.append(panel)
        height = max([img.shape[0]] + [p.shape[0] for p in panels])
        pad = lambda a: np.pad(a, ((0, height - a.shape[0]), (0, 0), (0, 0)), constant_values=40)
        return np.concatenate([pad(img)] + [pad(p) for p in panels], axis=1)

    def _global_chars(self) -> Dict[Pos, str]:
        return {}

    def render_text(self, color: bool = True, show_views: bool = True) -> str:
        """ASCII rendering. Agents are arrows (agent 0 blue, agent 1 red when ``color``)."""
        chars = self._global_chars()
        lines = []
        for r in range(self.height):
            row = ""
            for c in range(self.width):
                p = (r, c)
                ch = "#" if self.walls[p] else chars.get(p, ".")
                for i in range(self.n_agents):
                    if self.pos[i] == p:
                        ch = ARROWS[self.dirs[i]]
                        if color:
                            ch = f"\x1b[{34 if i == 0 else 31};1m{ch}\x1b[0m"
                for i in range(self.n_agents):
                    if p in self.beams[i] and ch == ".":
                        ch = "*"
                row += ch + " "
            lines.append(row.rstrip())
        text = "\n".join(lines)
        if show_views:
            vl = [self._view_text(i, color).split("\n") for i in range(self.n_agents)]
            text += "\n\n" + "\n".join("    ".join(parts) for parts in zip(*vl))
        return text

    def _view_text(self, i: int, color: bool) -> str:
        v = self._view(i).transpose(1, 2, 0)
        rows = []
        for y in range(self.view_h):
            row = ""
            for x in range(self.view_w):
                rgb = tuple(int(u) for u in v[y, x])
                if color:
                    row += f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m  \x1b[0m"
                else:
                    row += COLOR_CHARS.get(rgb, "?") * 2
            rows.append(row)
        return f"agent {i} view".ljust(self.view_w * 2) + "\n" + "\n".join(rows)

    # ------------------------------------------------------------- misc
    @property
    def spec(self) -> dict:
        return dict(n_agents=self.n_agents, n_actions=self.n_actions, obs_shape=self.obs_shape)

    def describe(self) -> str:
        return (f"{type(self).__name__}(layout={self.layout_name}, view={self.view_ahead}/{self.view_side}"
                f"/{self.view_behind}, max_steps={self.max_steps})")
