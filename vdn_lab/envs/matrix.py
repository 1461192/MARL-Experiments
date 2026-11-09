"""Two-player matrix games, the classic small sanity checks for cooperative MARL.

Two kinds of game, both with the same interface as the grid worlds (so every agent, the
runner, the CLI and the visuals work unchanged):

* ``matrix`` - a normal-form game. Agent 0 picks a **row**, agent 1 picks a **column**,
  both get the payoff in that cell. One round = one step.
* ``two_stage_matrix`` - agent 0 first picks **which matrix** is played (agent 1's action
  in that step is ignored), then both play the chosen matrix game as above. One round = two
  steps. The textbook example is the two-step game of QMIX (Rashid et al., 2018).

Payoffs
    A matrix is either *common-payoff* (``R x C`` numbers: both players get the cell value,
    the team reward is that value) or *general-sum* (``R x C x 2``: one payoff per player,
    the team reward is their sum). ``info["agent_rewards"]`` always sums to the team reward:
    a common payoff is split evenly between the two players, general-sum payoffs are given
    per player.

Actions
    All agents share one action space of size ``max(rows, columns[, number of matrices])``.
    When a matrix is smaller than that (non-square matrices, or two-stage games whose
    matrices have different shapes), action ``a`` selects row/column/matrix ``a mod size``.

Observations
    uint8 ``(n_agents, 1, 1, F)`` one-hot features scaled to 0/255 (the networks flatten
    them). In ``matrix`` there is no state, so the observation is a constant. In
    ``two_stage_matrix`` the features are ``[choosing, playing, matrix 0, ..., matrix K-1]``;
    with ``observe_choice=False`` agent 1 only sees *that* a matrix is being played, not
    which one (a partially observable variant).

Episodes
    Like the grid worlds the game repeats: an episode is ``max_steps`` steps of back-to-back
    rounds (the recurrent learners sample 8-step trajectories, so one-step episodes would
    not work). Rounds are independent, so the per-round optimum is unchanged; pass
    ``gamma=0`` (matrix) or a small ``gamma`` (two-stage) to the agent for a purely
    myopic learner.

Games are picked with ``layout`` (``--map`` in the CLI) from ``LAYOUTS``, or given directly
with ``payoffs=`` (a matrix for ``matrix``, a list of matrices for ``two_stage_matrix``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..registry import register_env

# ----------------------------------------------------------------------------- presets
# Each preset: payoffs (common-payoff R x C, or R x C x 2), optional action labels, a note.
MATRIX_GAMES: Dict[str, Dict[str, Any]] = {
    "coordination": dict(
        payoffs=[[1, 0],
                 [0, 1]],
        note="pure coordination: two equally good equilibria"),
    "climbing": dict(                          # Claus & Boutilier (1998)
        payoffs=[[11, -30, 0],
                 [-30, 7, 6],
                 [0, 0, 5]],
        labels=("a", "b", "c"),
        note="optimum (a,a)=11 is surrounded by heavy penalties"),
    "penalty": dict(                           # Claus & Boutilier (1998), k = -10
        payoffs=[[10, 0, -10],
                 [0, 2, 0],
                 [-10, 0, 10]],
        labels=("a", "b", "c"),
        note="two optima, mis-coordinating between them costs 10"),
    "nonmonotonic": dict(                      # QTRAN (Son et al., 2019), sec. 5.1
        payoffs=[[8, -12, -12],
                 [-12, 0, 0],
                 [-12, 0, 0]],
        labels=("A", "B", "C"),
        note="not representable by an additive (VDN) factorisation"),
    "prisoners_dilemma": dict(
        payoffs=[[(3, 3), (0, 5)],
                 [(5, 0), (1, 1)]],
        labels=("cooperate", "defect"),
        note="general-sum; team reward = sum of both payoffs"),
    "stag_hunt": dict(
        payoffs=[[(4, 4), (0, 3)],
                 [(3, 0), (3, 3)]],
        labels=("stag", "hare"),
        note="general-sum; payoff-dominant vs risk-dominant equilibrium"),
    "chicken": dict(
        payoffs=[[(0, 0), (-1, 1)],
                 [(1, -1), (-10, -10)]],
        labels=("swerve", "straight"),
        note="general-sum; team reward = sum of both payoffs"),
}

TWO_STAGE_GAMES: Dict[str, Dict[str, Any]] = {
    "two_step": dict(                          # QMIX (Rashid et al., 2018), fig. 3
        payoffs=[[[7, 7],
                  [7, 7]],
                 [[0, 1],
                  [1, 8]]],
        labels=("A", "B"),
        note="matrix 0 always pays 7; matrix 1 pays 8 only if both coordinate"),
    "climbing_or_safe": dict(
        payoffs=[MATRIX_GAMES["climbing"]["payoffs"],
                 [[6, 6, 6],
                  [6, 6, 6],
                  [6, 6, 6]]],
        labels=("a", "b", "c"),
        note="climbing game (best 11) vs a safe matrix that always pays 6"),
}


def _as_payoffs(m) -> np.ndarray:
    """-> float array ``(R, C, 2)`` of per-player payoffs, plus whether it was common-payoff."""
    a = np.asarray(m, dtype=np.float64)
    if a.ndim == 2:
        return np.stack([a, a], axis=-1)
    if a.ndim == 3 and a.shape[-1] == 2:
        return a
    raise ValueError(f"a payoff matrix must be R x C (common payoff) or R x C x 2, got shape {a.shape}")


def _is_common(m) -> bool:
    return np.asarray(m).ndim == 2


def _fmt(x: float) -> str:
    return f"{x:g}"


class _MatrixGameBase:
    """Shared bookkeeping, observation encoding and rendering for both matrix games."""

    LAYOUTS: Dict[str, Dict[str, Any]] = {}
    default_layout_name = ""
    n_agents = 2

    def _setup(self, layout, payoffs, labels, max_steps, seed, as_list: bool) -> None:
        if payoffs is None:
            if layout is None:
                layout = self.default_layout_name
            if layout not in self.LAYOUTS:
                raise KeyError(f"unknown game '{layout}'. Available: {sorted(self.LAYOUTS)}")
            preset = self.LAYOUTS[layout]
            payoffs = preset["payoffs"]
            labels = labels if labels is not None else preset.get("labels")
            self.note = preset.get("note", "")
            self.layout_name = layout
        else:
            self.note = ""
            self.layout_name = "custom"
        mats = list(payoffs) if as_list else [payoffs]
        if not mats:
            raise ValueError("need at least one payoff matrix")
        self.common = [_is_common(m) for m in mats]
        self.matrices = [_as_payoffs(m) for m in mats]
        self.labels = tuple(labels) if labels is not None else None
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.last_rewards = [0.0, 0.0]
        self.last_actions: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------- payoffs
    def _play(self, k: int, a0: int, a1: int) -> Tuple[int, int, List[float]]:
        """Play matrix ``k`` -> (row, col, per-agent contributions to the team reward)."""
        m = self.matrices[k]
        r, c = a0 % m.shape[0], a1 % m.shape[1]
        if self.common[k]:
            v = float(m[r, c, 0])
            return r, c, [v / 2, v / 2]
        return r, c, [float(m[r, c, 0]), float(m[r, c, 1])]

    def _team_value(self, k: int) -> np.ndarray:
        """Team reward of every cell of matrix ``k``."""
        m = self.matrices[k]
        return m[..., 0] if self.common[k] else m.sum(-1)

    def action_label(self, i: int, a: int, k: int) -> str:
        size = self.matrices[k].shape[i]
        idx = a % size
        if self.labels is not None and size <= len(self.labels):
            return str(self.labels[idx])
        return str(idx)

    # ------------------------------------------------------------- core API
    @property
    def spec(self) -> dict:
        return dict(n_agents=self.n_agents, n_actions=self.n_actions, obs_shape=self.obs_shape)

    def _encode(self, feats: np.ndarray) -> np.ndarray:
        return (feats * 255).astype(np.uint8).reshape(self.n_agents, *self.obs_shape)

    def render_rgb(self) -> np.ndarray:
        return self.render_scene()

    # ------------------------------------------------------------- drawing
    def _matrix_text(self, k: int, cell: Optional[Tuple[int, int]], title: str) -> List[str]:
        m = self.matrices[k]
        R, C = m.shape[:2]

        def entry(r, c):
            return _fmt(m[r, c, 0]) if self.common[k] else f"{_fmt(m[r, c, 0])},{_fmt(m[r, c, 1])}"

        rl = [self.action_label(0, r, k) for r in range(R)]
        cl = [self.action_label(1, c, k) for c in range(C)]
        w = max([len(entry(r, c)) for r in range(R) for c in range(C)] + [len(x) for x in cl]) + 2
        lw = max(len(x) for x in rl) + 1
        lines = [title, " " * lw + "".join(x.center(w) for x in cl)]
        for r in range(R):
            row = rl[r].ljust(lw)
            for c in range(C):
                e = entry(r, c)
                row += (f"[{e}]" if cell == (r, c) else f" {e} ").center(w)
            lines.append(row)
        return lines

    def render_scene(self, info: Optional[Dict[str, Any]] = None, *, cell: int = 44,
                     header: Optional[str] = None, **_: Any) -> np.ndarray:
        """Picture for the replay window / GIFs: every matrix as a table (the one in play
        outlined, the cell just played highlighted) and a panel with each agent's action and
        reward. Agent 0's choices are rows (blue), agent 1's are columns (red)."""
        from PIL import Image, ImageDraw
        from .scene import _font, agent_color, PANEL, TEXT, DIM, GRIDLINE, FLOOR

        info = info or {}
        fs = max(11, cell // 3)
        f, f_small, f_big = _font(fs), _font(max(10, fs - 2)), _font(fs + 2)
        pad, line = max(8, cell // 4), fs + 6
        head_h = (fs + 2 + 2 * pad) if header else 0

        def entry(k, r, c):
            m = self.matrices[k]
            return _fmt(m[r, c, 0]) if self.common[k] else f"{_fmt(m[r, c, 0])}, {_fmt(m[r, c, 1])}"

        K = len(self.matrices)
        texts = [entry(k, r, c) for k in range(K) for r in range(self.matrices[k].shape[0])
                 for c in range(self.matrices[k].shape[1])]
        texts += [self.action_label(1, c, k) for k in range(K) for c in range(self.matrices[k].shape[1])]
        cw = int(max([cell * 1.6] + [f.getlength(t) + 14 for t in texts]))   # matrix cell size
        ch = cell
        active, hit, chosen = self._scene_state()
        title_lines = self._scene_titles()
        label_w = [int(max(f_small.getlength(self.action_label(0, r, k))
                           for r in range(self.matrices[k].shape[0]))) + 12 for k in range(K)]
        block_w = [int(max(label_w[k] + self.matrices[k].shape[1] * cw, f.getlength(title_lines[k]))) + 2 * pad
                   for k in range(K)]
        mats_w = sum(block_w)
        mats_h = max(m.shape[0] for m in self.matrices) * ch + ch + line * 2
        panel = self._panel_lines(info)
        panel_w = int(max(f.getlength(t) for t, _ in panel)) + 2 * pad
        W = pad + mats_w + panel_w + pad
        H = head_h + pad + max(mats_h, line * (len(panel) + 1)) + pad
        img = Image.new("RGB", (W, H), PANEL)
        d = ImageDraw.Draw(img)
        if header:
            d.text((pad, pad), header, fill=TEXT, font=f_big)

        lo, hi = min(self._team_value(k).min() for k in range(len(self.matrices))), \
            max(self._team_value(k).max() for k in range(len(self.matrices)))
        x0 = pad
        for k, m in enumerate(self.matrices):
            R, C = m.shape[:2]
            y0 = head_h + pad
            d.text((x0, y0), title_lines[k], fill=TEXT if k == active or active is None else DIM, font=f)
            gy, gx = y0 + line * 2, x0 + label_w[k]         # grid origin (after labels)
            for c in range(C):
                lab = self.action_label(1, c, k)
                d.text((gx + c * cw + cw / 2 - f_small.getlength(lab) / 2, gy - line), lab,
                       fill=agent_color(1), font=f_small)
            for r in range(R):
                lab = self.action_label(0, r, k)
                d.text((gx - f_small.getlength(lab) - 6, gy + r * ch + ch / 2 - fs / 2), lab,
                       fill=agent_color(0), font=f_small)
            tv = self._team_value(k)
            for r in range(R):
                for c in range(C):
                    t = 0.5 if hi == lo else (tv[r, c] - lo) / (hi - lo)
                    fill = (int(255 - 70 * (1 - t)), int(200 + 50 * t), int(200 + 20 * t))
                    if k != active and active is not None:
                        fill = FLOOR
                    box = (gx + c * cw, gy + r * ch, gx + (c + 1) * cw, gy + (r + 1) * ch)
                    d.rectangle(box, fill=fill, outline=GRIDLINE)
                    e = entry(k, r, c)
                    d.text((box[0] + cw / 2 - f.getlength(e) / 2, box[1] + ch / 2 - fs / 2 - 1), e,
                           fill=TEXT, font=f)
            if hit is not None and hit[0] == k:
                r, c = hit[1], hit[2]
                d.rectangle((gx + c * cw, gy + r * ch, gx + (c + 1) * cw, gy + (r + 1) * ch),
                            outline=(245, 160, 0), width=max(3, cell // 12))
            if chosen == k and len(self.matrices) > 1:
                d.rectangle((gx - 3, gy - 3, gx + C * cw + 3, gy + R * ch + 3), outline=agent_color(0),
                            width=2)
            x0 += block_w[k]

        px, py = pad + mats_w, head_h + pad
        for text, col in panel:
            d.text((px, py), text, fill=col, font=f)
            py += line
        return np.asarray(img)

    def _panel_lines(self, info: Dict[str, Any]) -> List[Tuple[str, Tuple[int, int, int]]]:
        from .scene import agent_color, TEXT, DIM
        out = [(self.scene_summary(), TEXT)]
        if self.note:
            out.append((self.note, DIM))
        rew = info.get("agent_rewards") or self.last_rewards
        for i in range(self.n_agents):
            out.append((f"agent {i}: {self.agent_status(i)}   reward {rew[i]:+g}", agent_color(i)))
        return out

    def describe(self) -> str:
        return f"{type(self).__name__}(game={self.layout_name}, max_steps={self.max_steps})"


@register_env("matrix")
class MatrixGame(_MatrixGameBase):
    """Repeated two-player matrix game: agent 0 picks the row, agent 1 the column."""

    LAYOUTS = MATRIX_GAMES
    default_layout_name = "climbing"

    def __init__(self, layout: Optional[str] = None, payoffs: Optional[Sequence] = None,
                 labels: Optional[Sequence[str]] = None, max_steps: int = 100,
                 seed: Optional[int] = None):
        """
        Args:
            layout: a preset from ``MATRIX_GAMES`` (ignored when ``payoffs`` is given).
            payoffs: ``R x C`` common payoffs or ``R x C x 2`` per-player payoffs.
            labels: optional action names (rows and columns share them).
            max_steps: episode length = number of rounds.
        """
        self._setup(layout, payoffs, labels, max_steps, seed, as_list=False)
        R, C = self.matrices[0].shape[:2]
        self.n_actions = max(R, C)
        self.obs_shape = (1, 1, 1)
        self.last_cell: Optional[Tuple[int, int]] = None
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.last_rewards = [0.0, 0.0]
        self.last_actions, self.last_cell = None, None
        return self._observe()

    def _observe(self) -> np.ndarray:
        return self._encode(np.ones((self.n_agents, 1)))

    def step(self, actions: Sequence[int]):
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions")
        a0, a1 = int(actions[0]), int(actions[1])
        r, c, contrib = self._play(0, a0, a1)
        self.last_actions, self.last_cell, self.last_rewards = (a0, a1), (r, c), contrib
        self.t += 1
        done = self.t >= self.max_steps
        info = {"agent_rewards": list(contrib), "t": self.t, "cell": (r, c)}
        return self._observe(), float(sum(contrib)), done, info

    # ------------------------------------------------------------- display
    def render_text(self, color: bool = False, show_views: bool = False) -> str:
        title = f"matrix game '{self.layout_name}'  (agent 0 = row, agent 1 = column)"
        return "\n".join(self._matrix_text(0, self.last_cell, title))

    def _scene_state(self):
        return 0, (None if self.last_cell is None else (0, *self.last_cell)), None

    def _scene_titles(self) -> List[str]:
        return [f"'{self.layout_name}'  rows: agent 0, columns: agent 1"]

    def scene_summary(self) -> str:
        if self.last_cell is None:
            return f"round {self.t + 1}: waiting for the first joint action"
        r, c = self.last_cell
        return f"round {self.t}: played ({self.action_label(0, r, 0)}, {self.action_label(1, c, 0)})"

    def agent_status(self, i: int) -> str:
        if self.last_actions is None:
            return "-"
        return f"{'row' if i == 0 else 'column'} {self.action_label(i, self.last_actions[i], 0)}"


@register_env("two_stage_matrix")
class TwoStageMatrixGame(_MatrixGameBase):
    """Agent 0 first picks one of several matrices, then both play it (two steps per round)."""

    LAYOUTS = TWO_STAGE_GAMES
    default_layout_name = "two_step"

    def __init__(self, layout: Optional[str] = None, payoffs: Optional[Sequence] = None,
                 labels: Optional[Sequence[str]] = None, choice_rewards: Optional[Sequence[float]] = None,
                 observe_choice: bool = True, max_steps: int = 100, seed: Optional[int] = None):
        """
        Args:
            layout: a preset from ``TWO_STAGE_GAMES`` (ignored when ``payoffs`` is given).
            payoffs: list of matrices (each ``R x C`` common or ``R x C x 2``); shapes may differ.
            labels: optional action names for the matrix stage.
            choice_rewards: team reward for choosing each matrix (default 0 for all).
            observe_choice: if False, agent 1 does not observe which matrix agent 0 chose.
            max_steps: episode length in steps (each round takes two).
        """
        self._setup(layout, payoffs, labels, max_steps, seed, as_list=True)
        K = len(self.matrices)
        self.choice_rewards = [0.0] * K if choice_rewards is None else [float(x) for x in choice_rewards]
        if len(self.choice_rewards) != K:
            raise ValueError(f"need one choice reward per matrix ({K})")
        self.observe_choice = observe_choice
        self.n_actions = max([K] + [s for m in self.matrices for s in m.shape[:2]])
        self.obs_shape = (1, 1, 2 + K)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.stage = 0                         # 0 = agent 0 chooses a matrix, 1 = play it
        self.chosen: Optional[int] = None
        self.last_rewards = [0.0, 0.0]
        self.last_actions = None
        self.last_hit: Optional[Tuple[int, int, int]] = None   # (matrix, row, col) last played
        return self._observe()

    def _observe(self) -> np.ndarray:
        K = len(self.matrices)
        feats = np.zeros((self.n_agents, 2 + K))
        feats[:, self.stage] = 1
        if self.stage == 1:
            for i in range(self.n_agents):
                if i == 0 or self.observe_choice:
                    feats[i, 2 + self.chosen] = 1
        return self._encode(feats)

    def step(self, actions: Sequence[int]):
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions")
        a0, a1 = int(actions[0]), int(actions[1])
        info: Dict[str, Any] = {"t": self.t + 1, "stage": self.stage}
        if self.stage == 0:
            k = a0 % len(self.matrices)
            v = self.choice_rewards[k]
            contrib = [v, 0.0]
            self.chosen, self.stage = k, 1
            info["matrix"] = k
        else:
            k = self.chosen
            r, c, contrib = self._play(k, a0, a1)
            self.last_hit = (k, r, c)
            self.stage = 0
            info.update(matrix=k, cell=(r, c))
        self.last_actions, self.last_rewards = (a0, a1), contrib
        self.t += 1
        done = self.t >= self.max_steps
        info["agent_rewards"] = list(contrib)
        return self._observe(), float(sum(contrib)), done, info

    # ------------------------------------------------------------- display
    def render_text(self, color: bool = False, show_views: bool = False) -> str:
        head = (f"two-stage matrix game '{self.layout_name}' - "
                + ("agent 0 picks a matrix" if self.stage == 0 else f"playing matrix {self.chosen}"))
        blocks = []
        for k in range(len(self.matrices)):
            cell = self.last_hit[1:] if self.last_hit is not None and self.last_hit[0] == k else None
            mark = " <" if (self.stage == 1 and k == self.chosen) else ""
            blocks.append(self._matrix_text(k, cell, f"matrix {k}{mark}"))
        width = [max(len(s) for s in b) for b in blocks]
        rows = max(len(b) for b in blocks)
        lines = []
        for j in range(rows):
            lines.append("    ".join((b[j] if j < len(b) else "").ljust(w) for b, w in zip(blocks, width)).rstrip())
        return head + "\n" + "\n".join(lines)

    def _scene_state(self):
        if self.stage == 1:
            return self.chosen, None, self.chosen
        return None, self.last_hit, None

    def _scene_titles(self) -> List[str]:
        out = []
        for k in range(len(self.matrices)):
            extra = f"  (choice reward {_fmt(self.choice_rewards[k])})" if self.choice_rewards[k] else ""
            out.append(f"matrix {k}{extra}")
        return out

    def scene_summary(self) -> str:
        if self.stage == 1:
            return f"step {self.t}: agent 0 chose matrix {self.chosen}; next both play it"
        if self.last_hit is None:
            return "agent 0 is choosing a matrix"
        k, r, c = self.last_hit
        return (f"step {self.t}: played ({self.action_label(0, r, k)}, {self.action_label(1, c, k)}) "
                f"in matrix {k}; agent 0 chooses again")

    def agent_status(self, i: int) -> str:
        if self.last_actions is None:
            return "-"
        if self.stage == 1:                               # last step was the choice
            return f"chose matrix {self.chosen}" if i == 0 else "waiting (action ignored)"
        k = self.last_hit[0]
        return f"{'row' if i == 0 else 'column'} {self.action_label(i, self.last_actions[i], k)}"
