"""Full-environment picture for humans (replay window and GIFs).

Unlike ``GridWorld.render_rgb`` (a pixel dump close to what the agents observe), this draws
the *whole* map so the interaction is easy to follow:

* light floor with grid lines, dark walls,
* each agent as a numbered disc with a heading arrow, in a fixed colour (agent 0 blue,
  agent 1 red, ...), plus a dot when it carries something,
* each agent's field of view outlined in its colour (optional),
* game objects as labelled squares, beams as yellow lines,
* a side panel with legend, per-agent status (position, heading, last action, reward), and
  optionally each agent's own egocentric observation (small, labelled).

Games describe their objects through the ``scene_items`` / ``scene_legend`` /
``agent_status`` hooks of ``GridWorld``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .base import ACTION_NAMES

AGENT_COLORS = [(40, 90, 230), (220, 50, 50), (40, 160, 70), (200, 120, 0)]
HEADINGS = ["N", "E", "S", "W"]
FLOOR, WALL, GRIDLINE, PANEL, TEXT, DIM = (238, 238, 232), (60, 60, 66), (205, 205, 198), (250, 250, 248), (25, 25, 25), (110, 110, 110)
BEAM = (245, 205, 0)


@lru_cache(maxsize=8)
def _font(size: int):
    from PIL import ImageFont
    try:                                                # matplotlib ships DejaVu Sans
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size)
    except TypeError:                                   # Pillow < 10.1
        return ImageFont.load_default()


def agent_color(i: int):
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def _blend(a, b, t: float):
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def render_scene(env, info: Optional[Dict[str, Any]] = None, *, cell: int = 44, show_fov: bool = True,
                 show_views: bool = True, header: Optional[str] = None) -> np.ndarray:
    """RGB uint8 image of the full environment plus an info panel.

    ``info`` may hold ``step``, ``reward``, ``return``, ``actions`` (as passed to renderers);
    ``header`` adds a title line on top (used for GIFs, the window has its own title)."""
    custom = getattr(env, "render_scene", None)
    if callable(custom):                                # non-grid games draw themselves
        return custom(info, cell=cell, show_fov=show_fov, show_views=show_views, header=header)
    from PIL import Image, ImageDraw

    info = info or {}
    H, W, n = env.height, env.width, env.n_agents
    pad = max(8, cell // 4)
    fs = max(11, cell // 3)
    f, f_small, f_big = _font(fs), _font(max(10, fs - 2)), _font(fs + 2)
    line = fs + 6
    head_h = (fs + 2 + 2 * pad) if header else 0

    map_w, map_h = W * cell, H * cell
    vc = max(6, cell // 3)                              # cell size of the observation insets
    view_w, view_h = env.view_w * vc, env.view_h * vc
    legend = list(env.scene_legend())
    actions = info.get("actions")
    agent_rewards = info.get("agent_rewards")

    summary = env.scene_summary()
    fov_note = "tinted/outlined squares = what each agent can see"
    status = []                                         # (title, detail lines) per agent
    for i in range(n):
        st = env.agent_status(i)
        r, c = env.pos[i]
        act = ACTION_NAMES[int(actions[i])] if actions is not None and i < len(actions) else "-"
        rew = agent_rewards[i] if agent_rewards is not None else env.last_rewards[i]
        status.append((f"agent {i}" + (f"  - {st}" if st else ""),
                       [f"pos ({r},{c})  facing {HEADINGS[env.dirs[i]]}  action {act}",
                        f"reward this step {rew:+g}"]))
    views_note = "what each agent sees (egocentric, facing up)"

    # ---- side panel layout (sized to its text) ---------------------------------------
    sw = fs
    text_w = [f_small.getlength("legend")]
    text_w += [sw + 6 + f_small.getlength(t) for _, t in legend]
    text_w += [f_small.getlength(t) for t in (summary, fov_note if show_fov else "",
                                              views_note if show_views else "")]
    for title, lines in status:
        text_w += [sw + 6 + f.getlength(title)] + [sw + 6 + f_small.getlength(t) for t in lines]
    panel_w = int(max(max(text_w), (view_w + pad) * n if show_views else 0)) + pad
    panel_h = (line * (2 + len(legend) + bool(summary)) + pad
               + n * (line * 3 + pad)
               + ((view_h + line + pad) if show_views else 0))
    body_h = max(map_h, panel_h)
    img = Image.new("RGB", (pad + map_w + pad + panel_w + pad, head_h + pad + body_h + pad), PANEL)
    d = ImageDraw.Draw(img)
    ox, oy = pad, head_h + pad

    if header:
        d.text((pad, pad), header, fill=TEXT, font=f_big)

    def box(r, c, shrink=0):
        x0, y0 = ox + c * cell, oy + r * cell
        return [x0 + shrink, y0 + shrink, x0 + cell - 1 - shrink, y0 + cell - 1 - shrink]

    def centre(r, c):
        return ox + c * cell + cell / 2, oy + r * cell + cell / 2

    # ---- map: floor, walls, field of view ------------------------------------------
    fov_owner = {}
    if show_fov:
        for i in range(n):
            for p in env.visible_cells(i):
                fov_owner.setdefault(p, []).append(i)
    for r in range(H):
        for c in range(W):
            if env.walls[r, c]:
                col = WALL
            else:
                col = FLOOR
                for i in fov_owner.get((r, c), []):
                    col = _blend(col, agent_color(i), 0.13)
            d.rectangle(box(r, c), fill=col, outline=GRIDLINE if not env.walls[r, c] else WALL)
    if show_fov:                                        # outline each agent's view region
        for i in range(n):
            vis = set(env.visible_cells(i))
            colr = agent_color(i)
            wpx = max(2, cell // 16)
            for (r, c) in vis:
                x0, y0, x1, y1 = ox + c * cell, oy + r * cell, ox + (c + 1) * cell, oy + (r + 1) * cell
                if (r - 1, c) not in vis:
                    d.line([x0, y0, x1, y0], fill=colr, width=wpx)
                if (r + 1, c) not in vis:
                    d.line([x0, y1, x1, y1], fill=colr, width=wpx)
                if (r, c - 1) not in vis:
                    d.line([x0, y0, x0, y1], fill=colr, width=wpx)
                if (r, c + 1) not in vis:
                    d.line([x1, y0, x1, y1], fill=colr, width=wpx)

    # ---- objects ---------------------------------------------------------------------
    for (r, c), colr, label in env.scene_items():
        d.rectangle(box(r, c, shrink=max(2, cell // 10)), fill=tuple(colr), outline=_blend(colr, (0, 0, 0), 0.4))
        if label:
            txt = (255, 255, 255) if sum(colr) < 400 else TEXT
            d.text(centre(r, c), label, fill=txt, font=f, anchor="mm")

    # ---- beams -----------------------------------------------------------------------
    for i in range(n):
        cells = env.beams[i]
        if cells:
            a, b = centre(*env.pos[i]), centre(*cells[-1])
            d.line([a, b], fill=BEAM, width=max(3, cell // 8))

    # ---- agents ----------------------------------------------------------------------
    carrying = getattr(env, "scene_carrying", None)
    for i in range(n):
        r, c = env.pos[i]
        cx, cy = centre(r, c)
        rad = cell * 0.40
        colr = agent_color(i)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=colr, outline=(0, 0, 0), width=max(1, cell // 22))
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][env.dirs[i]]
        tip = (cx + dc * rad * 1.05, cy + dr * rad * 1.05)
        base = (cx + dc * rad * 0.45, cy + dr * rad * 0.45)
        side = (-dr * rad * 0.42, dc * rad * 0.42)
        d.polygon([tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])],
                  fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((cx - dc * rad * 0.25, cy - dr * rad * 0.25), str(i), fill=(255, 255, 255), font=f, anchor="mm")
        if carrying is not None and carrying(i):
            k = rad * 0.38
            d.ellipse([cx + rad * 0.45 - k, cy - rad * 0.95 - k / 2, cx + rad * 0.45 + k, cy - rad * 0.95 + k * 1.5],
                      fill=(220, 0, 220), outline=(0, 0, 0))

    # ---- side panel ------------------------------------------------------------------
    px, py = ox + map_w + pad + pad // 2, oy
    d.text((px, py), "legend", fill=DIM, font=f_small)
    py += line
    for colr, text in legend:
        d.rectangle([px, py + 1, px + sw, py + 1 + sw], fill=tuple(colr), outline=(0, 0, 0))
        d.text((px + sw + 6, py), text, fill=TEXT, font=f_small)
        py += line
    if summary:
        d.text((px, py), summary, fill=TEXT, font=f_small)
        py += line
    if show_fov:
        d.text((px, py), fov_note, fill=DIM, font=f_small)
    py += line + pad // 2

    for i in range(n):
        colr = agent_color(i)
        d.ellipse([px, py + 1, px + sw, py + 1 + sw], fill=colr, outline=(0, 0, 0))
        title, lines = status[i]
        d.text((px + sw + 6, py), title, fill=TEXT, font=f)
        py += line
        for t in lines:
            d.text((px + sw + 6, py), t, fill=DIM, font=f_small)
            py += line
        py += pad // 2

    if show_views:
        d.text((px, py), views_note, fill=DIM, font=f_small)
        py += line
        for i in range(n):
            v = env._view(i).transpose(1, 2, 0)
            v = np.kron(v, np.ones((vc, vc, 1), dtype=np.uint8))
            x = px + i * (view_w + pad)
            img.paste(Image.fromarray(v), (x, py))
            d.rectangle([x - 1, py - 1, x + view_w, py + view_h], outline=agent_color(i), width=2)
    return np.asarray(img)
