"""Pluggable visual back-ends. Add your own by subclassing ``Renderer``."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Renderer:
    """Interface used by the runner. All methods are optional no-ops."""

    def begin_episode(self, env, title: str = "") -> None: ...
    def frame(self, env, info: Dict[str, Any]) -> None: ...
    def end_episode(self, summary: str = "") -> None: ...
    def close(self) -> None: ...


class NullRenderer(Renderer):
    """Headless: no output, no overhead."""


class TextRenderer(Renderer):
    """Terminal rendering (ASCII map + each agent's observation window)."""

    def __init__(self, delay: float = 0.08, every: int = 1, color: bool = True,
                 views: bool = True, stream=None, clear: Optional[bool] = None):
        self.delay, self.every, self.color, self.views = delay, max(1, every), color, views
        self.stream = stream or sys.stdout
        self.clear = self.stream.isatty() if clear is None else clear
        self.title = ""

    def begin_episode(self, env, title: str = "") -> None:
        self.title = title
        self.frame(env, {"step": 0, "reward": 0.0, "return": 0.0, "force": True})

    def frame(self, env, info: Dict[str, Any]) -> None:
        step = info.get("step", env.t)
        if step % self.every and not info.get("force"):
            return
        head = (f"{self.title}  step {step}/{env.max_steps}  reward {info.get('reward', 0):+.1f}  "
                f"return {info.get('return', 0):.1f}")
        if self.clear:
            self.stream.write("\x1b[H\x1b[2J")
        else:
            self.stream.write("\n" + "-" * 60 + "\n")
        self.stream.write(head + "\n\n" + env.render_text(color=self.color, show_views=self.views) + "\n")
        self.stream.flush()
        if self.delay > 0:
            time.sleep(self.delay)

    def end_episode(self, summary: str = "") -> None:
        if summary:
            self.stream.write(summary + "\n")
            self.stream.flush()


class WindowRenderer(Renderer):
    """Live matplotlib window shown *while* episodes run (used to peek at training episodes).

    For finished agents ``make_renderer("window")`` returns the interactive
    ``viewer.ReplayWindowRenderer`` instead (pause / replay / speed control)."""

    def __init__(self, delay: float = 0.08, every: int = 1, cell: int = 36, **_):
        import matplotlib.pyplot as plt
        self.plt, self.delay, self.every, self.cell = plt, delay, max(1, every), cell
        self.fig = self.ax = self.im = None
        self.title = ""

    def begin_episode(self, env, title: str = "") -> None:
        self.title = title
        self.plt.ion()
        if self.fig is None:
            self.fig, self.ax = self.plt.subplots(figsize=(11, 5.5))
            self.fig.subplots_adjust(0.01, 0.01, 0.99, 0.93)
            self.ax.axis("off")
        self.im = None
        self.frame(env, {"step": 0, "return": 0.0, "force": True})

    def frame(self, env, info: Dict[str, Any]) -> None:
        step = info.get("step", env.t)
        if step % self.every and not info.get("force"):
            return
        from ..envs.scene import render_scene
        img = render_scene(env, info, cell=self.cell)
        if self.im is None or self.im.get_array().shape != img.shape:
            self.im = self.ax.imshow(img)
        else:
            self.im.set_data(img)
        self.ax.set_title(f"{self.title}  step {step}/{env.max_steps}  reward {info.get('reward', 0):+.1f}  "
                          f"return {info.get('return', 0):.1f}")
        self.fig.canvas.draw_idle()
        self.plt.pause(max(self.delay, 0.001))

    def close(self) -> None:
        if self.fig is not None:
            self.plt.close(self.fig)
            self.fig = None


class GifRenderer(Renderer):
    """Saves each episode as an animated GIF (no display needed)."""

    def __init__(self, out_dir: str = "gifs", delay: float = 0.08, every: int = 1, cell: int = 36,
                 prefix: str = "episode", max_frames: int = 600, **_):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.delay, self.every, self.cell, self.prefix = delay, max(1, every), cell, prefix
        self.max_frames = max_frames
        self.frames: list = []
        self.count = 0
        self.last_path: Optional[Path] = None

    def begin_episode(self, env, title: str = "") -> None:
        self.frames = []
        self.title = title
        self.frame(env, {"step": 0, "reward": 0.0, "return": 0.0, "force": True})

    def frame(self, env, info: Dict[str, Any]) -> None:
        step = info.get("step", env.t)
        if (step % self.every and not info.get("force")) or len(self.frames) >= self.max_frames:
            return
        from PIL import Image
        from ..envs.scene import render_scene
        head = (f"{self.title}   step {step}/{env.max_steps}   reward {info.get('reward', 0):+.1f}   "
                f"return {info.get('return', 0):.1f}")
        self.frames.append(Image.fromarray(render_scene(env, info, cell=self.cell, header=head)))

    def end_episode(self, summary: str = "") -> None:
        if not self.frames:
            return
        self.last_path = self.dir / f"{self.prefix}_{self.count:03d}.gif"
        self.frames[0].save(self.last_path, save_all=True, append_images=self.frames[1:],
                            duration=max(20, int(self.delay * 1000)), loop=0)
        self.count += 1
        self.frames = []
        print(f"saved {self.last_path}")


def make_renderer(kind: str, *, delay: float = 0.08, every: int = 1, color: bool = True,
                  views: bool = True, out_dir: str = "gifs", prefix: str = "episode",
                  live: bool = False) -> Renderer:
    """``live=True`` selects the plain live window for ``"window"`` (training peeks); otherwise
    sessions are recorded and opened in the interactive replay window when they are done."""
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return NullRenderer()
    if kind == "text":
        return TextRenderer(delay, every, color, views)
    if kind in ("window", "mpl", "matplotlib"):
        if live:
            return WindowRenderer(delay, every)
        from .viewer import ReplayWindowRenderer
        return ReplayWindowRenderer(delay, every, fallback_dir=out_dir, prefix=prefix)
    if kind == "gif":
        return GifRenderer(out_dir, delay, every, prefix=prefix)
    raise ValueError(f"unknown visual mode '{kind}' (none | text | window | gif)")
