"""Interactive replay window for finished agents.

Sessions run with a trained agent are *recorded* (a cheap snapshot of the game state per
frame), then shown in a separate matplotlib window where you can

* pause / resume          (Pause button, or ``space``)
* replay a session        (Replay button, or ``r``; ``n`` / ``p`` = next / previous session)
* change the replay speed (speed slider, or ``+`` / ``-``; ``0`` = back to 1x)
* scrub through a session (frame slider, or ``left`` / ``right`` to step one frame while paused)
* toggle the agents' own views (``v``) and the field-of-view outlines (``f``)

Every frame shows the *whole* environment (``vdn_lab.envs.scene.render_scene``): the full map
with numbered agents and their headings, labelled objects, each agent's field of view, a legend,
per-agent status, and - smaller - what each agent itself observes.

Because frames are re-rendered from recorded game states, replaying is exact and never
re-runs the agent. ``q`` or closing the window ends the viewer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .render import Renderer

NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass
class SessionRecording:
    """One recorded session: game-state snapshots plus the per-frame info shown in the title."""

    title: str = ""
    states: List[Any] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def __len__(self) -> int:
        return len(self.states)

    def image(self, k: int, cell: int = 44, show_fov: bool = True, show_views: bool = True,
              header: bool = False):
        from ..envs.scene import render_scene
        info = self.infos[k]
        head = None
        if header:
            head = (f"{self.title}   step {info['step']}/{info['max_steps']}   "
                    f"reward {info['reward']:+.1f}   return {info['return']:.1f}")
        return render_scene(self.states[k], info, cell=cell, show_fov=show_fov, show_views=show_views,
                            header=head)


class SessionRecorder(Renderer):
    """Records every ``every``-th frame of each episode (no output while the session runs)."""

    def __init__(self, every: int = 1):
        self.every = max(1, every)
        self.recordings: List[SessionRecording] = []
        self._cur: Optional[SessionRecording] = None

    def begin_episode(self, env, title: str = "") -> None:
        self._cur = SessionRecording(title=title)
        self.recordings.append(self._cur)
        self._snap(env, {"step": 0, "reward": 0.0, "return": 0.0})

    def frame(self, env, info: Dict[str, Any]) -> None:
        step = info.get("step", env.t)
        if self._cur is None:
            return
        if step % self.every and not (env.t >= env.max_steps):     # always keep the last frame
            return
        self._snap(env, info)

    def _snap(self, env, info: Dict[str, Any]) -> None:
        state = copy.deepcopy(env)
        self._cur.states.append(state)
        self._cur.infos.append({"step": info.get("step", env.t), "reward": float(info.get("reward", 0.0)),
                                "return": float(info.get("return", 0.0)), "max_steps": env.max_steps,
                                "actions": None if info.get("actions") is None else [int(a) for a in info["actions"]],
                                "agent_rewards": info.get("agent_rewards")})

    def end_episode(self, summary: str = "") -> None:
        if self._cur is not None:
            self._cur.summary = summary
            if summary:
                print(summary)
        self._cur = None


def has_interactive_display() -> bool:
    import matplotlib
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


class ReplayViewer:
    """Matplotlib window with play/pause, replay, session switching, speed control and a scrubber.

    ``delay`` is the seconds per frame at 1x speed. Call ``show()`` to open the window (blocks
    until it is closed). The control methods (``toggle_pause``, ``replay``, ``set_speed``,
    ``goto_session``, ``step``) can also be called from code.
    """

    def __init__(self, recordings: Sequence[SessionRecording], delay: float = 0.08, cell: int = 44,
                 speed: float = 1.0, autoplay: bool = True, title: str = "vdn_lab replay"):
        recordings = [r for r in recordings if len(r)]
        if not recordings:
            raise ValueError("nothing to replay: no recorded frames")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.recordings = recordings
        self.delay, self.cell = max(delay, 0.001), cell
        self.session, self.k = 0, 0
        self.paused = not autoplay
        self.speed = speed
        self.show_fov, self.show_views = True, True
        self._cache: Dict[tuple, Any] = {}
        self._syncing = False

        self.fig = plt.figure(figsize=(12, 7))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except AttributeError:
            pass
        self.ax = self.fig.add_axes([0.01, 0.20, 0.98, 0.72])
        self.ax.axis("off")
        self.im = self.ax.imshow(self._image(), interpolation="antialiased")
        self.status = self.fig.text(0.5, 0.94, "", ha="center", va="center", fontsize=11, family="monospace")
        self.fig.text(0.5, 0.015, "space pause   r replay   n/p next/prev session   "
                                  "\u2190/\u2192 step   +/- speed   0 reset speed   v agent views   f field of view   q quit",
                      ha="center", fontsize=8, color="0.4")

        self.s_frame = Slider(self.fig.add_axes([0.12, 0.135, 0.76, 0.03]), "frame", 0,
                              max(1, len(self.rec) - 1), valinit=0, valstep=1)
        self.s_frame.on_changed(self._on_scrub)
        self.s_speed = Slider(self.fig.add_axes([0.70, 0.06, 0.18, 0.04]), "speed",
                              0, len(SPEEDS) - 1, valinit=self._speed_index(speed), valstep=1)
        self.s_speed.valtext.set_text(self._speed_label())
        self.s_speed.on_changed(self._on_speed)

        self.buttons = {}
        for name, x, w, fn in [("◀◀ prev", 0.05, 0.10, lambda _: self.goto_session(self.session - 1)),
                               ("↻ replay", 0.16, 0.10, lambda _: self.replay()),
                               ("pause", 0.27, 0.10, lambda _: self.toggle_pause()),
                               ("next ▶▶", 0.38, 0.10, lambda _: self.goto_session(self.session + 1))]:
            b = Button(self.fig.add_axes([x, 0.05, w, 0.06]), name)
            b.on_clicked(fn)
            self.buttons[name] = b
        self.b_pause = self.buttons["pause"]
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.timer = self.fig.canvas.new_timer(interval=self._interval())
        self.timer.add_callback(self._tick)
        self._refresh()

    # ---- state helpers -------------------------------------------------------------
    @property
    def rec(self) -> SessionRecording:
        return self.recordings[self.session]

    def _interval(self) -> int:
        return max(10, int(1000 * self.delay / self.speed))

    def _image(self):
        key = (self.session, self.k, self.show_fov, self.show_views)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = self.rec.image(self.k, self.cell, show_fov=self.show_fov,
                                              show_views=self.show_views)
        return self._cache[key]

    @staticmethod
    def _speed_index(speed: float) -> int:
        return min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))

    def _speed_label(self) -> str:
        return f"{self.speed:g}x"

    @property
    def at_end(self) -> bool:
        return self.k >= len(self.rec) - 1

    # ---- controls --------------------------------------------------------------------
    def toggle_pause(self) -> None:
        self.set_paused(not self.paused)

    def set_paused(self, paused: bool) -> None:
        if not paused and self.at_end and self.session == len(self.recordings) - 1:
            self.k = 0                       # "play" on the finished last session = replay it
        self.paused = paused
        self._refresh()

    def replay(self) -> None:
        """Restart the current session from its first frame and play it."""
        self.k = 0
        self.paused = False
        self._refresh()

    def goto_session(self, i: int) -> None:
        self.session = i % len(self.recordings)
        self.k = 0
        self._sync_frame_slider_range()
        self._refresh()

    def step(self, delta: int) -> None:
        """Move ``delta`` frames (pauses playback)."""
        self.paused = True
        self.k = min(max(self.k + delta, 0), len(self.rec) - 1)
        self._refresh()

    def set_speed(self, speed: float) -> None:
        self.speed = SPEEDS[self._speed_index(speed)]
        self.timer.interval = self._interval()
        self._syncing = True
        self.s_speed.set_val(self._speed_index(self.speed))
        self._syncing = False
        self.s_speed.valtext.set_text(self._speed_label())
        self._refresh()

    def toggle_views(self) -> None:
        """Show / hide the panels with each agent's own (egocentric) observation."""
        self.show_views = not self.show_views
        self._refresh()

    def toggle_fov(self) -> None:
        """Show / hide the field-of-view outlines on the map."""
        self.show_fov = not self.show_fov
        self._refresh()

    def faster(self, n: int = 1) -> None:
        self.set_speed(SPEEDS[min(max(self._speed_index(self.speed) + n, 0), len(SPEEDS) - 1)])

    # ---- callbacks -------------------------------------------------------------------
    def _tick(self) -> None:
        if self.paused:
            return
        if not self.at_end:
            self.k += 1
        elif self.session < len(self.recordings) - 1:
            self.session += 1                # auto-advance to the next session
            self.k = 0
            self._sync_frame_slider_range()
        else:
            self.paused = True               # finished the last session
        self._refresh()

    def _on_scrub(self, val) -> None:
        if self._syncing:
            return
        self.k = int(min(max(val, 0), len(self.rec) - 1))
        self.paused = True
        self._refresh()

    def _on_speed(self, val) -> None:
        if not self._syncing:
            self.set_speed(SPEEDS[int(val)])

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        actions = {" ": self.toggle_pause, "r": self.replay, "n": lambda: self.goto_session(self.session + 1),
                   "p": lambda: self.goto_session(self.session - 1), "right": lambda: self.step(1),
                   "left": lambda: self.step(-1), "+": self.faster, "=": self.faster, "up": self.faster,
                   "-": lambda: self.faster(-1), "down": lambda: self.faster(-1), "0": lambda: self.set_speed(1.0),
                   "home": lambda: self.step(-len(self.rec)), "end": lambda: self.step(len(self.rec)),
                   "v": self.toggle_views, "f": self.toggle_fov}
        if key in actions:
            actions[key]()

    # ---- drawing ---------------------------------------------------------------------
    def _sync_frame_slider_range(self) -> None:
        n = max(1, len(self.rec) - 1)
        self.s_frame.valmax = n
        self.s_frame.ax.set_xlim(0, n)

    def status_text(self) -> str:
        info = self.rec.infos[self.k]
        state = "paused" if self.paused else "playing"
        if self.paused and self.at_end:
            state = "finished - r to replay"
        return (f"{self.rec.title or f'session {self.session + 1}'}  [{self.session + 1}/{len(self.recordings)}]"
                f"   step {info['step']}/{info['max_steps']}   reward {info['reward']:+.1f}"
                f"   return {info['return']:.1f}   {self._speed_label()}   ({state})")

    def _refresh(self) -> None:
        img = self._image()
        if img.shape != self.im.get_array().shape:
            self.im.remove()
            self.im = self.ax.imshow(img, interpolation="antialiased")
        else:
            self.im.set_data(img)
        self.status.set_text(self.status_text())
        self.b_pause.label.set_text("▶ play" if self.paused else "❚❚ pause")
        self._syncing = True
        self.s_frame.set_val(self.k)
        self._syncing = False
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the window and block until it is closed."""
        self.timer.start()
        self.plt.show(block=True)
        self.timer.stop()

    def close(self) -> None:
        self.timer.stop()
        self.plt.close(self.fig)


class ReplayWindowRenderer(SessionRecorder):
    """``visual="window"`` for finished agents: record the sessions, then open a ``ReplayViewer``.

    Without an interactive display the sessions are saved as GIFs instead (``fallback_dir``).
    """

    def __init__(self, delay: float = 0.08, every: int = 1, cell: int = 44, fallback_dir: str = "gifs",
                 prefix: str = "episode", speed: float = 1.0, **_):
        super().__init__(every)
        self.delay, self.cell, self.fallback_dir, self.prefix, self.speed = delay, cell, fallback_dir, prefix, speed
        self.viewer: Optional[ReplayViewer] = None

    def close(self) -> None:
        recs = [r for r in self.recordings if len(r)]
        if not recs:
            return
        if has_interactive_display():
            self.viewer = ReplayViewer(recs, delay=self.delay, cell=self.cell, speed=self.speed)
            self.viewer.show()
        else:
            print("no interactive display available for the replay window; saving GIFs instead")
            save_gifs(recs, self.fallback_dir, self.prefix, self.delay)
        self.recordings = []


def save_gifs(recordings: Sequence[SessionRecording], out_dir: str, prefix: str = "episode",
              delay: float = 0.08, cell: int = 36) -> List[Path]:
    from PIL import Image
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, rec in enumerate(recordings):
        frames = [Image.fromarray(rec.image(k, cell, header=True)) for k in range(len(rec))]
        p = d / f"{prefix}_{n:03d}.gif"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=max(20, int(delay * 1000)), loop=0)
        print(f"saved {p}")
        paths.append(p)
    return paths
