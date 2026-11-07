from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one run. JSON-serialisable."""

    name: str = "exp"
    seed: int = 0
    device: str = "cpu"

    # ---- game -------------------------------------------------------------------------
    env: str = "fetch"                       # fetch | switch | checkers | <anything registered>
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    #   e.g. {"layout": "single", "view_ahead": 4, "view_side": 2, "view_behind": 0}
    episode_len: int = 500                   # training episode length  (paper: 5000)
    eval_episode_len: int = 500              # evaluation episode length (paper: 2000)

    # ---- algorithm --------------------------------------------------------------------
    agent: str = "vdn"                       # vdn | independent | centralized | centralized_vdn | random
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ---- training budget & periodic evaluation ---------------------------------------
    train_steps: int = 50_000                # total environment steps
    eval_every: int = 10_000                 # steps between greedy evaluations (0 = never)
    eval_episodes: int = 3

    # ---- after training: run N sessions with the finished agent -----------------------
    final_eval_sessions: int = 5

    # ---- visuals ("none" | "text" | "window" | "gif") ---------------------------------
    train_visual: str = "none"               # show *training* episodes ...
    train_visual_every: int = 0              # ... every N episodes (0 = never; train headless)
    eval_visual: str = "none"                # show the final evaluation sessions
    visual_delay: float = 0.08               # seconds per frame (text/window); GIF frame duration
    visual_every: int = 1                    # render every k-th env step
    visual_color: bool = True                # ANSI colours in text mode
    visual_views: bool = True                # also show each agent's own observation window
    visual_dir: Optional[str] = None         # where gifs go (default: <run dir>/gifs)

    # ---- bookkeeping -----------------------------------------------------------------
    out_dir: Optional[str] = "runs"          # None -> write nothing
    save_agent: bool = True
    verbose: int = 1                         # 0 silent, 1 summaries, 2 every episode

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise KeyError(f"unknown config fields: {sorted(bad)}")
        return cls(**copy.deepcopy(d))

    def replace(self, **overrides: Any) -> "ExperimentConfig":
        """Copy with overrides; ``env_kwargs`` / ``agent_kwargs`` are *merged*, not replaced."""
        d = self.to_dict()
        for k, v in overrides.items():
            if k in ("env_kwargs", "agent_kwargs"):
                d[k] = {**d[k], **v}
            else:
                d[k] = v
        return ExperimentConfig.from_dict(d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
