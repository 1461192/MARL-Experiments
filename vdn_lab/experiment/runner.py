"""Experiment runner: train headless (or with visuals), evaluate periodically, then run N
sessions with the finished agent. Algorithm- and game-agnostic: it only uses the ``Agent``
and ``GridWorld`` interfaces plus the registries."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agents.base import Agent
from ..envs import make_env
from ..registry import AGENTS
from .config import ExperimentConfig
from .render import NullRenderer, Renderer, make_renderer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_episode(env, agent: Agent, *, train: bool, explore: Optional[bool] = None,
                renderer: Optional[Renderer] = None, title: str = "") -> Dict[str, Any]:
    """Play one full episode. ``train=True`` feeds transitions to the agent and calls
    ``agent.update()`` every step; otherwise the agent is frozen (and acts greedily unless
    ``explore`` says otherwise)."""
    explore = train if explore is None else explore
    renderer = renderer or NullRenderer()
    agent.set_training(train)
    obs = env.reset()
    agent.reset_episode()
    renderer.begin_episode(env, title)
    total, per_agent, steps, done = 0.0, np.zeros(env.n_agents), 0, False
    losses: List[float] = []
    while not done:
        actions = agent.act(obs, explore=explore)
        nxt, reward, done, info = env.step(actions)
        if train:
            agent.observe(obs, actions, reward, nxt, done)
            m = agent.update()
            if "loss" in m:
                losses.append(m["loss"])
        obs = nxt
        total += reward
        per_agent += np.asarray(info["agent_rewards"])
        steps += 1
        renderer.frame(env, {"step": steps, "reward": reward, "return": total, "actions": actions,
                             "agent_rewards": info["agent_rewards"]})
    renderer.end_episode(f"{title}: return {total:.1f} in {steps} steps")
    return {"return": total, "steps": steps, "agent_returns": per_agent.tolist(),
            "loss": float(np.mean(losses)) if losses else float("nan")}


class Callback:
    """Hook into the training loop (logging, early stopping, custom metrics ...)."""

    def on_train_start(self, exp: "Experiment") -> None: ...
    def on_episode_end(self, exp: "Experiment", rec: Dict[str, Any]) -> None: ...
    def on_eval_end(self, exp: "Experiment", rec: Dict[str, Any]) -> None: ...
    def on_train_end(self, exp: "Experiment", result: Dict[str, Any]) -> None: ...


class ConsoleLogger(Callback):
    def __init__(self, verbose: int = 1):
        self.verbose, self.t0 = verbose, time.time()

    def on_train_start(self, exp):
        if self.verbose:
            print(f"[{exp.cfg.name}] {exp.env.describe()} | {exp.agent.describe()} | "
                  f"params={int(exp.agent.diagnostics().get('n_params', 0))}")

    def on_episode_end(self, exp, rec):
        if self.verbose >= 2:
            print(f"  ep {rec['episode']:4d} step {rec['step']:7d} return {rec['return']:7.1f} "
                  f"eps {rec['epsilon']:.2f} loss {rec['loss']:.4f}")

    def on_eval_end(self, exp, rec):
        if self.verbose:
            print(f"  [eval] step {rec['step']:7d}  return {rec['mean_return']:7.2f} ± {rec['std_return']:.2f}"
                  f"   ({time.time() - self.t0:.0f}s)")


class Experiment:
    def __init__(self, cfg: ExperimentConfig, callbacks: Optional[Sequence[Callback]] = None):
        self.cfg = cfg
        self.callbacks: List[Callback] = list(callbacks) if callbacks is not None else [ConsoleLogger(cfg.verbose)]
        self.run_dir: Optional[Path] = None
        self.build()

    def build(self) -> None:
        c = self.cfg
        set_global_seed(c.seed)
        self.env = make_env(c.env, **{**c.env_kwargs, "max_steps": c.episode_len, "seed": c.seed})
        self.eval_env = make_env(c.env, **{**c.env_kwargs, "max_steps": c.eval_episode_len,
                                           "seed": c.seed + 10_000})
        self.agent: Agent = AGENTS.create(c.agent, **self.env.spec, device=c.device, seed=c.seed,
                                          **c.agent_kwargs)

    def _make_run_dir(self) -> Optional[Path]:
        if not self.cfg.out_dir:
            return None
        d = Path(self.cfg.out_dir) / f"{self.cfg.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        d.mkdir(parents=True, exist_ok=True)
        self.cfg.save(d / "config.json")
        return d

    def _renderer(self, kind: str, prefix: str, live: bool = False) -> Renderer:
        c = self.cfg
        gif_dir = c.visual_dir or str((self.run_dir or Path(".")) / "gifs")
        return make_renderer(kind, delay=c.visual_delay, every=c.visual_every, color=c.visual_color,
                             views=c.visual_views, out_dir=gif_dir, prefix=prefix, live=live)

    def evaluate(self, episodes: int, visual: str = "none", label: str = "eval",
                 explore: bool = False) -> Dict[str, Any]:
        """Run ``episodes`` frozen-agent sessions (greedy by default) on the evaluation env.

        With ``visual="window"`` the sessions are recorded and, once all have run, opened in the
        replay window (pause / replay / speed control); this call returns when it is closed."""
        r = self._renderer(visual, prefix=label)
        results = []
        was_training = self.agent.training
        for k in range(episodes):
            results.append(run_episode(self.eval_env, self.agent, train=False, explore=explore,
                                       renderer=r, title=f"{label} {k + 1}/{episodes}"))
        self.agent.set_training(was_training)
        r.close()
        rets = [x["return"] for x in results]
        return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)), "returns": rets,
                "agent_returns": np.mean([x["agent_returns"] for x in results], axis=0).tolist()}

    def train(self) -> Dict[str, Any]:
        c = self.cfg
        for cb in self.callbacks:
            cb.on_train_start(self)
        train_curve: List[Dict[str, Any]] = []
        eval_curve: List[Dict[str, Any]] = []
        step, episode = 0, 0
        next_eval = c.eval_every if c.eval_every else None
        while step < c.train_steps:
            show = bool(c.train_visual_every) and c.train_visual != "none" and episode % c.train_visual_every == 0
            r = self._renderer(c.train_visual if show else "none", prefix=f"train_ep{episode:04d}", live=True)
            out = run_episode(self.env, self.agent, train=True, renderer=r, title=f"train ep {episode}")
            r.close()
            step += out["steps"]
            episode += 1
            rec = {"episode": episode, "step": step, "return": out["return"], "loss": out["loss"],
                   "epsilon": self.agent.diagnostics().get("epsilon", float("nan"))}
            train_curve.append(rec)
            for cb in self.callbacks:
                cb.on_episode_end(self, rec)
            if next_eval is not None and step >= next_eval:
                ev = self.evaluate(c.eval_episodes, label="eval")
                erec = {"step": step, **{k: ev[k] for k in ("mean_return", "std_return")}}
                eval_curve.append(erec)
                for cb in self.callbacks:
                    cb.on_eval_end(self, erec)
                next_eval += c.eval_every
        rets = np.array([x["return"] for x in train_curve])
        tail = max(1, len(rets) // 10)
        return {"train_curve": train_curve, "eval_curve": eval_curve,
                "train_auc": float(rets.mean()), "train_final": float(rets[-tail:].mean())}

    def run(self) -> Dict[str, Any]:
        """train headless -> save -> run ``final_eval_sessions`` with the finished agent."""
        c = self.cfg
        self.run_dir = self._make_run_dir()
        t0 = time.time()
        result = self.train()
        if c.save_agent and self.run_dir is not None:
            try:
                self.save_agent(self.run_dir / "agent.pt")
            except NotImplementedError:
                pass
        if c.final_eval_sessions > 0:
            if c.verbose:
                print(f"[{c.name}] training done ({time.time() - t0:.0f}s). "
                      f"Running {c.final_eval_sessions} final session(s)...")
            result["final_eval"] = self.evaluate(c.final_eval_sessions, visual=c.eval_visual, label="final")
            if c.verbose:
                fe = result["final_eval"]
                print(f"[{c.name}] final: {fe['mean_return']:.2f} ± {fe['std_return']:.2f}  "
                      f"(per-agent contribution {np.round(fe['agent_returns'], 1).tolist()})")
        result["config"] = c.to_dict()
        result["seconds"] = time.time() - t0
        if self.run_dir is not None:
            (self.run_dir / "result.json").write_text(json.dumps(result, indent=2))
        for cb in self.callbacks:
            cb.on_train_end(self, result)
        return result

    def save_agent(self, path) -> None:
        import torch
        torch.save(self.agent.state_dict(), path)

    def load_agent(self, path) -> None:
        import torch
        self.agent.load_state_dict(torch.load(path, map_location=self.cfg.device, weights_only=False))

    @classmethod
    def from_run_dir(cls, run_dir, **overrides) -> "Experiment":
        """Rebuild an experiment from a saved run and load its trained agent (no training)."""
        run_dir = Path(run_dir)
        cfg = ExperimentConfig.load(run_dir / "config.json").replace(**overrides)
        exp = cls(cfg)
        exp.run_dir = run_dir
        exp.load_agent(run_dir / "agent.pt")
        return exp
