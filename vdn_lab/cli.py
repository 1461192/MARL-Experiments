"""Command line interface:  python -m vdn_lab <command> ...

  list      show registered agents / games / maps
  watch     look at a game with random actions (or a saved run via --run)
  train     train one agent (optionally headless, then N visual sessions)
  compare   several agents/variants x several seeds, paper-style summary
  play      re-run a saved run directory with visuals (no training)
"""
from __future__ import annotations

import argparse
import ast
from typing import Any, Dict, List

from . import AGENTS, ENVS
from .experiment import Benchmark, Experiment, ExperimentConfig, make_renderer, run_episode


def _kv(items: List[str]) -> Dict[str, Any]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _add_game_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("game settings")
    g.add_argument("--env", default="fetch", help="game name (see `list`)")
    g.add_argument("--map", dest="layout", default=None, help="named layout of the game")
    g.add_argument("--view-ahead", type=int, default=None, help="squares visible ahead (paper: 4)")
    g.add_argument("--view-side", type=int, default=None, help="squares visible each side (paper: 2)")
    g.add_argument("--view-behind", type=int, default=None, help="squares visible behind (paper: 0)")
    g.add_argument("--episode-len", type=int, default=None, help="training episode length (paper: 5000)")
    g.add_argument("--eval-episode-len", type=int, default=None, help="test episode length (paper: 2000)")
    g.add_argument("--env-kw", nargs="*", default=[], metavar="K=V", help="extra game kwargs")


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("algorithm")
    g.add_argument("--agent", default="vdn", help="agent name (see `list`)")
    g.add_argument("--share-weights", action="store_true", help="agent-invariant shared network")
    g.add_argument("--role-info", action="store_true", help="one-hot agent id as extra input")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--agent-kw", nargs="*", default=[], metavar="K=V",
                   help="extra agent kwargs, e.g. eps_decay_steps=20000")


def _add_visual_opts(p) -> None:
    p.add_argument("--delay", type=float, default=0.08, help="seconds per frame")
    p.add_argument("--render-every", type=int, default=1, help="render every k-th step")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--no-views", action="store_true", help="hide the agents' observation windows")
    p.add_argument("--gif-dir", default=None)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("training / evaluation")
    g.add_argument("--steps", type=int, default=50_000, help="training environment steps")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")
    g.add_argument("--eval-every", type=int, default=10_000, help="steps between greedy evals (0 = never)")
    g.add_argument("--eval-episodes", type=int, default=3)
    g.add_argument("--final-sessions", type=int, default=5, help="sessions to run after training")
    g.add_argument("--out", default="runs", help="output dir ('' = write nothing)")
    g.add_argument("--name", default=None)
    g.add_argument("--quiet", action="store_true")
    v = p.add_argument_group("visuals")
    v.add_argument("--train-visual", default="none", choices=["none", "text", "window", "gif"])
    v.add_argument("--train-visual-every", type=int, default=0, help="render every N-th training episode")
    v.add_argument("--eval-visual", default="none", choices=["none", "text", "window", "gif"],
                   help="how to show the final sessions (window = replay window with pause / replay / speed)")
    _add_visual_opts(v)


def _config(a: argparse.Namespace) -> ExperimentConfig:
    env_kw = _kv(a.env_kw)
    for k in ("layout", "view_ahead", "view_side", "view_behind"):
        if getattr(a, k, None) is not None:
            env_kw[k] = getattr(a, k)
    agent_kw = _kv(a.agent_kw)
    if getattr(a, "share_weights", False):
        agent_kw["share_weights"] = True
    if getattr(a, "role_info", False):
        agent_kw["role_info"] = True
    if getattr(a, "lr", None) is not None:
        agent_kw["lr"] = a.lr
    cfg = ExperimentConfig(
        name=a.name or f"{a.agent}_{a.env}_{env_kw.get('layout', 'default')}",
        env=a.env, env_kwargs=env_kw, agent=a.agent, agent_kwargs=agent_kw,
        train_steps=a.steps, seed=a.seed, device=a.device, eval_every=a.eval_every,
        eval_episodes=a.eval_episodes, final_eval_sessions=a.final_sessions,
        out_dir=a.out or None, verbose=0 if a.quiet else 1,
        train_visual=a.train_visual, train_visual_every=a.train_visual_every, eval_visual=a.eval_visual,
        visual_delay=a.delay, visual_every=a.render_every, visual_color=not a.no_color,
        visual_views=not a.no_views, visual_dir=a.gif_dir)
    if a.episode_len:
        cfg.episode_len = a.episode_len
        cfg.eval_episode_len = a.eval_episode_len or a.episode_len
    elif a.eval_episode_len:
        cfg.eval_episode_len = a.eval_episode_len
    return cfg


def cmd_list(a) -> None:
    print("agents:", ", ".join(AGENTS.names()))
    print("games:")
    for n in ENVS.names():
        cls = ENVS.get(n)
        print(f"  {n:16s} maps: {', '.join(cls.LAYOUTS)} (default: {cls.default_layout_name})")


def cmd_watch(a) -> None:
    if a.run:
        exp = Experiment.from_run_dir(a.run)
        explore = False
    else:
        exp = Experiment(_config(a).replace(agent="random", agent_kwargs={}), callbacks=[])
        explore = True
    r = make_renderer(a.eval_visual, delay=a.delay, every=a.render_every, color=not a.no_color,
                      views=not a.no_views, out_dir=a.gif_dir or "gifs")
    for k in range(a.sessions):
        run_episode(exp.eval_env, exp.agent, train=False, explore=explore, renderer=r, title=f"watch {k + 1}")
    r.close()


def cmd_train(a) -> None:
    Experiment(_config(a)).run()


def cmd_compare(a) -> None:
    base = _config(a)
    variants: Dict[str, Dict[str, Any]] = {}
    for spec in a.agents:                      # "vdn", "vdn+share", "vdn+share+role", ...
        name, *flags = spec.split("+")
        kw = {}
        if "share" in flags:
            kw["share_weights"] = True
        if "role" in flags:
            kw["role_info"] = True
        variants[spec] = {"agent": name, "agent_kwargs": kw}
    Benchmark(base, variants, seeds=a.seeds, out_dir=a.out or None, name=a.name or f"compare_{a.env}").run()


def cmd_play(a) -> None:
    exp = Experiment.from_run_dir(a.run)
    ev = exp.evaluate(a.sessions, visual=a.visual, label="play")
    print(f"mean return {ev['mean_return']:.2f} ± {ev['std_return']:.2f} over {a.sessions} sessions")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vdn_lab", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)

    w = sub.add_parser("watch", help="render a game with random actions (or a saved run via --run)")
    _add_game_args(w)
    w.add_argument("--run", default=None, help="saved run directory to replay greedily")
    w.add_argument("--sessions", type=int, default=1)
    w.add_argument("--eval-visual", default="window", choices=["text", "window", "gif"],
                   help="window = replay window with pause / replay / speed controls")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--device", default="cpu")
    w.add_argument("--steps", type=int, default=0)
    w.add_argument("--eval-every", type=int, default=0)
    w.add_argument("--eval-episodes", type=int, default=1)
    w.add_argument("--final-sessions", type=int, default=0)
    w.add_argument("--out", default="")
    w.add_argument("--name", default="watch")
    w.add_argument("--quiet", action="store_true")
    w.add_argument("--train-visual", default="none")
    w.add_argument("--train-visual-every", type=int, default=0)
    w.add_argument("--agent", default="random")
    w.add_argument("--agent-kw", nargs="*", default=[])
    _add_visual_opts(w)
    w.set_defaults(fn=cmd_watch)

    t = sub.add_parser("train", help="train one agent")
    _add_game_args(t)
    _add_agent_args(t)
    _add_run_args(t)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("compare", help="compare agents over seeds")
    _add_game_args(c)
    _add_agent_args(c)
    c.add_argument("--agents", nargs="+", default=["vdn", "independent", "centralized"],
                   help="e.g. vdn vdn+share+role independent centralized centralized_vdn")
    c.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    _add_run_args(c)
    c.set_defaults(fn=cmd_compare, out="runs/benchmarks")

    pl = sub.add_parser("play", help="run sessions of a saved run (no training)")
    pl.add_argument("run", help="run directory containing config.json and agent.pt")
    pl.add_argument("--sessions", type=int, default=3)
    pl.add_argument("--visual", default="window", choices=["none", "text", "window", "gif"],
                    help="window (default) = replay window with pause / replay / speed controls")
    pl.set_defaults(fn=cmd_play)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
