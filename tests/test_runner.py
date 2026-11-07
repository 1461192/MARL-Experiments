from pathlib import Path

import pytest

from vdn_lab import register_agent
from vdn_lab.agents.base import Agent
from vdn_lab.experiment import Benchmark, Experiment, ExperimentConfig


def tiny(**kw):
    base = dict(name="tiny", env="fetch", env_kwargs={"layout": "open"}, episode_len=60, eval_episode_len=30,
                agent="vdn", agent_kwargs=dict(warmup_steps=60, batch_size=4), train_steps=240, eval_every=120,
                eval_episodes=1, final_eval_sessions=2, verbose=0)
    base.update(kw)
    return ExperimentConfig(**base)


def test_train_headless_then_sessions_with_text_visuals(tmp_path, capsys):
    cfg = tiny(out_dir=str(tmp_path), eval_visual="text", visual_delay=0, visual_color=False)
    res = Experiment(cfg).run()
    out = capsys.readouterr().out
    assert "final 1/2" in out and "final 2/2" in out          # the visual sessions ran
    assert "train ep" not in out                               # training stayed headless
    assert len(res["final_eval"]["returns"]) == 2
    run = next(Path(tmp_path).iterdir())
    assert (run / "config.json").exists() and (run / "result.json").exists() and (run / "agent.pt").exists()


def test_gif_visuals(tmp_path):
    cfg = tiny(out_dir=str(tmp_path), eval_visual="gif", final_eval_sessions=1, train_steps=120, eval_every=0)
    Experiment(cfg).run()
    assert list(Path(tmp_path).rglob("*.gif"))


def test_reload_trained_agent_and_play(tmp_path):
    Experiment(tiny(out_dir=str(tmp_path), train_steps=120, eval_every=0)).run()
    run = next(Path(tmp_path).iterdir())
    assert "mean_return" in Experiment.from_run_dir(run).evaluate(1)


def test_custom_agent_plugs_in_without_touching_the_runner():
    @register_agent("always_forward_test")
    class AlwaysForward(Agent):
        def reset_episode(self): pass
        def act(self, obs, explore=True): return [0] * self.n_agents
        def observe(self, *a, **k): pass
        def update(self): return {}

    cfg = tiny(agent="always_forward_test", agent_kwargs={}, out_dir=None, train_steps=60, eval_every=0)
    assert "final_eval" in Experiment(cfg).run()


def test_benchmark_variants_with_game_settings(tmp_path):
    base = tiny(out_dir=None, train_steps=120, eval_every=0, final_eval_sessions=1)
    bench = Benchmark(base, variants={"vdn": {"agent": "vdn"},
                                      "vdn_fov2": {"agent": "vdn", "env_kwargs": {"view_ahead": 2, "view_side": 1}},
                                      "ind": {"agent": "independent"}},
                      seeds=[0, 1], out_dir=str(tmp_path))
    rows = bench.run()
    assert {r["variant"] for r in rows} == {"vdn", "vdn_fov2", "ind"}
    assert (bench.out / "summary.csv").exists()


def test_config_roundtrip(tmp_path):
    cfg = tiny()
    cfg.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json") == cfg
    assert cfg.replace(env_kwargs={"view_side": 1}).env_kwargs == {"layout": "open", "view_side": 1}
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"nope": 1})
