import matplotlib

matplotlib.use("Agg")

from vdn_lab.envs import make_env  # noqa: E402
from vdn_lab.agents import AGENTS  # noqa: F401,E402
from vdn_lab.experiment import Experiment, ExperimentConfig, ReplayViewer, SessionRecorder, run_episode  # noqa: E402
from vdn_lab.registry import AGENTS as REG  # noqa: E402


def recordings(n=2, steps=12, every=1):
    env = make_env("switch", max_steps=steps, seed=0)
    agent = REG.create("random", **env.spec, seed=0)
    rec = SessionRecorder(every=every)
    for k in range(n):
        run_episode(env, agent, train=False, explore=True, renderer=rec, title=f"s{k}")
    return rec.recordings


def test_recorder_keeps_every_frame_and_the_last_one():
    recs = recordings(n=2, steps=12)
    assert [len(r) for r in recs] == [13, 13]                      # step 0 .. 12
    assert recs[0].infos[-1]["step"] == 12
    assert len(recordings(n=1, steps=12, every=5)[0]) == 4         # 0, 5, 10, 12
    assert recs[0].image(0).ndim == 3


def test_viewer_pause_replay_speed_and_sessions():
    v = ReplayViewer(recordings(n=2, steps=12), delay=0.1)
    assert not v.paused and v.k == 0
    v._tick(); v._tick()
    assert v.k == 2
    v.toggle_pause(); v._tick()
    assert v.paused and v.k == 2                                  # paused -> no progress
    v.step(1)
    assert v.k == 3
    v.replay()
    assert v.k == 0 and not v.paused
    v.set_speed(4.0)
    assert v.speed == 4.0 and v.timer.interval == 25
    v.faster(); assert v.speed == 8.0
    v.faster(-3); assert v.speed == 1.0
    for _ in range(13):                                           # run off the end -> next session
        v._tick()
    assert v.session == 1 and v.k == 0
    for _ in range(20):
        v._tick()
    assert v.paused and v.at_end and "finished" in v.status_text()
    v.toggle_pause()                                              # play on finished last session restarts it
    assert v.k == 0 and not v.paused
    v.goto_session(0)
    assert v.session == 0
    v._on_scrub(7)
    assert v.k == 7 and v.paused
    v.close()


def test_window_visual_without_display_falls_back_to_gifs(tmp_path, capsys):
    cfg = ExperimentConfig(name="t", env="switch", episode_len=20, eval_episode_len=10, agent="random",
                           train_steps=20, eval_every=0, final_eval_sessions=2, eval_visual="window",
                           out_dir=str(tmp_path), verbose=0)
    Experiment(cfg).run()
    assert len(list(tmp_path.rglob("*.gif"))) == 2
    assert "no interactive display" in capsys.readouterr().out
