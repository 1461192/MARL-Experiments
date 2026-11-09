import numpy as np
import pytest

from vdn_lab import ENVS
from vdn_lab.envs import make_env
from vdn_lab.experiment import Experiment, ExperimentConfig


@pytest.mark.parametrize("name,layout", [("matrix", g) for g in ENVS.get("matrix").LAYOUTS] +
                         [("two_stage_matrix", g) for g in ENVS.get("two_stage_matrix").LAYOUTS])
def test_all_games_run(name, layout):
    env = make_env(name, layout=layout, max_steps=11, seed=0)
    obs = env.reset()
    assert obs.shape == (2, *env.obs_shape) and obs.dtype == np.uint8
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, r, done, info = env.step(rng.integers(env.n_actions, size=2))
        assert obs.shape == (2, *env.obs_shape)
        assert len(info["agent_rewards"]) == 2 and abs(sum(info["agent_rewards"]) - r) < 1e-9
    assert env.t == 11
    env.render_text()
    assert env.render_rgb().ndim == 3


def test_matrix_game_row_column_payoffs():
    env = make_env("matrix", layout="climbing")
    assert env.n_actions == 3 and env.obs_shape == (1, 1, 1)
    assert env.step([0, 0])[1] == 11
    assert env.step([0, 1])[1] == -30
    _, r, _, info = env.step([1, 2])                    # row b, column c
    assert r == 6 and info["cell"] == (1, 2) and info["agent_rewards"] == [3, 3]


def test_general_sum_payoffs_per_player():
    env = make_env("matrix", layout="prisoners_dilemma")
    _, r, _, info = env.step([1, 0])                    # agent 0 defects, agent 1 cooperates
    assert info["agent_rewards"] == [5, 0] and r == 5


def test_custom_non_square_matrix_wraps_actions():
    env = make_env("matrix", payoffs=[[1, 2, 3]])       # 1 row, 3 columns
    assert env.n_actions == 3
    assert env.step([2, 2])[1] == 3                     # row 2 -> row 0


def test_two_stage_choice_then_matrix():
    env = make_env("two_stage_matrix", layout="two_step")
    obs = env.reset()
    assert env.n_actions == 2 and env.obs_shape == (1, 1, 4)
    assert (obs[:, 0, 0] == [[255, 0, 0, 0]] * 2).all()           # both: choosing stage
    obs, r, _, info = env.step([1, 0])                  # agent 0 picks matrix 1, agent 1 ignored
    assert r == 0 and info["matrix"] == 1
    assert (obs[:, 0, 0] == [[0, 255, 0, 255]] * 2).all()         # playing matrix 1, both see it
    obs, r, _, info = env.step([1, 1])
    assert r == 8 and info["cell"] == (1, 1)
    assert (obs[:, 0, 0, 0] == 255).all()                         # back to choosing
    env.step([0, 1])
    assert env.step([0, 1])[1] == 7                     # matrix 0 always pays 7


def test_two_stage_hidden_choice_and_choice_rewards():
    env = make_env("two_stage_matrix", payoffs=[[[1]], [[2, 0], [0, 2]]], choice_rewards=[0.5, 0],
                   observe_choice=False)
    obs, r, *_ = env.step([0, 1])
    assert r == 0.5
    assert obs[0, 0, 0, 2] == 255 and obs[1, 0, 0, 2:].sum() == 0  # agent 1 does not see the choice
    assert obs[1, 0, 0, 1] == 255                                  # ... only that a matrix is played
    assert env.step([1, 1])[1] == 1                     # 1x1 matrix: actions wrap


def test_bad_inputs():
    with pytest.raises(KeyError):
        make_env("matrix", layout="nope")
    with pytest.raises(ValueError):
        make_env("matrix", payoffs=[1, 2, 3])
    with pytest.raises(ValueError):
        make_env("two_stage_matrix", choice_rewards=[1.0])


@pytest.mark.parametrize("env,agent", [("matrix", "vdn"), ("two_stage_matrix", "centralized")])
def test_runner_trains_and_renders_matrix_games(tmp_path, env, agent):
    cfg = ExperimentConfig(name="m", env=env, episode_len=40, eval_episode_len=20, agent=agent,
                           agent_kwargs=dict(warmup_steps=40, batch_size=4), train_steps=120, eval_every=0,
                           final_eval_sessions=1, eval_visual="gif", out_dir=str(tmp_path), verbose=0)
    res = Experiment(cfg).run()
    assert len(res["final_eval"]["returns"]) == 1
    assert list(tmp_path.rglob("*.gif"))
