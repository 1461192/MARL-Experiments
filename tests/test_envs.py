import numpy as np
import pytest

from vdn_lab.envs import make_env
from vdn_lab.envs.base import FWD, NOOP, N_ACTIONS, BEAM, C_WALL, C_YELLOW


def colours(view):
    return {tuple(int(u) for u in view[:, y, x]) for y in range(view.shape[1]) for x in range(view.shape[2])}


@pytest.mark.parametrize("name,layout", [("switch", m) for m in ("open", "single", "double")] +
                         [("fetch", m) for m in ("open", "single", "double")] +
                         [("checkers", m) for m in ("default", "mixed")])
def test_all_maps_run(name, layout):
    env = make_env(name, layout=layout, max_steps=40, seed=0)
    obs = env.reset()
    assert obs.shape == (2, 3, 5, 5) and obs.dtype == np.uint8
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, r, done, info = env.step(rng.integers(N_ACTIONS, size=2))
        assert obs.shape == (2, 3, 5, 5)
        assert len(info["agent_rewards"]) == 2 and abs(sum(info["agent_rewards"]) - r) < 1e-9
    assert env.t == 40
    env.render_text(color=False)
    assert env.render_rgb().ndim == 3


def test_configurable_field_of_view():
    env = make_env("switch", layout="open", view_ahead=2, view_side=1, view_behind=1)
    assert env.obs_shape == (3, 4, 3)
    assert env.reset().shape == (2, 3, 4, 3)


def test_view_is_egocentric():
    env = make_env("switch", layout="open", view_ahead=2, view_side=2, seed=0)
    # open room: rows 1-4 are floor, rows 0 and 5 are walls. Stand on the bottom floor row.
    env.pos[0], env.dirs[0] = (4, 5), 1                # facing east
    v = env._view(0)
    assert tuple(v[:, 2, 2]) == (0, 0, 255)            # self in blue at (view_ahead, view_side)
    assert tuple(v[:, 2, 3]) == C_WALL                 # right of an east-facing agent = south = wall
    assert tuple(v[:, 2, 1]) == (0, 0, 0)              # left = north = floor
    env.dirs[0] = 3                                    # facing west: left/right swap
    v = env._view(0)
    assert tuple(v[:, 2, 1]) == C_WALL and tuple(v[:, 2, 3]) == (0, 0, 0)


def test_beam_lights_squares_ahead_only_visually():
    env = make_env("switch", layout="open", seed=0)
    env.reset()
    obs, r, *_ = env.step([BEAM, NOOP])
    assert C_YELLOW in colours(obs[0]) and r == 0.0
    obs, *_ = env.step([NOOP, NOOP])
    assert C_YELLOW not in colours(obs[0])


def test_collisions_block_movement():
    env = make_env("fetch", layout="open", seed=0)
    env.reset()
    env.pos = [(2, 3), (2, 4)]
    env.dirs = [1, 3]
    env.step([FWD, FWD])
    assert env.pos[0] == (2, 3) and env.pos[1] == (2, 4)


def test_fetch_cycle_rewards():
    env = make_env("fetch", layout="open", seed=0)
    env.reset()
    P, D = env.pickup, env.drop
    env.pos, env.dirs = [(P[0], P[1] - 1), (3, 1)], [1, 1]          # agent 0 one step west of P
    _, r, *_ = env.step([FWD, NOOP])
    assert r == 3.0 and env.carrying == [True, False] and not env.item_available      # pick-up = +3
    _, r, *_ = env.step([NOOP, NOOP])
    assert r == 0.0
    env.pos[0], env.dirs[0] = (D[0], D[1] + 1), 3                  # one step east of D, facing west
    _, r, *_ = env.step([FWD, NOOP])
    assert r == 5.0 and env.item_available and env.carrying == [False, False]         # drop-off = +5


def test_fetch_item_is_taken_until_dropped():
    env = make_env("fetch", layout="open", seed=0)
    env.reset()
    P = env.pickup
    env.pos, env.dirs = [(P[0], P[1] - 1), (P[0] + 1, P[1])], [1, 0]
    env.step([FWD, NOOP])                                               # agent 0 takes the item
    env.pos[0] = (P[0] - 1, P[1] + 1)
    _, r, *_ = env.step([NOOP, FWD])                                    # agent 1 steps onto P: nothing there
    assert r == 0.0 and not env.carrying[1]


def test_switch_goals_and_reset():
    env = make_env("switch", layout="open", seed=0)
    env.reset()
    env.pos = [(4, 8), (4, 2)]                      # one step from their own goals
    env.dirs = [1, 3]
    _, r, *_ = env.step([FWD, NOOP])
    assert r == 1.0 and env.reached == [True, False]
    _, r, *_ = env.step([NOOP, FWD])
    assert r == 1.0
    assert env.pos == env.starts and env.reached == [False, False]          # full reset


def test_switch_goal_only_visible_to_owner():
    env = make_env("switch", layout="open", seed=0)
    env.reset()
    env.pos = [(4, 8), (3, 8)]
    env.dirs = [1, 1]
    green = (0, 220, 0)
    assert green in colours(env._view(0))      # agent 0 sees its goal at (4,9)
    assert green not in colours(env._view(1))  # agent 1's goal is elsewhere


def test_checkers_rewards():
    env = make_env("checkers", seed=0)
    env.reset()
    env.pos = [(1, 4), (3, 4)]                       # both adjacent (west) to the lemon wall at col 5
    env.dirs = [1, 1]
    _, r, _, info = env.step([FWD, FWD])
    assert info["agent_rewards"] == [-10.0, -1.0] and r == -11.0


def test_checkers_resets_when_apples_gone():
    env = make_env("checkers", seed=0)
    env.reset()
    for p, k in list(env.items.items()):
        if k == "apple":
            del env.items[p]
    env.items[(1, 9)] = "apple"
    env.pos, env.dirs = [(1, 8), (3, 1)], [1, 1]
    _, r, *_ = env.step([FWD, NOOP])
    assert r == 10.0
    assert sum(k == "apple" for k in env.items.values()) == 6 and env.pos == env.starts


def test_max_steps_sets_done():
    env = make_env("fetch", max_steps=3)
    env.reset()
    assert [env.step([NOOP, NOOP])[2] for _ in range(3)] == [False, False, True]
