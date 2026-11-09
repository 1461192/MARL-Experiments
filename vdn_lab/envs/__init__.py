from .base import GridWorld, ACTION_NAMES, N_ACTIONS
from .switch import Switch
from .fetch import Fetch
from .checkers import Checkers
from .matrix import MatrixGame, TwoStageMatrixGame
from ..registry import ENVS


def make_env(name: str, **kwargs):
    return ENVS.create(name, **kwargs)


__all__ = ["GridWorld", "Switch", "Fetch", "Checkers", "MatrixGame", "TwoStageMatrixGame", "make_env", "ACTION_NAMES", "N_ACTIONS"]
