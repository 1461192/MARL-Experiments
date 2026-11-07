"""vdn_lab: a light, StarCraft-free testbed for Value-Decomposition Networks
(Sunehag et al., 2017) and related cooperative MARL baselines."""
from .registry import AGENTS, ENVS, register_agent, register_env
from . import envs as _envs      # noqa: F401  (registers games)
from . import agents as _agents  # noqa: F401  (registers agents)

__all__ = ["AGENTS", "ENVS", "register_agent", "register_env"]
