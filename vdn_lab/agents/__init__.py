from .base import Agent
from .recurrent_dqn import RecurrentDQNAgent, DQNConfig, lambda_returns
from .independent import IndependentAgent
from .vdn import VDNAgent
from .centralized import CentralizedAgent, CentralizedVDNAgent
from .baselines import RandomAgent
from ..registry import AGENTS


def make_agent(name: str, **kwargs) -> Agent:
    return AGENTS.create(name, **kwargs)


__all__ = ["Agent", "RecurrentDQNAgent", "DQNConfig", "lambda_returns", "IndependentAgent", "VDNAgent",
           "CentralizedAgent", "CentralizedVDNAgent", "RandomAgent", "make_agent"]
