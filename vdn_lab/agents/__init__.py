from .base import Agent
from .recurrent_dqn import RecurrentDQNAgent, DQNConfig, lambda_returns
from .independent import IndependentAgent
from .vdn import VDNAgent
from .centralized import CentralizedAgent, CentralizedVDNAgent
from .mixing import MixingAgent, QMIXAgent, QPLEXAgent
from .mappo import MAPPOAgent, IPPOAgent, PPOConfig
from .macpf import MACPFAgent, MACPFConfig
from .atd import ATDConfig, AdaptiveLambda, DensityRatioEstimator, VDNATDAgent, QMIXATDAgent, QPLEXATDAgent, MAPPOATDAgent
from .baselines import RandomAgent
from ..registry import AGENTS


def make_agent(name: str, **kwargs) -> Agent:
    return AGENTS.create(name, **kwargs)


__all__ = ["Agent", "RecurrentDQNAgent", "DQNConfig", "lambda_returns", "IndependentAgent", "VDNAgent",
           "CentralizedAgent", "CentralizedVDNAgent", "MixingAgent", "QMIXAgent", "QPLEXAgent", "RandomAgent",
           "MAPPOAgent", "IPPOAgent", "PPOConfig", "MACPFAgent", "MACPFConfig", "ATDConfig", "AdaptiveLambda",
           "DensityRatioEstimator", "VDNATDAgent", "QMIXATDAgent", "QPLEXATDAgent", "MAPPOATDAgent", "make_agent"]
