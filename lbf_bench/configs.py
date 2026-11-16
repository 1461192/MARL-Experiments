"""Hyper-parameters per algorithm for LBF.

Precedence of sources (see README "LBF benchmark" for the full table and the reasons):

1. Papoudakis et al., *Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks*
   (NeurIPS 2021 Datasets & Benchmarks, arXiv:2006.07869v4), sec. 4 and appendix I: the LBF values its grid
   search chose for QMIX (table 28) and MAPPO (table 24), with parameter sharing; 2M steps for off-policy and
   20M for on-policy algorithms; 41 evaluations of 100 episodes. Its implementation is EPyMARL, which fills in
   the structural details (episode batches, Double-Q, n-step critic targets, target critic, ...).
2. The algorithm's own paper, for algorithms the benchmark does not include:
   * QPLEX (Wang et al., ICLR 2021, appendix B.2 / table 2): PyMARL defaults (RMSprop lr 5e-4, GRU 64, batch 32,
     5000-episode buffer, epsilon 1 -> 0.05 over 50k steps, target update every 200 episodes), two gradient
     updates per collected episode, SMAC mixer configuration (1 layer, 4 heads, stop-gradient on A_i).
   * MACPF (Wang, Ye & Lu, ICLR 2023, appendix C.2): see ``macpf.py``.
   * ATD(lambda) (Deng, Wang & Zhang, IJCAI 2026, table 3 and sec. 4): estimator MLP-GRU(64), Adam lr 1e-3,
     buffer ratio 50x, an estimator step and a lambda-cache clear every target-update period. Each ATD variant
     otherwise keeps its base learner's hyper-parameters.
3. Fixed-lambda controls: QMIX / QPLEX with lambda = 0.4 (the ATD paper's QMIX / QPLEX baselines, table 3);
   MAPPO with critic lambda-returns at lambda = 0.95 (MAPPO's GAE lambda, Yu et al. 2022).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

from .algos import ACLearner, Algo, QLearner
from .macpf import MACPF

OFF_T_MAX = 2_000_000
ON_T_MAX = 20_000_000


@dataclass
class AlgoSpec:
    cls: Type[Algo]
    kwargs: Dict[str, Any]
    n_envs: int = 1
    t_max: int = OFF_T_MAX
    label: str = ""
    base: str = ""


ALGOS: Dict[str, AlgoSpec] = {
    # benchmark paper, table 28 (QMIX, parameter sharing, LBF)
    "qmix": AlgoSpec(QLearner, dict(hidden=64, lr=3e-4, optim="adam", mixer="qmix", eps_anneal=200_000,
                                    eval_eps=0.05, target=0.01, std_rewards=True), label="QMIX", base="qmix"),
    # QPLEX paper, appendix B.2 + PyMARL defaults
    "qplex": AlgoSpec(QLearner, dict(hidden=64, lr=5e-4, optim="rmsprop", mixer="qplex", qplex_heads=4,
                                     qplex_layers=1, eps_anneal=50_000, eval_eps=0.0, target=200,
                                     std_rewards=False, updates_per_episode=2), label="QPLEX", base="qplex"),
    # benchmark paper, table 24 (MAPPO, parameter sharing, LBF); EPyMARL: 10 parallel episodes per update
    "mappo": AlgoSpec(ACLearner, dict(hidden=128, lr=3e-4, use_rnn=False, critic="central", entropy=0.001, nstep=5,
                                      target=0.01, std_rewards=False, epochs=4, ppo_clip=0.2), n_envs=10,
                      t_max=ON_T_MAX, label="MAPPO", base="mappo"),
    # MACPF paper, appendix C.2
    "macpf": AlgoSpec(MACPF, dict(), label="MACPF", base="macpf"),
}

for _b, _ctl, _kw in (("qmix", "qmix_l0.4", dict(td_lambda=0.4)), ("qplex", "qplex_l0.4", dict(td_lambda=0.4)),
                      ("mappo", "mappo_l0.95", dict(critic_lambda=0.95))):
    _s = ALGOS[_b]
    ALGOS[f"{_b}_atd"] = AlgoSpec(_s.cls, {**_s.kwargs, "atd": True}, _s.n_envs, _s.t_max,
                                  label=f"{_s.label}-ATD", base=_b)
    lam = list(_kw.values())[0]
    ALGOS[_ctl] = AlgoSpec(_s.cls, {**_s.kwargs, **_kw}, _s.n_envs, _s.t_max,
                           label=f"{_s.label} (λ={lam})", base=_b)

STUDY_ALGOS = ["qmix", "qmix_atd", "qmix_l0.4", "qplex", "qplex_atd", "qplex_l0.4",
               "mappo", "mappo_atd", "mappo_l0.95", "macpf"]
