"""QMIX and QPLEX: mixer properties, plumbing, and the matrix games that separate them.

* QMIX's two-step game (Rashid et al., 2018, sec. 6.1): VDN's additive fit undervalues matrix 1
  and settles for the safe 7, QMIX's state-conditioned monotonic mixer learns the 8.
* The non-monotonic matrix (QTRAN sec. 5.1, also QPLEX's didactic game): no monotonic mixer can
  fit it, so QMIX misses the 8 at (A, A); QPLEX's duplex dueling mixer can and does.

Like in both papers, the learners see a *uniform* data distribution (epsilon = 1 throughout)
and 1-step TD targets (``lam=0``); the greedy policy is evaluated afterwards.
A representation-only test (mixers regressed straight onto the payoff table) backs this up.
"""
import itertools

import numpy as np
import pytest
import torch

from vdn_lab.agents import make_agent
from vdn_lab.agents.mixers import QMixer, QPLEXMixer
from vdn_lab.envs.matrix import MATRIX_GAMES
from vdn_lab.experiment import Experiment, ExperimentConfig
from vdn_lab.experiment.matrix_study import fit_payoffs, run_seeds, train_uniform

SPEC = dict(n_agents=2, n_actions=8, obs_shape=(3, 5, 5))


# ----------------------------------------------------------------------------- mixer properties
def test_qmix_mixer_is_monotonic_in_every_agent_q():
    torch.manual_seed(0)
    mixer = QMixer(3, state_dim=10)
    q = torch.randn(64, 1, 3, 4, requires_grad=True)
    actions = torch.randint(4, (64, 1, 3))
    mixer(q, actions, torch.randn(64, 1, 10)).sum().backward()
    assert (q.grad >= 0).all()
    assert q.grad.gather(-1, actions.unsqueeze(-1)).gt(0).any()


@pytest.mark.parametrize("mixer_cls", [QMixer, QPLEXMixer])
def test_greedy_individual_actions_maximise_q_tot(mixer_cls):
    """IGM: the per-agent argmax is the argmax of Q_tot over all joint actions."""
    torch.manual_seed(1)
    n, nA, S = 3, 4, 6
    mixer = mixer_cls(n, nA, S) if mixer_cls is QPLEXMixer else mixer_cls(n, S)
    q, state = torch.randn(20, 1, n, nA), torch.randn(20, 1, S)
    joint = torch.tensor(list(itertools.product(range(nA), repeat=n)))           # (nA**n, n)
    with torch.no_grad():
        all_q = torch.stack([mixer(q, j.expand(20, 1, n), state) for j in joint]).squeeze(-1)
        greedy = mixer(q, q.argmax(-1), state).squeeze(-1)
    assert torch.allclose(greedy, all_q.max(0).values, atol=1e-5)


def test_qplex_lambdas_positive_and_detach_trick_same_value():
    torch.manual_seed(2)
    m = QPLEXMixer(2, 3, 5)
    q, a, s = torch.randn(8, 2, 2, 3), torch.randint(3, (8, 2, 2)), torch.randn(8, 2, 5)
    assert (m.lambdas(a, s) > 0).all()
    v1 = m(q, a, s)
    m.detach_advantage = True
    assert torch.allclose(v1, m(q, a, s), atol=1e-5)


# ----------------------------------------------------------------------------- plumbing
@pytest.mark.parametrize("name,kw", [("qmix", {}), ("qplex", {}), ("qmix", dict(double_q=True, share_weights=True)),
                                     ("qplex", dict(n_heads=2, detach_advantage=True, role_info=True))])
def test_mixing_agents_act_learn_and_save(name, kw):
    agent = make_agent(name, **SPEC, seed=0, warmup_steps=50, batch_size=4, **kw)
    agent.reset_episode()
    rng = np.random.default_rng(0)
    a = agent.act(rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8))
    assert a.shape == (2,) and ((0 <= a) & (a < 8)).all()
    for t in range(120):
        agent.observe(rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8), rng.integers(0, 8, 2),
                      float(rng.normal()), rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8), (t + 1) % 40 == 0)
    out = {}
    for _ in range(8):
        out = agent.update() or out
    assert np.isfinite(out["loss"])
    assert any(k.startswith("mixer.") for k in agent.state_dict()["net"])        # mixer is trained and saved
    clone = make_agent(name, **SPEC, seed=1, **kw)
    clone.load_state_dict(agent.state_dict())
    for p, p2 in zip(agent.net.parameters(), clone.net.parameters()):
        assert torch.equal(p, p2)


def test_mixer_sees_other_agents_but_execution_is_decentralised():
    agent = make_agent("qmix", **SPEC, seed=0)
    x = torch.rand(1, 3, 2, 75)
    y = x.clone(); y[:, :, 1] = torch.rand(1, 3, 75)
    h = agent.net.initial_hidden(1, "cpu")
    assert torch.allclose(agent.net(x, h)[0][:, :, 0], agent.net(y, h)[0][:, :, 0])   # Q_0 ignores obs_1


def test_runner_trains_qplex_on_matrix_game(tmp_path):
    cfg = ExperimentConfig(name="m", env="two_stage_matrix", episode_len=40, eval_episode_len=20, agent="qplex",
                           agent_kwargs=dict(warmup_steps=40, batch_size=4), train_steps=120, eval_every=0,
                           final_eval_sessions=1, out_dir=str(tmp_path), verbose=0)
    assert len(Experiment(cfg).run()["final_eval"]["returns"]) == 1


# ----------------------------------------------------------------------------- matrix games
# Seeds run in parallel processes; see vdn_lab/experiment/matrix_study.py for the setup
# (uniform joint actions, 1-step targets, one round per episode, greedy round afterwards).
NONMONOTONIC = MATRIX_GAMES["nonmonotonic"]["payoffs"]          # [[8,-12,-12],[-12,0,0],[-12,0,0]]


def test_two_step_game_qmix_solves_vdn_does_not():
    """Optimum: agent 0 picks matrix 1, then both play B -> 8. Matrix 0 always pays 7.
    VDN's additive fit of matrix 1 peaks at 6.5, so it picks the safe matrix."""
    seeds = range(4)
    vdn = run_seeds(train_uniform, seeds, "vdn", "two_stage_matrix", "two_step", 3000, gamma=0.5, lr=2e-3)
    qmix = run_seeds(train_uniform, seeds, "qmix", "two_stage_matrix", "two_step", 3000, gamma=0.5, lr=2e-3)
    assert vdn == [7.0] * 4, vdn
    assert qmix == [8.0] * 4, qmix


def test_nonmonotonic_matrix_qplex_solves_qmix_does_not():
    """Optimum (A, A) = 8 is not representable by a monotonic mixer: QMIX never finds it. QPLEX
    can represent it and finds it in most seeds (5/8 here; the misses are optimisation local
    optima where lambda at (A, A) collapses, see the README)."""
    seeds = range(8)
    qmix = run_seeds(train_uniform, seeds, "qmix", "matrix", "nonmonotonic", 5000, gamma=0.0)
    qplex = run_seeds(train_uniform, seeds, "qplex", "matrix", "nonmonotonic", 5000, gamma=0.0)
    assert all(r < 8 for r in qmix), qmix
    assert sum(r == 8 for r in qplex) >= 4, qplex


def test_nonmonotonic_matrix_representation():
    """Regress Q_tot straight onto the payoff table (no RL). VDN and QMIX cannot fit it whatever
    the initialisation; QPLEX fits it exactly, with the optimal greedy joint action."""
    seeds = range(8)
    for mixer, floor in (("vdn", 50.0), ("qmix", 35.0)):
        assert all(loss > floor for loss, _ in run_seeds(fit_payoffs, seeds, mixer, NONMONOTONIC)), mixer
    fits = run_seeds(fit_payoffs, seeds, "qplex", NONMONOTONIC, detach_advantage=False)
    assert any(loss < 1e-3 and greedy == (0, 0) for loss, greedy in fits), fits
