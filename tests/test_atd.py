"""Adaptive TD(lambda): per-step lambda in the targets, the density-ratio estimator, the agents."""
import numpy as np
import pytest
import torch

from vdn_lab.agents import make_agent
from vdn_lab.agents.atd import AdaptiveLambda, ATDConfig
from vdn_lab.agents.mappo import gae
from vdn_lab.agents.recurrent_dqn import lambda_returns
from vdn_lab.agents.replay import SequenceReplay

SPEC = dict(n_agents=2, n_actions=8, obs_shape=(3, 5, 5))


def drive(agent, steps, ep_len=37, seed=0):
    rng = np.random.default_rng(seed)
    agent.set_training(True)
    metrics, t = {}, 0
    while t < steps:
        agent.reset_episode()
        o = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
        for k in range(ep_len):
            a = agent.act(o)
            o2 = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
            agent.observe(o, a, float(rng.normal()), o2, k == ep_len - 1)
            metrics = agent.update() or metrics
            o, t = o2, t + 1
            if t == steps:
                break
    return metrics


# ------------------------------------------------------------------ targets with per-step lambda
def test_lambda_returns_tensor_lambda_matches_scalar():
    g = torch.Generator().manual_seed(0)
    r, boot = torch.randn(4, 8, 1, generator=g), torch.randn(4, 8, 3, generator=g)
    assert torch.allclose(lambda_returns(r, boot, 0.9, 0.7), lambda_returns(r, boot, 0.9, torch.full((4, 8, 1), 0.7)))


def test_lambda_returns_zero_lambda_cuts_the_trace():
    r, boot = torch.ones(1, 4, 1), torch.zeros(1, 4, 1)
    boot[0, 1] = 10.0
    lam = torch.ones(1, 4, 1)
    lam[0, 1] = 0.0                                  # after step 1: only boot_1, not the later rewards
    G = lambda_returns(r, boot, 1.0, lam)
    assert G[0, 1, 0] == pytest.approx(1 + 10)
    assert G[0, 0, 0] == pytest.approx(1 + 11)       # lam_0 = 1 -> passes G_1 through
    assert G[0, 3, 0] == pytest.approx(1.0)


def test_gae_array_lambda_matches_scalar_and_shares_over_agents():
    rng = np.random.default_rng(0)
    T = 20
    r, v, nv = rng.normal(size=T), rng.normal(size=(T, 2)), rng.normal(size=(T, 2))
    dones = np.zeros(T, bool)
    dones[9] = True
    a = gae(r, v, nv, dones, 0.99, 0.95)
    assert np.allclose(a, gae(r, v, nv, dones, 0.99, np.full(T, 0.95)))
    assert np.allclose(a, gae(r, v, nv, dones, 0.99, np.full((T, 2), 0.95)))
    lam = np.full(T, 0.95)
    lam[3] = 0.0                                     # A_3 = delta_3 only
    b = gae(r, v, nv, dones, 0.99, lam)
    assert b[3, 0] == pytest.approx(r[3] + 0.99 * v[4, 0] - v[3, 0])
    assert np.allclose(a[4:], b[4:])


# -------------------------------------------------------------------------- replay / estimator
def test_replay_sample_steps_recent_and_valid_only():
    R = SequenceReplay(50, 2, (1,), seed=0)
    for ep in range(6):
        for t in range(9):
            R.add(np.full((2, 1), ep, np.uint8), np.array([ep, t]), 0.0)
        R.end_episode(np.full((2, 1), 255, np.uint8))
    s = R.sample_steps(200, recent=10)               # newest 10 entries: last 9 steps of ep 5 + its final obs
    assert (s["obs"] == 5).all() and (s["actions"][:, 0] == 5).all()
    s = R.sample_steps(500)
    assert not (s["obs"] == 255).any()               # final observations are never sampled as transitions
    assert set(np.unique(s["actions"][:, 0])) <= set(range(6))


def test_adaptive_lambda_scaling():
    cfg = ATDConfig(atd_warmup=5, atd_temperature=1.0, lam_min=0.1, lam_max=1.0, atd_lr=1e-2)
    al = AdaptiveLambda(cfg, 0.5, state_dim=4, n_agents=2, n_actions=3, device="cpu")
    s_new, s_old = torch.zeros(64, 4), torch.zeros(64, 4)
    s_new[:, 0], s_old[:, 1] = 1.0, 1.0
    a = torch.zeros(64, 2, dtype=torch.long)
    assert torch.allclose(al.lambdas(s_new, a), torch.full((64,), 0.5))        # base lambda before warm-up
    for _ in range(300):
        al.train_step(s_new, a, torch.cat([s_old[:48], s_new[:16]]), a)        # large buffer: 1/4 recent data
    assert al.last["atd_acc"] > 0.6
    both = torch.cat([s_new[:32], s_old[:32]])
    lam = al.lambdas(both, torch.zeros(64, 2, dtype=torch.long))
    assert (lam[:32] > 0.5).all() and (lam[32:] < 0.5).all()     # on-policy pairs get the larger lambda
    assert lam.mean() == pytest.approx(0.5, abs=0.05)            # normalised: the base lambda on average
    assert ((lam >= 0.1) & (lam <= 1.0)).all()
    # with a mask only the selected pairs enter the normalisation
    mask = torch.zeros(64, dtype=torch.bool)
    mask[32:] = True
    assert al.lambdas(both, torch.zeros(64, 2, dtype=torch.long), mask=mask)[32:].mean() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------------- agents
@pytest.mark.parametrize("name", ["vdn_atd", "qmix_atd", "qplex_atd"])
def test_dqn_atd_agents_learn(name):
    agent = make_agent(name, **SPEC, seed=0, warmup_steps=100, batch_size=8, train_every=2,
                       atd_warmup=3, atd_batch=16, atd_small=60, buffer_size=600)
    m = drive(agent, 300)
    for k in ("loss", "atd_loss", "atd_acc", "lam_mean", "lam_std"):
        assert np.isfinite(m[k]), k
    assert 0.5 <= m["lam_mean"] <= 0.99 and agent.atd.n_steps == agent.n_updates
    state = agent.state_dict()
    clone = make_agent(name, **SPEC, seed=1, atd_small=60, buffer_size=600)
    clone.load_state_dict(state)
    x = torch.rand(5, 150)
    acts = torch.zeros(5, 2, dtype=torch.long)
    assert torch.allclose(clone.atd.est(x, acts), agent.atd.est(x, acts))


def test_dqn_atd_untrained_is_base_algorithm():
    """Before the estimator warm-up the lambdas equal the base lambda, so the targets are QMIX's."""
    base = make_agent("qmix", **SPEC, seed=0)
    atd = make_agent("qmix_atd", **SPEC, seed=0)
    obs = torch.rand(4, 10, 2, 75)
    acts = torch.randint(0, 8, (4, 9, 2))
    lam = atd._trace_lambda(obs, acts)
    assert lam.shape == (4, 8, 1) and torch.allclose(lam, torch.full_like(lam, base.cfg.lam))


def test_mappo_atd_learns_with_per_step_lambda():
    agent = make_agent("mappo_atd", **SPEC, seed=0, rollout_len=100, ppo_epochs=2, atd_warmup=10, atd_steps=10)
    m = drive(agent, 300)
    assert agent.n_updates == 3 and len(agent._history) == 3
    for k in ("loss", "atd_loss", "lam_mean", "lam_std"):
        assert np.isfinite(m[k]), k
    assert agent._lam.shape == (100,) and ((agent._lam >= 0.5) & (agent._lam <= 0.99)).all()
    a = agent.act(np.zeros((2, 3, 5, 5), np.uint8), explore=False)
    assert a.shape == (2,)


def test_mappo_atd_history_is_bounded():
    agent = make_agent("mappo_atd", **SPEC, seed=0, rollout_len=50, ppo_epochs=1, atd_large=100, atd_steps=1)
    drive(agent, 250)
    assert agent.n_updates == 5 and len(agent._history) == 2
