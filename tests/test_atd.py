"""Adaptive TD(lambda): per-step lambda in the targets, the estimator, the lambda cache, the agents."""
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


def test_lambda_returns_follow_eq_10():
    """R_t = r_t + gamma [omega_t R_{t+1} + (1 - omega_t) boot_t]: omega_t = 0 cuts the trace at t."""
    r, boot = torch.ones(1, 4, 1), torch.zeros(1, 4, 1)
    boot[0, 1] = 10.0
    lam = torch.ones(1, 4, 1)
    lam[0, 1] = 0.0
    G = lambda_returns(r, boot, 1.0, lam)
    assert G[0, 1, 0] == pytest.approx(1 + 10)       # only the bootstrap after step 1
    assert G[0, 0, 0] == pytest.approx(1 + 11)       # omega_0 = 1 passes R_1 through
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
def test_replay_recent_windows_stay_in_the_newest_entries():
    R = SequenceReplay(60, 2, (1,), seed=0)
    for ep in range(6):
        for t in range(9):
            R.add(np.full((2, 1), ep, np.uint8), np.array([ep, t]), 0.0)
        R.end_episode(np.full((2, 1), 200 + ep, np.uint8))
    b = R.sample(100, 4, recent=10)                  # newest 10 entries = episode 5 only
    assert (b["actions"][:, :, 0] == 5).all() and b["idx"].shape == (100, 4)
    assert ((R.ptr - b["idx"]) % R.cap <= 10).all()
    b = R.sample(200, 4)
    assert set(np.unique(b["actions"][:, :, 0])) == set(range(6))


@pytest.mark.parametrize("recurrent", [True, False])
def test_estimator_learns_the_on_policy_probability(recurrent):
    """The sigmoid output is P(on-policy buffer | s, a): near 1 for pairs only in D_on, near 0 for
    pairs only in D_off, about |D_on| / (|D_on| + |D_off|) for pairs in both (eq. 8)."""
    torch.manual_seed(0)
    cfg = ATDConfig(atd_lr=1e-2, atd_recurrent=recurrent)
    al = AdaptiveLambda(cfg, state_dim=3, n_agents=2, n_actions=3, device="cpu")
    new, old, both = (torch.eye(3)[i].expand(64, 4, 3) for i in range(3))
    a = torch.zeros(64, 4, 2, dtype=torch.long)
    for _ in range(300):
        al.train_step(dict(state=torch.cat([new[:32], both[:32]]), actions=a),
                      dict(state=torch.cat([old[:32], both[:32]]), actions=a))
    assert al.last["atd_acc"] > 0.7
    assert (al.lambdas(new, a) > 0.9).all() and (al.lambdas(old, a) < 0.1).all()
    assert al.lambdas(both, a).mean() == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------------- agents
@pytest.mark.parametrize("name", ["vdn_atd", "qmix_atd", "qplex_atd"])
def test_dqn_atd_agents_learn(name):
    agent = make_agent(name, **SPEC, seed=0, warmup_steps=100, batch_size=8, train_every=2,
                       target_update_every=10, buffer_size=600, atd_ratio=10)
    m = drive(agent, 400)
    for k in ("loss", "atd_loss", "atd_acc", "lam_mean", "lam_std"):
        assert np.isfinite(m[k]), k
    assert 0.0 < m["lam_mean"] < 1.0
    assert agent.atd.n_steps == agent.n_updates // 10              # one estimator step per target update
    clone = make_agent(name, **SPEC, seed=1, buffer_size=600)
    clone.load_state_dict(agent.state_dict())
    x, acts = torch.rand(5, 3, 150), torch.zeros(5, 3, 2, dtype=torch.long)
    assert torch.allclose(clone.atd.net(x, acts), agent.atd.net(x, acts))


def test_dqn_lambda_cache_is_reused_and_cleared_at_target_updates():
    agent = make_agent("qmix_atd", **SPEC, seed=0, warmup_steps=100, batch_size=8, train_every=2,
                       target_update_every=10_000, buffer_size=600)
    drive(agent, 200)
    assert np.isfinite(agent.lam_cache).any()
    b = agent.replay.sample(8, agent.cfg.seq_len + 1)
    obs, acts = agent._to_input(b["obs"]), torch.from_numpy(b["actions"])
    first = agent._trace_lambda(obs, acts, b["idx"])
    with torch.no_grad():                            # a changed estimator does not change cached lambdas
        for p in agent.atd.net.parameters():
            p.add_(1.0)
    assert torch.equal(agent._trace_lambda(obs, acts, b["idx"]), first)
    agent._on_target_update()
    assert np.isnan(agent.lam_cache).all()
    assert not torch.equal(agent._trace_lambda(obs, acts, b["idx"]), first)
    # overwriting replay entries invalidates their cached lambda
    agent.lam_cache[:] = 0.5
    start = agent.replay.ptr
    agent.observe(np.zeros((2, 3, 5, 5), np.uint8), np.zeros(2, np.int64), 0.0, np.zeros((2, 3, 5, 5), np.uint8), False)
    assert np.isnan(agent.lam_cache[start]) and np.isnan(agent.lam_cache).sum() == 1


def test_mappo_atd_changes_only_the_critic_targets():
    agent = make_agent("mappo_atd", **SPEC, seed=0, rollout_len=100, ppo_epochs=2, atd_ratio=2)
    m = drive(agent, 300)
    assert agent.n_updates == 3 and len(agent._history) == 2
    for k in ("loss", "atd_loss", "lam_mean", "lam_std"):
        assert np.isfinite(m[k]), k
    assert agent._lam.shape == (100,) and ((agent._lam > 0) & (agent._lam < 1)).all()
    assert agent.cfg.gae_lambda == 0.95                           # the actor's GAE is unchanged
    a = agent.act(np.zeros((2, 3, 5, 5), np.uint8), explore=False)
    assert a.shape == (2,)
