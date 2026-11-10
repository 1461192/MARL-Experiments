import numpy as np
import pytest
import torch

from vdn_lab.agents import make_agent
from vdn_lab.agents.mappo import ValueNorm, gae
from vdn_lab.experiment import Experiment, ExperimentConfig

SPEC = dict(n_agents=2, n_actions=8, obs_shape=(3, 5, 5))


def rollout(agent, steps, ep_len=37, seed=0):
    """Drive the agent like the runner does: reset, act, observe, update."""
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


@pytest.mark.parametrize("name,kw", [
    ("mappo", {}), ("ippo", {}), ("mappo", dict(critic_input="concat")),
    ("mappo", dict(share_weights=False, role_info=True)), ("mappo", dict(recurrent=False)),
    ("mappo", dict(value_norm=False, huber_delta=None, value_clip=None, num_mini_batch=4)),
])
def test_mappo_acts_and_learns(name, kw):
    agent = make_agent(name, **SPEC, seed=0, rollout_len=100, ppo_epochs=2, **kw)
    m = rollout(agent, 100)
    assert agent.n_updates == 1 and agent.ptr == 0
    for k in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        assert np.isfinite(m[k]), k
    a = agent.act(np.zeros((2, 3, 5, 5), np.uint8), explore=False)
    assert a.shape == (2,) and ((0 <= a) & (a < 8)).all()


def test_ippo_uses_local_critic():
    assert make_agent("ippo", **SPEC).cfg.critic_input == "local"
    assert make_agent("mappo", **SPEC).cfg.critic_input == "agent_specific"


def test_agent_specific_critic_input_puts_own_obs_first():
    agent = make_agent("mappo", **SPEC, seed=0)
    x = torch.rand(4, 2, 75)
    ci = agent._critic_input(x)                                   # (4, 2, 2*75 + 2)
    assert ci.shape == (4, 2, 152)
    assert torch.equal(ci[:, 0, :75], x[:, 0]) and torch.equal(ci[:, 0, 75:150], x[:, 1])
    assert torch.equal(ci[:, 1, :75], x[:, 1]) and torch.equal(ci[:, 1, 75:150], x[:, 0])
    assert torch.equal(ci[:, 0, 150:], torch.tensor([1., 0.]).expand(4, 2))
    assert torch.equal(ci[:, 1, 150:], torch.tensor([0., 1.]).expand(4, 2))


def test_gae_matches_hand_computation():
    g, lam = 0.9, 0.5
    r = np.array([1.0, 0.0, 2.0, 1.0], np.float32)
    v = np.array([[0.5], [1.0], [0.2], [0.3]], np.float32)
    nv = np.array([[0.0], [7.0], [0.0], [4.0]], np.float32)        # read only at t=1 (done) and t=3 (end)
    done = np.array([False, True, False, False])
    adv = gae(r, v, nv, done, g, lam)[:, 0]
    d3 = 1.0 + g * 4.0 - 0.3
    d2 = 2.0 + g * 0.3 - 0.2
    d1 = 0.0 + g * 7.0 - 1.0
    d0 = 1.0 + g * 1.0 - 0.5
    np.testing.assert_allclose(adv, [d0 + g * lam * d1, d1, d2 + g * lam * d3, d3], rtol=1e-6)


def test_chunked_training_forward_reproduces_collection_logprobs():
    """BPTT over chunks from the stored hidden states (with resets at episode starts) gives
    exactly the policy that acted, so the first PPO ratio is 1."""
    agent = make_agent("mappo", **SPEC, seed=0, rollout_len=120, chunk_len=10)
    rollout(agent, 119, ep_len=23)                                 # buffer almost full, episodes cut mid-chunk
    b, L = agent.buf, 10
    n = 110
    x = agent._to_input(b["obs"][:n]).reshape(n // L, L, 2, 75).transpose(0, 1)
    starts = torch.from_numpy(b["starts"][:n]).reshape(n // L, L).T
    h0 = torch.from_numpy(b["actor_h"][:n:L])
    with torch.no_grad():
        logits, _ = agent.actor(x, h0, starts)
    acts = torch.from_numpy(b["actions"][:n]).reshape(n // L, L, 2).transpose(0, 1)
    logp = torch.log_softmax(logits, -1).gather(-1, acts.unsqueeze(-1)).squeeze(-1)
    stored = torch.from_numpy(b["logp"][:n]).reshape(n // L, L, 2).transpose(0, 1)
    torch.testing.assert_close(logp, stored, atol=1e-5, rtol=1e-5)


def test_value_norm_tracks_mean_and_std():
    vn = ValueNorm()
    x = torch.randn(10_000) * 5 + 3
    vn.update(x)
    m, s = vn.stats()
    assert abs(m.item() - 3) < 0.2 and abs(s.item() - 5) < 0.2
    y = torch.randn(7)
    torch.testing.assert_close(vn.denormalize(vn.normalize(y)), y)


def test_mappo_rejects_bad_config():
    with pytest.raises(ValueError):
        make_agent("mappo", **SPEC, rollout_len=105, chunk_len=10)
    with pytest.raises(ValueError):
        make_agent("mappo", **SPEC, critic_input="global")
    with pytest.raises(TypeError):
        make_agent("mappo", **SPEC, not_a_field=1)


def test_mappo_save_load_roundtrip(tmp_path):
    a = make_agent("mappo", **SPEC, seed=0, rollout_len=50, ppo_epochs=1)
    rollout(a, 50)
    torch.save(a.state_dict(), tmp_path / "a.pt")
    b = make_agent("mappo", **SPEC, seed=1, rollout_len=50, ppo_epochs=1)
    b.load_state_dict(torch.load(tmp_path / "a.pt", weights_only=False))
    o = np.random.default_rng(0).integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
    a.reset_episode(), b.reset_episode()
    assert np.array_equal(a.act(o, explore=False), b.act(o, explore=False))
    assert b.n_updates == 1 and b.value_norm.debias.item() > 0


def test_mappo_solves_coordination_matrix_game():
    cfg = ExperimentConfig(env="matrix", env_kwargs={"layout": "penalty"}, agent="mappo", episode_len=10,
                           eval_episode_len=10, train_steps=6000, eval_every=0, final_eval_sessions=1,
                           out_dir=None, verbose=0, agent_kwargs=dict(rollout_len=200, gamma=0.0, lr=3e-3))
    res = Experiment(cfg).run()
    assert res["final_eval"]["mean_return"] == 100.0             # optimum 10 every round
