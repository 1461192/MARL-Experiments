import numpy as np
import pytest
import torch

from vdn_lab import AGENTS
from vdn_lab.agents import make_agent, lambda_returns
from vdn_lab.agents.networks import AgentQNet, IndependentNets
from vdn_lab.agents.replay import SequenceReplay

SPEC = dict(n_agents=2, n_actions=8, obs_shape=(3, 5, 5))


def fill(agent, steps=120, ep_len=40, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(steps):
        o = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
        o2 = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
        agent.observe(o, rng.integers(0, 8, 2), float(rng.normal()), o2, (t + 1) % ep_len == 0)


def test_paper_architecture_sizes():
    net = AgentQNet(75, 8, 32)
    assert net.fc.out_features == 32 and net.lstm.hidden_size == 32
    assert net.head.value.in_features == 32 and net.head.adv.out_features == 8


@pytest.mark.parametrize("name,kw", [("independent", {}), ("vdn", {}), ("vdn", dict(share_weights=True, role_info=True)),
                                     ("centralized", {}), ("centralized_vdn", {}), ("random", {})])
def test_agents_act_and_learn(name, kw):
    agent = make_agent(name, **SPEC, seed=0, **({} if name == "random" else dict(warmup_steps=50, batch_size=4, **kw)))
    agent.reset_episode()
    rng = np.random.default_rng(0)
    a = agent.act(rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8))
    assert a.shape == (2,) and ((0 <= a) & (a < 8)).all()
    fill(agent)
    out = {}
    for _ in range(8):
        out = agent.update() or out
    if name != "random":
        assert "loss" in out and np.isfinite(out["loss"])


def test_shared_weights_equal_manual_per_agent_forward():
    torch.manual_seed(0)
    net = IndependentNets(2, 75, 8, 32, share_weights=True, role_info=True)
    x = torch.rand(3, 5, 2, 75)
    q, _ = net(x, net.initial_hidden(3, "cpu"))
    for i in range(2):
        xi = torch.cat([x[:, :, i], torch.eye(2)[i].expand(3, 5, 2)], -1)
        qi, _ = net.nets[0](xi, torch.zeros(3, 2, 32))
        assert torch.allclose(q[:, :, i], qi, atol=1e-6)


def test_decentralised_vs_centralised_information_flow():
    """VDN / independent: agent 0's Q must not depend on agent 1's observation. Centralised: it does."""
    x = torch.rand(1, 3, 2, 75)
    y = x.clone(); y[:, :, 1] = torch.rand(1, 3, 75)
    for name in ("vdn", "independent"):
        ag = make_agent(name, **SPEC, seed=0)
        q1, _ = ag.net(x, ag.net.initial_hidden(1, "cpu")); q2, _ = ag.net(y, ag.net.initial_hidden(1, "cpu"))
        assert torch.allclose(q1[:, :, 0], q2[:, :, 0])
    ag = make_agent("centralized", **SPEC, seed=0)
    q1, _ = ag.net(x, ag.net.initial_hidden(1, "cpu")); q2, _ = ag.net(y, ag.net.initial_hidden(1, "cpu"))
    assert not torch.allclose(q1, q2)


def test_vdn_chosen_and_bootstrap_are_sums():
    ag = make_agent("vdn", **SPEC, seed=0)
    q = torch.randn(4, 3, 2, 8)
    a = torch.randint(0, 8, (4, 3, 2))
    manual = q[..., 0, :].gather(-1, a[..., :1]) + q[..., 1, :].gather(-1, a[..., 1:])
    assert torch.allclose(ag._chosen_q(q, a), manual)
    assert torch.allclose(ag._max_q(q), q.max(-1).values.sum(-1, keepdim=True))
    assert make_agent("independent", **SPEC, seed=0)._chosen_q(q, a).shape == (4, 3, 2)


def test_joint_action_roundtrip():
    ag = make_agent("centralized", **SPEC, seed=0)
    for a0 in range(8):
        for a1 in range(8):
            idx = int(ag._encode(torch.tensor([a0, a1])))
            assert list(ag._decode(idx)) == [a0, a1] and 0 <= idx < 64


def test_lambda_return_special_cases():
    g = 0.9
    r = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1)
    boot = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1)
    G0 = lambda_returns(r, boot, g, 0.0)[0, :, 0]                 # lam=0 -> one-step TD target
    assert torch.allclose(G0, r[0, :, 0] + g * boot[0, :, 0])
    G1 = lambda_returns(r, boot, g, 1.0)[0, :, 0]                 # lam=1 -> n-step return
    assert abs(G1[0].item() - (1 + g * 2 + g ** 2 * 3 + g ** 3 * 30)) < 1e-5
    assert abs(G1[2].item() - (3 + g * 30)) < 1e-5
    Gm = lambda_returns(r, boot, g, 0.9)[0, :, 0]                 # mixture
    assert abs(Gm[2].item() - (3 + g * 30)) < 1e-5
    assert abs(Gm[1].item() - (2 + g * (0.1 * 20 + 0.9 * Gm[2].item()))) < 1e-5


def test_replay_windows_never_cross_episodes():
    rep = SequenceReplay(100, 2, (3, 5, 5), seed=0)
    rng = np.random.default_rng(0)
    for t in range(330):                              # wraps the ring buffer several times
        rep.add(np.full((2, 3, 5, 5), t % 40, dtype=np.uint8), rng.integers(0, 8, 2), 0.0)
        if (t + 1) % 40 == 0:
            rep.end_episode(np.full((2, 3, 5, 5), 40, dtype=np.uint8))
    for _ in range(50):
        marks = rep.sample(16, 9)["obs"][:, :, 0, 0, 0, 0].astype(int)
        assert (np.diff(marks, axis=1) == 1).all(), "window crossed an episode / ring boundary"


def test_save_and_load_roundtrip():
    a = make_agent("vdn", **SPEC, seed=0)
    b = make_agent("vdn", **SPEC, seed=1)
    b.load_state_dict(a.state_dict())
    obs = np.random.default_rng(0).integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
    a.reset_episode(); b.reset_episode()
    assert (a.act(obs, explore=False) == b.act(obs, explore=False)).all()


def test_registry_lists_agents():
    assert {"vdn", "independent", "centralized", "centralized_vdn", "random"} <= set(AGENTS.names())
