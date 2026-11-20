import numpy as np
import pytest
import torch

from lbf_bench.algos import lambda_returns, nstep_returns
from lbf_bench.buffer import Collector, EpisodeBuffer
from lbf_bench.configs import ALGOS
from lbf_bench.env import TASKS, LBFEnv, make_envs

G = 0.9


def _episodes():
    """Two padded episodes: one terminated at L=3, one truncated at L=5 (T = 5)."""
    r = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0, 2.0]])
    term = torch.tensor([[0, 0, 1.0, 0, 0], [0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1.0, 0, 0], [1, 1, 1, 1, 1.0]])
    length = torch.tensor([3, 5])
    vals = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1) + 10      # V(s_t), t = 0..5
    return r, term, mask, length, vals


def _brute_nstep(r, term, length, vals, n):
    B, T = r.shape
    out = torch.zeros(B, T, 1)
    for b in range(B):
        L = int(length[b])
        for t in range(L):
            m = min(n, L - t)
            g = sum(G ** k * r[b, t + k] for k in range(m))
            terminal = (t + m == L) and term[b, L - 1] > 0
            out[b, t, 0] = g + (0.0 if terminal else G ** m * vals[b, t + m, 0])
    return out


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_nstep_matches_brute_force(n):
    r, term, mask, length, vals = _episodes()
    got = nstep_returns(r, term, length, vals, n, G)
    want = _brute_nstep(r, term, length, vals, n)
    assert torch.allclose(got * mask[..., None], want, atol=1e-5)


def test_lambda_returns_limits():
    r, term, mask, length, vals = _episodes()
    lambda_returns.gamma = G
    boot = vals[:, 1:]
    one_step = lambda_returns(r, term, mask, boot, 0.0)
    assert torch.allclose(one_step * mask[..., None], _brute_nstep(r, term, length, vals, 1), atol=1e-5)
    mc = lambda_returns(r, term, mask, boot, 1.0)
    assert torch.allclose(mc * mask[..., None], _brute_nstep(r, term, length, vals, 100), atol=1e-5)
    lam = torch.full((2, 5), 1.0)
    assert torch.allclose(lambda_returns(r, term, mask, boot, lam), mc)


def test_env_terminated_vs_truncated():
    env = LBFEnv("10x10-3p-3f", seed=0)
    obs = env.reset()
    assert obs.shape == (3, 18) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(200):
        t, done = 0, False
        env.reset()
        while not done:
            _, r, term, trunc = env.step(rng.integers(0, 6, 3))
            t += 1
            done = term or trunc
        assert not (term and trunc)
        if trunc:
            assert t == 50
        if term:
            assert env.env.field.sum() == 0


def test_tasks_build():
    for name, spec in TASKS.items():
        env = LBFEnv(name, seed=1)
        o = env.reset()
        assert o.shape == (spec.players, 3 * (spec.foods + spec.players))
        assert env.episode_limit == spec.max_steps


def test_collector_and_buffer():
    algo = ALGOS["qmix"]
    env = LBFEnv("8x8-2p-2f-c")
    a = algo.cls(env.n_agents, env.n_actions, env.obs_dim, env.episode_limit, 0, **algo.kwargs)
    col = Collector(make_envs("8x8-2p-2f-c", 3, 0))
    b = col.run(a, explore=True, t_env=0)
    for e, L in enumerate(b["length"]):
        assert b["filled"][e, :L].all() and not b["filled"][e, L:].any()
        assert b["term"][e, :L - 1].sum() == 0
    buf = EpisodeBuffer(4, env.episode_limit, 2, env.obs_dim, np.random.default_rng(0))
    buf.insert(b)
    buf.insert(b)                     # wraps around: 6 episodes into capacity 4
    assert buf.size == 4 and buf.ptr == 2
    recent = buf.sample_idx(2, recent=2)
    assert set(recent.tolist()) == {0, 1}
    old = buf.sample_idx(2, skip_recent=2)
    assert set(old.tolist()) == {2, 3}


@pytest.mark.parametrize("name", sorted(ALGOS))
def test_every_algorithm_trains(name):
    spec = ALGOS[name]
    env = LBFEnv("10x10-3p-3f")
    kw = dict(spec.kwargs)
    if spec.cls.__name__ in ("QLearner", "MACPF"):
        kw.update(batch=2)
        if kw.get("atd"):
            kw.update(atd_period=2)
    a = spec.cls(env.n_agents, env.n_actions, env.obs_dim, env.episode_limit, 0, **kw)
    col = Collector(make_envs("10x10-3p-3f", spec.n_envs, 0))
    out = {}
    for i in range(6):
        out = a.train(col.run(a, explore=True, t_env=i * 100), i * 100) or out
    assert out and all(np.isfinite(v) for v in out.values())
    if kw.get("atd"):
        assert a.lam_log and 0.0 <= a.lam_log[-1][1] <= 1.0


def test_agent_step_matches_sequence():
    from lbf_bench.nets import AgentNet
    torch.manual_seed(0)
    net = AgentNet(7, 16, 6, use_rnn=True)
    x = torch.randn(3, 9, 2, 7)                            # (B, T, n, in)
    seq = net.forward_seq(x)
    h = net.init_hidden(3 * 2)
    for t in range(9):
        out, h = net.step(x[:, t].reshape(6, 7), h)
        assert torch.allclose(out.reshape(3, 2, 6), seq[:, t], atol=1e-5)


def test_qplex_mixer_igm():
    """The per-agent greedy joint action maximises Q_tot over all joint actions (IGM)."""
    import itertools
    from lbf_bench.nets import QPLEXMixer
    torch.manual_seed(0)
    n, A, S = 3, 4, 10
    mix = QPLEXMixer(n, A, S, n_heads=4, layers=1)
    for _ in range(5):
        q = torch.randn(1, 1, n, A)
        s = torch.randn(1, 1, S)
        joint = torch.tensor(list(itertools.product(range(A), repeat=n)))           # (A^n, n)
        qt = mix(q.expand(len(joint), 1, n, A), joint[:, None], s.expand(len(joint), 1, S)).squeeze()
        greedy = q[0, 0].argmax(-1)
        assert qt.max() <= qt[(joint == greedy).all(1)].item() + 1e-5


def test_item_levels_follow_lbforaging_v1():
    """Non-cooperative items are 1..S-1 (S = sum of the 3 lowest agent levels), cooperative ones exactly S."""
    for name, coop in (("10x10-3p-3f", False), ("15x15-4p-5f", False), ("8x8-2p-2f-c", True)):
        env = LBFEnv(name, seed=3)
        seen_max = False
        for _ in range(300):
            env.reset()
            s = sum(sorted(p.level for p in env.env.players)[:3])
            items = env.env.field[env.env.field > 0]
            if coop:
                assert (items == s).all()
            else:
                assert items.min() >= 1 and items.max() <= max(s - 1, 1)
                seen_max |= bool((items == s - 1).any())
        assert coop or seen_max
