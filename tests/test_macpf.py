import numpy as np
import pytest
import torch

from vdn_lab.agents import make_agent
from vdn_lab.agents.macpf import td_lambda_targets
from vdn_lab.experiment import Experiment, ExperimentConfig

SPEC = dict(n_agents=2, n_actions=8, obs_shape=(3, 5, 5))


def rollout(agent, episodes, ep_len=13, seed=0):
    rng = np.random.default_rng(seed)
    agent.set_training(True)
    metrics = {}
    for _ in range(episodes):
        agent.reset_episode()
        o = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
        for k in range(ep_len):
            a = agent.act(o)
            o2 = rng.integers(0, 255, (2, 3, 5, 5), dtype=np.uint8)
            agent.observe(o, a, float(rng.normal()), o2, k == ep_len - 1)
            metrics = agent.update() or metrics
            o = o2
    return metrics


@pytest.mark.parametrize("kw", [{}, dict(mixer="qmix"), dict(shared_mixer=True), dict(max_train_len=5),
                                dict(obs_last_action=False, obs_agent_id=False), dict(eval_policy="dep")])
def test_macpf_acts_and_learns(kw):
    agent = make_agent("macpf", **SPEC, seed=0, batch_size=4, **kw)
    m = rollout(agent, 6)
    assert agent.n_updates == 3 and len(agent.buffer) == 6
    assert (agent.mixer_dep is agent.mixer_ind) == bool(kw.get("shared_mixer"))
    for k in ("policy_loss_ind", "policy_loss_dep", "critic_loss_ind", "critic_loss_dep", "loss"):
        assert np.isfinite(m[k]), k
    agent.reset_episode()
    a = agent.act(np.zeros((2, 3, 5, 5), np.uint8), explore=False)
    assert a.shape == (2,) and ((0 <= a) & (a < 8)).all()


def test_dependency_correction_only_sees_preceding_agents():
    agent = make_agent("macpf", **SPEC, seed=0)
    h = torch.randn(5, 2, agent.cfg.hidden)
    a1 = torch.tensor([[0, 3]] * 5)
    a2 = torch.tensor([[0, 6]] * 5)                    # differs only in the last agent's action
    a3 = torch.tensor([[4, 3]] * 5)                    # differs in agent 0's action
    q1, q2, q3 = (agent._critic_q(agent.critic, h, a, dep=True) for a in (a1, a2, a3))
    assert torch.allclose(q1, q2)                      # nobody conditions on the last agent
    assert torch.allclose(q1[:, 0], q3[:, 0])          # agent 0 conditions on nobody
    assert not torch.allclose(q1[:, 1], q3[:, 1])      # agent 1 conditions on agent 0
    q_ind = agent._critic_q(agent.critic, h, a1, dep=False)
    assert torch.allclose(q_ind, agent.critic.ind_head(h))


def test_dependent_loss_does_not_train_independent_head():
    agent = make_agent("macpf", **SPEC, seed=0, batch_size=2)
    rollout(agent, 2)
    batch = agent._sample()
    x = agent._inputs(batch["obs"], batch["last"])
    before = agent.actor.fc2.weight.detach().clone()
    agent._train_actor(batch, x, dep=True)
    assert torch.equal(before, agent.actor.fc2.weight)  # stop-gradient on the independent logits
    agent._train_actor(batch, x, dep=False)
    assert not torch.equal(before, agent.actor.fc2.weight)


def test_td_lambda_targets_match_one_step_and_mc():
    r = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])          # episode of 3 steps, padded to 4
    q = torch.tensor([[10.0, 20.0, 30.0, 40.0, 40.0]])   # bootstrap at index 3, copied to 4
    g = 0.5
    one = td_lambda_targets(r, mask, q, g, 0.0)
    assert torch.allclose(one[0, :3], torch.tensor([1 + g * 20, 2 + g * 30, 3 + g * 40]))
    mc = td_lambda_targets(r, mask, q, g, 1.0)
    assert torch.allclose(mc[0, :3], torch.tensor([1 + g * 2 + g * g * 3 + g ** 3 * 40,
                                                   2 + g * 3 + g * g * 40, 3 + g * 40]))


def test_macpf_save_load_and_runner(tmp_path):
    cfg = ExperimentConfig(name="m", env="fetch", env_kwargs={"layout": "open"}, agent="macpf",
                           agent_kwargs=dict(batch_size=2, max_train_len=10), episode_len=20,
                           eval_episode_len=20, train_steps=100, eval_every=50, final_eval_sessions=1,
                           out_dir=str(tmp_path), verbose=0)
    exp = Experiment(cfg)
    res = exp.run()
    assert exp.agent.n_updates > 0 and np.isfinite(res["final_eval"]["mean_return"])
    exp2 = Experiment.from_run_dir(exp.run_dir, verbose=0, agent_kwargs={"eval_policy": "dep"})
    for k, v in exp.agent.actor.state_dict().items():
        assert torch.equal(v, exp2.agent.actor.state_dict()[k])
    assert exp2.agent.cfg.eval_policy == "dep"
    exp2.evaluate(1)


def test_fop_mixer_value_and_advantage_split():
    """Q_jt = sum_i (w_i Q_i + V/d) + sum_i (lambda_i - 1) w_i (Q_i - V_i): at the greedy joint
    action (Q_i = V_i for alpha -> 0) the lambda term vanishes."""
    from vdn_lab.agents.macpf import FOPMixer
    torch.manual_seed(0)
    mix = FOPMixer(2, 8, 75, 16, 16, 4, 0.001)
    q = torch.randn(3, 5, 2, 8)
    state = torch.rand(3, 5, 150)
    greedy = q.argmax(-1)
    vs = q.max(-1).values
    onehot = torch.nn.functional.one_hot(greedy, 8).float()
    q_jt = mix(vs, state, onehot, vs)
    other = mix(vs, state, torch.roll(onehot, 1, -1), vs)          # lambda changes, advantages are 0
    assert torch.allclose(q_jt, other)
    a = (greedy + 1) % 8
    qa = q.gather(-1, a.unsqueeze(-1)).squeeze(-1)
    assert not torch.allclose(mix(qa, state, torch.nn.functional.one_hot(a, 8).float(), vs),
                              mix(qa, state, torch.roll(torch.nn.functional.one_hot(a, 8).float(), 1, -1), vs))
