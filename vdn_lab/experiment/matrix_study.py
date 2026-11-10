"""Small studies on the matrix games: which value factorisation can learn which payoff table.

Two kinds of check, both used by ``tests/test_mixing.py`` and ``examples/07_matrix_qmix_qplex.py``:

* ``train_uniform`` - full RL: the agent's networks learn from uniformly random joint actions
  (epsilon = 1 throughout, 1-step targets) as in the didactic games of the QMIX and QPLEX papers,
  then the greedy joint policy is played once. One round per episode, so every trajectory starts
  from a fresh recurrent state, exactly like the greedy evaluation.
* ``fit_payoffs`` - representation only: free per-agent utilities plus a mixer, regressed
  directly (full batch, every joint action) onto a payoff table. Shows what a mixer can express
  without any RL noise.

``run_seeds`` runs either over several seeds in parallel processes.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Defaults for the matrix-game studies (not the paper's grid-world settings).
UNIFORM_LEARNING = dict(eps_start=1.0, eps_end=1.0, lam=0.0, batch_size=64, warmup_steps=200, train_every=1,
                        target_update_every=50, lr=5e-3, loss="mse")


def train_uniform(agent: str, env: str, layout: str, steps: int, seed: int, **agent_kwargs: Any) -> float:
    """Train ``agent`` on ``env``/``layout`` for ``steps`` env steps of uniformly random joint
    actions, then return the team reward of one greedy round."""
    import torch
    from ..agents import make_agent
    from ..envs import make_env

    torch.set_num_threads(1)
    round_len = 2 if env == "two_stage_matrix" else 1
    game = make_env(env, layout=layout, max_steps=round_len, seed=seed)
    learner = make_agent(agent, **game.spec, seed=seed, **{**UNIFORM_LEARNING, "seq_len": round_len, **agent_kwargs})
    for _ in range(steps // round_len):
        obs, done = game.reset(), False
        learner.reset_episode()
        while not done:
            actions = learner.act(obs, explore=True)
            nxt, r, done, _ = game.step(actions)
            learner.observe(obs, actions, r, nxt, done)
            learner.update()
            obs = nxt
    learner.set_training(False)
    obs, total, done = game.reset(), 0.0, False
    learner.reset_episode()
    while not done:
        obs, r, done, _ = game.step(learner.act(obs, explore=False))
        total += r
    return float(total)


def fit_payoffs(mixer: str, payoffs: Sequence[Sequence[float]], seed: int, steps: int = 1500,
                lr: float = 1e-2, **mixer_kwargs: Any) -> Tuple[float, Tuple[int, ...]]:
    """Regress ``Q_tot(a) = mixer(Q_1..Q_d)(a)`` onto a common-payoff table, every joint action
    weighted equally. Utilities ``Q_i`` are free parameters (the state is a constant).

    ``mixer``: ``"vdn"`` (sum), ``"qmix"`` or ``"qplex"``. Returns (final mean squared error,
    greedy joint action ``argmax_a Q_i`` of each agent)."""
    import torch
    from ..agents.mixers import QMixer, QPLEXMixer

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    R = torch.as_tensor(np.asarray(payoffs, dtype=np.float32))
    d, n_actions = R.ndim, max(R.shape)
    joint = torch.tensor(list(itertools.product(*[range(k) for k in R.shape])))        # (N, d)
    target = R[tuple(joint.T)]
    if mixer == "vdn":
        mix: Callable = lambda q, a, s: q.gather(-1, a.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
        params: List[torch.nn.Parameter] = []
    else:
        mod = QMixer(d, 1, **mixer_kwargs) if mixer == "qmix" else QPLEXMixer(d, n_actions, 1, **mixer_kwargs)
        mix, params = mod, list(mod.parameters())
    q = torch.nn.Parameter(0.1 * torch.randn(1, d, n_actions))
    opt = torch.optim.Adam([q, *params], lr=lr)
    state = torch.ones(len(joint), 1)
    for _ in range(steps):
        loss = ((mix(q.expand(len(joint), d, n_actions), joint, state).squeeze(-1) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item()), tuple(q[0].argmax(-1).tolist())


def _call(job):
    fn, args, kwargs = job
    return fn(*args, **kwargs)


def run_seeds(fn: Callable, seeds: Sequence[int], *args: Any, processes: Optional[int] = None,
              **kwargs: Any) -> List[Any]:
    """``[fn(*args, seed, **kwargs) for seed in seeds]``, in parallel processes (spawned, so
    it is safe with torch). ``fn`` must be importable (a module-level function)."""
    jobs = [(fn, (*args, s), kwargs) for s in seeds]
    processes = processes or min(len(jobs), mp.cpu_count())
    if processes <= 1:
        return [_call(j) for j in jobs]
    with mp.get_context("spawn").Pool(processes) as pool:
        return pool.map(_call, jobs)


def summarize(results: Dict[str, List[float]], optimum: float) -> str:
    lines = [f"{'agent':<8} solved  greedy returns per seed"]
    for name, rets in results.items():
        ok = sum(r == optimum for r in rets)
        lines.append(f"{name:<8} {ok}/{len(rets):<5} {rets}")
    return "\n".join(lines)
