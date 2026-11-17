# `lbf_bench`: QMIX, QPLEX, MAPPO, MACPF and adaptive TD(λ) on Level-Based Foraging

The study runs four algorithms and the adaptive TD(λ) (ATD) variants of three of them on the seven
Level-Based Foraging (LBF) tasks of

> Papoudakis, Christianos, Schäfer & Albrecht, *Benchmarking Multi-Agent Deep Reinforcement Learning
> Algorithms in Cooperative Tasks*, NeurIPS 2021 Datasets & Benchmarks (arXiv:2006.07869v4),

with the environment from [semitable/lb-foraging](https://github.com/semitable/lb-foraging)
(`pip install lbforaging`, v2.0.0, with lbforaging v1's item-level rule restored; see below).

| name | algorithm | source of its hyper-parameters |
|---|---|---|
| `qmix` | QMIX (Rashid et al. 2018) | benchmark paper, appendix I table 28 (LBF, parameter sharing) |
| `qplex` | QPLEX (Wang et al. 2021) | mixer: QPLEX paper, table 2; training: QMIX's LBF settings (see below) |
| `mappo` | MAPPO (Yu et al. 2022) | benchmark paper, appendix I table 24 (LBF, parameter sharing) |
| `macpf` | MACPF (Wang, Ye & Lu 2023) | MACPF paper, appendix C (not in the benchmark) |
| `qmix_atd`, `qplex_atd`, `mappo_atd` | the same with ATD(λ) (Deng, Wang & Zhang 2026) | base learner + ATD paper table 3 / sec. 4 |
| `qmix_l0.4`, `qplex_l0.4`, `mappo_l0.95` | fixed-λ controls | see "Fixed-λ controls" |

```bash
pip install torch numpy matplotlib lbforaging
python -m lbf_bench.run --algo qmix_atd --task 10x10-3p-3f --seed 0 --out results/lbf   # one run
python -m lbf_bench.sweep --seeds 0 1 2 --workers 4 --out results/lbf                   # the whole study
python -m lbf_bench.analysis --root results/lbf                                          # tables + figures
pytest -q tests/test_lbf_bench.py
```

## Environment and tasks

The seven tasks of the benchmark (sec. 3.4, table 3), each with a 50-step time limit (appendix A,
`env_args.time_limit=50`). Agent levels are 1–2. `-2s` means each agent sees only a 5×5 window (sight 2).
`-c` means every item needs all agents to load it at once.

`8x8-2p-2f-c`, `8x8-2p-2f-2s-c`, `10x10-3p-3f`, `10x10-3p-3f-2s`, `15x15-3p-5f`, `15x15-4p-3f`, `15x15-4p-5f`

* **Reward.** All learners receive the *common* reward, the sum of the agents' rewards (EPyMARL's
  `common_reward=True`). lbforaging normalises rewards so an episode in which every item is collected
  returns 1.
* **Observations.** Each agent gets its lbforaging feature vector (x, y, level per item and per agent;
  -1, -1, 0 when an entity is not visible) plus a one-hot agent id. The global state used by the mixers,
  centralised critics and the ATD estimator is all agents' observations concatenated, as in EPyMARL's
  wrapper.
* **Item levels as in lbforaging v1.** The benchmark used lbforaging 1.0.x (task ids `Foraging-…-v1`).
  Let S be the sum of the three lowest agent levels. v1 draws a non-cooperative item's level from 1…S−1
  (`randint(1, S)`, exclusive upper bound); v2.0.0 draws it from 1…S. With three agents, v2 raises the
  share of items that only *all* agents together can load from about 2% to about 25% (and the share that
  needs more than one agent from 45% to 58%). `env.V1ItemLevels` restores v1's rule; everything else is
  lbforaging 2.0.0's code. Cooperative items are S in both versions, so the two `-c` tasks are unaffected.
  This was found after seed 0 had run with v2's rule: MAPPO and QMIX were far below the paper on the
  non-cooperative tasks (e.g. MAPPO 0.12 vs. 0.77 on `15x15-3p-5f`, QMIX 0.12 vs. 0.84 on `10x10-3p-3f`)
  and in line with it on the cooperative ones. Those 50 runs are kept in `results/lbf_v2items/` and are
  not part of the study.
* **Termination vs time limit.** lbforaging reports both as `done`. Here they are separated: an episode
  *terminates* when every item has been collected and is *truncated* at the time limit. Value targets
  bootstrap on truncation and not on termination.

## Protocol (benchmark paper, sec. 4.1 and 4.3)

* **Budget.** Off-policy algorithms (QMIX, QPLEX, MACPF) train for 2M environment steps, on-policy ones
  (MAPPO) for 20M.
* **Evaluation.** 41 evaluations at equal intervals (t = 0 included), each the mean return of 100 episodes.
  Value-based agents evaluate ε-greedily with their evaluation ε. MAPPO samples its policy, as the paper
  does. MACPF acts greedily with its independent policies, as in its paper's code.
* **Seeds.** 0, 1, 2. The paper used five. Three keep the whole study at about 2.5 days on 4 CPU cores.
* **Metrics.**
  * *Maximum return*: the evaluation step whose mean over seeds is highest, reported as mean ± 95% CI
    over seeds at that step.
  * *Average return*: the mean over all 41 evaluations.
  * The paper re-evaluates saved models at the best step; here that step's own 100-episode evaluation is
    used.

## Hyper-parameters

| | QMIX | QPLEX | MAPPO | MACPF |
|---|---|---|---|---|
| network | FC 64 → GRU 64 → FC | FC 64 → GRU 64 → FC | FC 128 → FC 128 → FC (no RNN) | FC 64 → GRU 64 → FC, + correction nets 64 (ELU) |
| optimiser, lr | Adam 3e-4 | Adam 3e-4 | Adam 3e-4 (actor and critic) | RMSprop 5e-4 |
| reward standardisation | yes | yes | no | no |
| exploration | ε 1 → 0.05 over 200k steps | ε 1 → 0.05 over 200k steps | sample the policy, entropy 0.001 | dependent policy mixed with uniform, ε 1 → 0.05 over 50k |
| evaluation | ε = 0.05 | ε = 0.05 | sample | greedy (independent policies) |
| target network | soft, τ = 0.01 per update | soft, τ = 0.01 per update | critic soft, τ = 0.01 | hard, every 200 episodes |
| data | 5000-episode replay, 32 episodes per update, 1 update per episode | same as QMIX | 10 parallel episodes per update, 4 epochs, clip 0.2 | 5000-episode replay, 32 episodes, 1 update per episode |
| targets | 1-step Double-Q | 1-step Double-Q | critic: 5-step returns; actor: 5-step advantage | soft TD(λ = 0.8), α 0.5 → 0.05 over 200k steps |
| mixer | QMIX hypernetworks (embed 32, hypernet 64, 2 layers) | duplex dueling; 1-layer w, b, λ, φ, υ; 4 heads; stop-gradient on Aᵢ | – | hypernetwork mixer without non-linearity (embed 64) |
| parameter sharing | yes (+ agent id) | yes | yes | yes |

Gradient-norm clip 10 and γ = 0.99 everywhere.

**Where each value comes from:**

1. **QMIX and MAPPO: the benchmark paper.** The tuned LBF values are in appendix I, table 28 (QMIX) and
   table 24 (MAPPO), with parameter sharing; the earlier arXiv versions only give ranges. Details neither
   version states come from EPyMARL, the authors' code: episode batches, Double-Q, MSE losses, n-step critic
   targets from a target critic, PPO epochs over the whole batch without minibatches, and no advantage
   normalisation.
2. **QPLEX: mixer from its own paper, training settings from QMIX.** The benchmark doesn't include QPLEX.
   The mixer follows QPLEX's table 2 (StarCraft column). The transformation is eq. 7's `wᵢ(s)Qᵢ + bᵢ(s)`; the
   paper mentions Qatten's transformation for SMAC but leaves its details to the code. QPLEX's own training
   settings (appendix B.2: PyMARL defaults, ε 1 → 0.05 over 50k steps, RMSprop lr 5e-4, two updates per
   episode, hard target every 200 episodes) never got above a return of 0 on LBF: 0.00 after 700k steps on
   `8x8-2p-2f-c`, where QMIX had reached 0.42 by 1M. Its greedy policy locked in before it had found food. On
   the user's decision QPLEX therefore uses the benchmark's LBF settings for QMIX, so QMIX and QPLEX differ
   only in the mixer.
3. **MACPF: its own paper.** Appendix C.2 (SMAC) gives the learning rate, network sizes, target period and
   α schedule; appendices C.1 and E give the hypernetwork mixer. Values not in the paper come from the
   authors' PyMARL code (`dfop.yaml`): RMSprop, TD(λ) 0.8, batch and buffer sizes, ε-mixture exploration and
   the last action as an input.
4. **ATD(λ): its own paper (sec. 4.1–4.3, appendix G, algorithm 1).**
   * The estimator ω is MLP(64) → GRU(64) → linear over [state, one-hot joint action], and λ = sigmoid(ω).
   * It is trained with BCE (eq. 8), D_on labelled 1 and D_off labelled 0, using Adam at lr 1e-3.
   * D_off is 50× larger than D_on, the paper's best ratio.
   * One estimator step on 32 episodes from each buffer per target-update period. Each transition's λ is
     cached and the cache is cleared at the same period. For QMIX, whose target is soft-updated, the period
     is 200 episodes, the PyMARL hard-update interval the ATD paper used.
   * In QMIX- and QPLEX-ATD, D_on is the newest 100 episodes of the replay and D_off the rest of it.
   * The TD targets are the λ-returns of eq. 10.
   * In MAPPO-ATD, D_on is the current batch and D_off the previous 50 batches. The critic regresses on
     ATD λ-returns and the actor's advantages are unchanged (sec. 4.3). The estimator takes one step per
     PPO update, since its target critic is updated at every update.
   * Every ATD variant otherwise keeps its base learner's settings, so ATD is the only difference.

### Fixed-λ controls

Base QMIX, QPLEX and MAPPO use 1-step or 5-step targets, so ATD changes two things at once: it uses
λ-returns at all, and it picks λ per state-action pair. The controls use λ-returns with a *fixed* λ:

* **λ = 0.4 for QMIX and QPLEX**: the `td_lambda` of the ATD paper's own QMIX and QPLEX baselines (table 3).
* **critic λ = 0.95 for MAPPO**: MAPPO's GAE λ (Yu et al. 2022).

## Grid search

None. Every value comes from one of the papers above.
