# vdn_lab

A small, StarCraft-free testbed for **Value-Decomposition Networks** (Sunehag et al., 2017,
[arXiv:1706.05296](https://arxiv.org/abs/1706.05296)) and the baselines it is compared with, plus the
mixing-network successors **QMIX** (Rashid et al., 2018, [arXiv:1803.11485](https://arxiv.org/abs/1803.11485))
and **QPLEX** (Wang et al., 2021, [arXiv:2008.01062](https://arxiv.org/abs/2008.01062)), and the policy-gradient
learner **MAPPO** (PPO, Schulman et al., 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347), set up for
cooperative multi-agent games as in Yu et al., 2022, [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)).
Pure Python + PyTorch + NumPy; runs on one CPU core.

```
vdn_lab/
  envs/         base.py (grid world, egocentric FOV, rendering)  scene.py (full-map view)  switch.py  fetch.py  checkers.py
                matrix.py (matrix / two-stage matrix games)
  agents/       base.py (Agent interface)  networks.py  recurrent_dqn.py  vdn.py
                independent.py  centralized.py  baselines.py  replay.py
                mixers.py (QMIX / QPLEX mixing networks)  mixing.py (qmix / qplex agents)
                mappo.py (MAPPO / IPPO)  macpf.py (MACPF)
  experiment/   config.py  runner.py  benchmark.py  render.py
  cli.py        python -m vdn_lab ...
examples/       nine runnable examples      tests/   pytest suite
```

## Install / run

```bash
pip install torch numpy pillow matplotlib pytest       # CPU torch is enough
python -m vdn_lab list
python -m vdn_lab watch --env switch --map single --episode-len 30     # random agents, replay window
python -m vdn_lab train --env fetch --map open --agent vdn --steps 150000 \
       --episode-len 200 --lr 5e-4 --agent-kw eps_decay_steps=75000 warmup_steps=1000 target_update_every=100 \
       --final-sessions 3 --eval-visual window
python -m vdn_lab play runs/<run_dir> --sessions 3                     # opens the replay window
pytest -q
```

## What is implemented (and where it comes from in the paper)

**Network (sec. 4.1):** `Linear(32) → ReLU → LSTM(32) → ReLU → dueling(32)` per agent;
centralised agents encode each observation with its own linear layer, concatenate (2×32 = 64),
ReLU, one LSTM. **Learner (sec. 2.2/4.1):** DQN with replay + target network, truncated BPTT and
forward-view λ-returns over trajectories of 8, λ = 0.9, Adam (default lr 1e-4).

| name | what it is |
|---|---|
| `independent` | one recurrent DQN per agent, each trained on the team reward |
| `vdn` | per-agent Q's summed for training; greedy per agent at execution. `share_weights=True` → one agent-invariant net; `role_info=True` → one-hot id appended to each observation |
| `centralized` | single Q over the joint action space (8² = 64 outputs) |
| `centralized_vdn` | centralised trunk, summed per-agent heads |
| `random` | uniform random (floor / template) |
| `qmix` | VDN's per-agent nets + QMIX's monotonic mixing network (see below) |
| `qplex` | VDN's per-agent nets + QPLEX's duplex dueling mixing network (see below) |
| `mappo` | PPO policies per agent + a centralised critic (see below) |
| `ippo` | the same PPO learner, each agent's critic sees only its own observations |
| `macpf` | multi-agent conditional policy factorisation: dependent policies in training, independent ones at execution (see below) |

Not implemented: the information-channel variants (low/high-level communication) from Table 1.

### QMIX and QPLEX (`vdn_lab/agents/mixers.py`, `mixing.py`)

Both use the same per-agent recurrent nets and learner as `vdn` (so the three differ **only** in how the
per-agent `Q_i` are combined into `Q_tot`), a target copy of the mixer, and decentralised greedy execution.
The mixers need a global state; the agent interface only passes observations, so the state is **all
agents' observations concatenated** (in the matrix games this is exactly the state).

* **QMIX** (paper sec. 4): `Q_tot = w2ᵀ ELU(W1ᵀ q + b1) + b2`, mixing layer of 32 units. `W1`, `w2` come from
  single-layer hypernetworks of the state followed by an absolute value (non-negative weights ⇒
  `∂Q_tot/∂Q_i ≥ 0`); `b1` from one linear layer; `b2 = V(s)` from a two-layer hypernetwork with a ReLU.
  The bootstrap mixes each agent's own max, which is `max Q_tot` by monotonicity.
* **QPLEX** (paper sec. 4): `V_i = max Q_i`, `A_i = Q_i − V_i`; transformation `Q_i(τ,a_i) = w_i(s) Q_i + b_i(s)`,
  `V_i(τ) = w_i(s) V_i + b_i(s)` with `w_i > 0`; `Q_tot = Σ V_i(τ) + Σ λ_i(τ,a) A_i(τ,a_i)` where
  `λ_i = Σ_k key_k(s)·σ(agent_k(s))_i·σ(action_k(s,a))_i > 0` (multi-head attention over state and joint action,
  4 heads). Since `A_i ≤ 0` with equality at the greedy action, the per-agent argmax always maximises `Q_tot`.
  By default (`detach_advantage=True`) the λ-term is computed as in the authors' PyMARL code:
  `Q_tot = Σ Q_i(τ,a_i) + Σ (λ_i − 1)·stopgrad(A_i)`, which has the same value but gradients only reach the
  agents through `Σ Q_i`. It learns more reliably here (below); `False` uses the plain eq.-9 gradient.
  `double_q=True` (both agents) picks bootstrap actions with the online net.

Agent kwargs: `mixing_embed_dim` (both), `n_heads`, `attn_hidden`, `attn_layers`, `detach_advantage` (QPLEX),
`double_q`, plus everything `vdn` takes (`share_weights`, `role_info`, `DQNConfig` fields).

**Matrix games that separate them** (`examples/07_matrix_qmix_qplex.py`, tests in `tests/test_mixing.py`;
helpers in `vdn_lab/experiment/matrix_study.py`). As in both papers the learners see uniformly random joint
actions (ε = 1) with 1-step targets, then play the greedy policy; one round per episode. Greedy return, 8 seeds:

| game (optimum 8) | `vdn` | `qmix` | `qplex` |
|---|---|---|---|
| two-step game, `two_stage_matrix --map two_step` (QMIX sec. 6.1) | 7 in 8/8 | **8 in 8/8** | |
| non-monotonic `matrix --map nonmonotonic` `[[8,-12,-12],[-12,0,0],[-12,0,0]]` | 0 in 8/8 | 0 in 8/8 | **8 in 5/8** |

* Two-step: VDN's additive fit of matrix 1 peaks at 6.5 < 7, so it picks the safe matrix. Matrix 1 is
  monotonic, and QMIX fits it exactly with a mixer conditioned on which matrix is being played.
* Non-monotonic: no monotonic mixer can represent the table. Fitting `Q_tot` straight onto it (no RL, every
  cell weighted equally) bottoms out at MSE 50.6 (VDN) and 35.6 (QMIX) from every initialisation. QPLEX can
  represent it **but does not always find it**: with the exact gradient it reaches MSE 0 with greedy (A, A)
  from 2 of 8 initialisations. The usual failure is a local optimum where the agents' greedy action is B,
  `V_tot ≈ 4` splits the difference between the cells (A,A) = 8 and (B,B) = 0, and λ(A,A) shrinks
  toward 0, which removes the gradient `Q_i(A)` needs to overtake `Q_i(B)`. The stop-gradient default avoids
  that collapse more often (5/8 in the RL run). The other parameters I tried (lr, number of heads, 1-layer λ
  networks, Huber loss, ε-greedy data, longer training) did not make it reliable.

```bash
python examples/07_matrix_qmix_qplex.py        # both games, 8 seeds, ~5 min on 4 cores
python -m vdn_lab train --env two_stage_matrix --map two_step --agent qmix --steps 3000 --episode-len 2 \
       --eval-episode-len 2 --eval-every 0 --lr 2e-3 --agent-kw eps_start=1 eps_end=1 lam=0 seq_len=2 gamma=0.5 \
       warmup_steps=200 train_every=1 batch_size=64 target_update_every=50 loss=mse --final-sessions 1 --eval-visual text
python -m vdn_lab train --env matrix --map nonmonotonic --agent qplex --steps 5000 --episode-len 1 \
       --eval-episode-len 1 --eval-every 0 --lr 5e-3 --agent-kw eps_start=1 eps_end=1 lam=0 seq_len=1 gamma=0 \
       warmup_steps=200 train_every=1 batch_size=64 target_update_every=50 loss=mse --final-sessions 1 --eval-visual text
```
Episodes are one round long so every trajectory starts from a fresh LSTM state, the same as the greedy test.
With long episodes of repeated rounds and `seq_len=1`, the greedy action can drift as the LSTM state moves
into states it was never trained on.

### MAPPO and IPPO (`vdn_lab/agents/mappo.py`)

**Learner (PPO paper):** the clipped surrogate `L^CLIP = E[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]` (eq. 7) plus an
entropy bonus (eq. 9), GAE advantages (eq. 11–12), and K epochs of Adam on each new batch of on-policy data.
Each agent has a stochastic policy over its own observation history. At execution time it acts on its own
(greedy evaluation takes the most likely action). The critic is used only during training.

**Multi-agent setup (Yu et al. 2022, sec. 5).** Each of the paper's five suggestions maps to a default:

| sec. | suggestion | here (agent kwarg, default) |
|---|---|---|
| 5.1 | value normalisation | `value_norm=True`: running mean/std of the returns (debiased EMA as in the authors' `ValueNorm`); the critic regresses onto normalised returns, which are de-normalised for GAE |
| 5.2 | agent-specific global state for the critic | `critic_input="agent_specific"`: own observation, then the other agents' observations, then a one-hot id. `"concat"` is the CL variant, and `"local"` gives IPPO. The agent interface has no global state, so the union of the agents' views stands in for it |
| 5.3 | few epochs, no mini-batch splitting | `ppo_epochs=10`, `num_mini_batch=1` |
| 5.4 | clip ε ≤ 0.2 | `clip=0.2`, value predictions clipped with the same range (`value_clip`) |
| 5.5 | large batches | one update per `rollout_len=1000` env steps (5 episodes × 2 agents = 2000 samples) |

Other details follow the paper's appendix and code. `share_weights=True` uses one policy and one critic for
both agents (`role_info` adds the id to the policy input). Policy and critic are each
`LayerNorm → 64 → 64 → GRU(64) → LayerNorm → head`, trained by BPTT on 10-step chunks (`chunk_len`) that start
from the hidden state recorded during collection, with resets at episode starts. The rest: orthogonal init,
Huber value loss (δ = 10), advantage normalisation, separate Adam optimisers (lr 5e-4, eps 1e-5), grad-norm
clip 10, entropy coefficient 0.01, γ = 0.99, GAE λ = 0.95. The grid-world episodes end on a time limit, not in
a terminal state, so the last step of an episode bootstraps from `V(next_obs)` (`bootstrap_on_done=True`).
`ippo` is the same agent with `critic_input="local"`.

**MAPPO vs VDN on fetch** (`examples/08_fetch_mappo_vs_vdn.py`, full tables, plots and GIFs in
`results/mappo_vs_vdn/`). Settings: 200k steps, 200-step episodes, 3 seeds. VDN uses the hyper-parameters of
example 6 (its `open` numbers match `results/learner_comparison/fetch` exactly, same seeds). MAPPO and IPPO use
the defaults above, with no tuning on this task. Each cell is the greedy return of the final policy over 5
sessions, mean ± std over seeds; "AUC" is the mean training return over the whole run.

| map | `vdn` | `mappo` | `ippo` | train AUC vdn / mappo / ippo |
|---|---|---|---|---|
| `open` | 139.7 ± 19.6 | 198.0 ± 43.2 | **216.3 ± 30.2** | 74 / 141 / 151 |
| `single` (one corridor) | 81.0 ± 10.7 | 86.7 ± 16.5 | **104.1 ± 7.7** | 16 / 58 / 64 |

* Both PPO learners learn much faster than VDN: their training AUC is roughly 2× VDN's on `open` and
  roughly 4× on `single`. Each PPO run took about 200 s on one core, against about 430 s for VDN.
* VDN's greedy policies are all "lazy agent" solutions: in every seed on both maps one agent earns all the
  reward and the other mostly stands still. MAPPO split the work in all 3 `open` seeds but only 1 of 3 `single`
  seeds. IPPO split it in every seed and scored best on both maps.
* In this small two-agent game the centralised critic did not beat a local critic. Yu et al. likewise report
  IPPO close to MAPPO on many tasks. With 3 seeds the gap between MAPPO and IPPO is within noise.

```bash
python -m vdn_lab train --env fetch --map open --agent mappo --steps 200000 --episode-len 200 \
       --eval-episode-len 200 --eval-every 25000 --final-sessions 3 --eval-visual window
python examples/08_fetch_mappo_vs_vdn.py           # VDN vs MAPPO vs IPPO, open + single, 3 seeds, ~30 min on 4 cores
```

### MACPF (`vdn_lab/agents/macpf.py`)

Wang, Ye & Lu, *More Centralized Training, Still Decentralized Execution: Multi-Agent Conditional Policy
Factorization*, ICLR 2023 ([arXiv:2209.12681](https://arxiv.org/abs/2209.12681)), section 3, with details from
the authors' code ([PKU-RL/FOP-DMAC-MACPF](https://github.com/PKU-RL/FOP-DMAC-MACPF), `dfop_*` modules).

* **Conditional factorisation (sec. 3.1).** Maximum-entropy (FOP-style) actor-critic in which the joint policy
  is `π_jt(a|τ) = Π_i π_i^dep(a_i | τ_i, a_<i)`: agent `i` sees the actions already chosen by agents `0..i-1`
  during centralised training, so the policy class contains the optimal joint policy.
* **Decentralised execution (sec. 3.2).** `π_i^dep = π_i^ind + b_i^dep` (logits; stop-gradient on the independent
  part) and `Q_i^dep = Q_i^ind + c_i^dep`. Two joint critics (`Q_jt^dep`, `Q_jt^ind`, each with its own mixer) and
  four losses: soft TD(λ) for each critic, and `E_π[α log π_i − Q_i]` for each policy. The dependent joint policy
  collects the data, so the independent policies are learned off-policy. At execution only `π_i^ind` is used.
* **Networks.** Shared policy and critic: `Linear(64) → ReLU → GRU(64) → Linear` on `[obs, last action, id]`.
  Dependency correction: `ELU(Linear(64))` embeddings of preceding agents' `[id, action]`, summed / n, concatenated
  with the GRU state.
* **Mixer.** `mixer="fop"` (default) is FOP's decomposition `Q_jt = V_jt + Σ λ_i w_i (Q_i − V_i)`, with
  `V_i = α logsumexp(Q_i/α)`, Qatten weights `w_i` and QPLEX-style `λ_i(s,a)` (4 heads each). `mixer="qmix"` is
  the monotonic hypernetwork mixer that the released MACPF code uses. With it the first agent's utility is not
  identified, because the last agent's dependent Q can absorb the whole joint value. On fetch it did worse (below).
* **Hyper-parameters (appendix C / `dfop.yaml`).** RMSprop lr 5e-4 (actor and critic), α 0.99, eps 1e-5;
  grad clip 10; γ 0.99; TD(λ) 0.8; batches of 32 whole episodes; buffer of 5000 episodes; one training step per
  episode; targets copied every 200 episodes; entropy temperature 0.001; embeddings 64. Exploration: ε-mixture
  with uniform, 1.0 → 0.05 over 50k steps.

The arXiv / OpenReview PDF could not be fetched from the environment these were written in. The paper-level
description above is from the abstract and summaries, and the details are from the authors' code. Check them
against appendix C if exact numbers matter.

`act(explore=False)` uses the independent policies; `eval_policy="dep"` evaluates the dependent ones instead
(centralised execution, for diagnosis).

**MACPF vs MAPPO / IPPO / VDN on fetch** (`examples/09_fetch_macpf.py`, tables, plots and GIFs in
`results/macpf_vs_baselines/`). Settings are those of example 8: 200k steps, 200-step episodes, seeds 0–2. The
baseline runs are reused from `results/mappo_vs_vdn/`. Greedy return of the final policy over 5 sessions,
mean ± std over seeds. "dep" is the same MACPF agent acting with its dependent policies.

| map | `macpf` | `macpf` dep | `macpf_qmix` | `mappo` | `ippo` | `vdn` |
|---|---|---|---|---|---|---|
| `open` | 96.9 ± 21.5 | 91.6 ± 27.8 | 34.9 ± 40.7 | 198.0 ± 43.2 | **216.3 ± 30.2** | 139.7 ± 19.6 |
| `single` | 1.0 ± 1.4 | 2.0 ± 1.4 | 0.0 ± 0.0 | 86.7 ± 16.5 | **104.1 ± 7.7** | 81.0 ± 10.7 |

* With the paper's hyper-parameters and this budget, MACPF is the weakest learner on both maps. On `open` its
  training return is still rising at 200k steps. On `single` (one corridor) it never gets past picking up an item.
* The budget is a likely cause. One gradient step per 200-step episode gives MACPF only about 970 updates in
  200k steps, while the paper's SMAC runs take millions of steps. Exploration is the other: an ε-mixture with
  entropy temperature 0.001 has little to drive it through `single`'s corridor. I did not tune it for this task.
* The decentralised (independent) policies match the dependent ones (96.9 vs 91.6 on `open`). Distillation itself
  costs nothing here; the gap to the baselines comes from the learning speed of the dependent policy.
* The FOP mixer did better than the released code's QMIX mixer (96.9 vs 34.9 on `open`, 3 seeds).
* On the climbing matrix game, both mixers settle on the safe joint action. The dependent critic learns the
  conditional best response (agent 1 plays `a` after `a`), but agent 0's utility does not.

```bash
python -m vdn_lab train --env fetch --map open --agent macpf --steps 200000 --episode-len 200 \
       --eval-episode-len 200 --eval-every 25000 --final-sessions 3 --eval-visual window
python examples/09_fetch_macpf.py                  # MACPF (both mixers) + reused baselines, ~25 min per variant on 4 cores
```

**Games (sec. 4.2):** 8 actions (forward/back/left/right, rotate ×2, beam, noop); observation is an
egocentric RGB window, default 3×5×5 (4 ahead, 2 each side).

* `switch` – reach your own goal at the other end; maps `open`, `single`, `double`; +1 per goal, reset when both arrive.
* `fetch` – pick up (+3) and drop at base (+5), pick-up reappears; maps `open`, `single`, `double`.
* `checkers` – apples/lemons; player 0 scores ±10, player 1 ±1; resets when all apples are eaten.

**Matrix games** (two players, not in the paper; the usual small sanity checks for cooperative MARL):

* `matrix` – agent 0 picks a **row**, agent 1 a **column**, both get that cell's payoff. Games (`--map`):
  `climbing` (default), `penalty`, `coordination`, `nonmonotonic` (QTRAN's example), and the general-sum
  `prisoners_dilemma`, `stag_hunt`, `chicken`.
* `two_stage_matrix` – agent 0 first **picks one of the matrices** (agent 1's action is ignored in that
  step), then both play the chosen matrix as above. Games: `two_step` (QMIX's two-step game, default),
  `climbing_or_safe`. `observe_choice=False` hides the choice from agent 1; `choice_rewards=[...]` pays
  for the first-stage choice.

Custom games via `payoffs=`: an `R x C` matrix of common payoffs or `R x C x 2` per-player payoffs (team
reward = sum of both); for `two_stage_matrix` a list of such matrices (shapes may differ). Both agents
share `max(rows, columns[, matrices])` actions; a smaller dimension takes action `a mod size`.
Observations are tiny one-hot vectors (stage and, if visible, the chosen matrix). The game repeats round
after round for `episode_len` steps (the recurrent learners need 8-step trajectories), so use
`--agent-kw gamma=0` for a myopic one-shot learner on `matrix`.

```bash
python -m vdn_lab watch --env two_stage_matrix --map two_step --episode-len 10
python -m vdn_lab train --env matrix --map climbing --agent vdn --steps 20000 --episode-len 50 \
       --agent-kw gamma=0 warmup_steps=500 eps_decay_steps=10000 --eval-visual text
```

The exact map layouts in the paper are only given as pictures, so the maps here are my own
approximations of them, and the numbers are **not** comparable with the paper's figures.

## Game settings (field of view etc.)

Any game keyword is a config field: `env_kwargs={"layout": "double", "view_ahead": 2, "view_side": 1,
"view_behind": 2, "max_steps": ..., "random_orientation": True}`; reward sizes via `pickup_reward`,
`dropoff_reward`, `goal_reward`, `values`. Observation shape follows the FOV automatically. Custom ASCII
maps can be passed as `layout=[...]`.

## Visuals

`train_visual` / `eval_visual` ∈ `none | text | window | gif`, with `visual_delay`, `visual_every`,
`visual_color`, `visual_views`. Default is **headless training, then `final_eval_sessions` greedy
sessions** with the finished agent (optionally rendered); `train_visual_every=N` peeks at every N-th
training episode. Trained agents are saved to `<run_dir>/agent.pt` and replayed with `play`.

**Replay window (`window`, the default for `play` / `watch`).** Sessions with a finished agent are
recorded and then opened in a separate matplotlib window instead of being printed:

| control | button / key |
|---|---|
| pause / resume | `play`/`pause` button, `space` |
| replay the current session | `replay` button, `r` |
| previous / next session | `prev` / `next` buttons, `p` / `n` (sessions auto-advance) |
| replay speed (0.25x – 16x) | speed slider, `+` / `-`, `0` resets to 1x |
| scrub / step frames | frame slider, `←` / `→` |
| show / hide each agent's own view, field-of-view outlines | `v`, `f` |

Each frame shows the **whole environment**: the full map with numbered agents and heading arrows,
labelled objects (pick-up/drop-off, goals, apples/lemons), each agent's field of view outlined in its
colour, beams, a legend, and per-agent status (position, heading, last action, reward, e.g. whether it
carries the item). What each agent itself observes is shown as a small inset. GIFs (`gif` mode) use the
same picture plus a title line. Games describe their objects via the `scene_items` / `scene_legend` /
`scene_summary` / `agent_status` hooks of `GridWorld`.

Replays re-render the recorded game states, so they are exact and do not re-run the agent. The window
needs an interactive matplotlib backend (e.g. TkAgg / QtAgg); without a display the sessions are saved
as GIFs instead. During training, `train_visual="window"` still shows a simple live window, and `text`
remains available for terminal output.

## Adding an algorithm

```python
from vdn_lab import register_agent
from vdn_lab.agents import Agent

@register_agent("my_algo")
class MyAlgo(Agent):
    def reset_episode(self): ...
    def act(self, obs, explore=True): ...      # obs (n_agents, 3, H, W) -> actions (n_agents,)
    def observe(self, obs, actions, reward, next_obs, done): ...
    def update(self): return {}                # metrics dict when a learning step ran
```
Then use `agent="my_algo"` in `ExperimentConfig`, in a `Benchmark` variant, or `--agent my_algo` in the CLI.
Games work the same way via `@register_env` and subclassing `GridWorld`. See `examples/05_custom_agent.py`.

## Examples

| file | shows |
|---|---|
| `01_fetch_vdn_vs_baselines.py` | VDN vs independent vs centralised, 3 seeds, AUC table + curves |
| `02_switch_field_of_view.py` | game settings: three fields of view |
| `03_checkers_weight_sharing.py` | separate vs shared vs shared+role networks |
| `04_train_headless_then_watch.py` | headless training, then text/GIF/window sessions |
| `05_custom_agent.py` | plugging in new agents |
| `06_compare_learners.py` | VDN vs independent vs centralised on fetch, switch and checkers, 3 seeds, plots + behaviour stats → `results/learner_comparison/` |
| `07_matrix_qmix_qplex.py` | matrix games that separate VDN / QMIX / QPLEX, 8 seeds, plus a representation-only fit |
| `08_fetch_mappo_vs_vdn.py` | MAPPO vs VDN (and IPPO) on fetch `open` and `single`, 3 seeds → `results/mappo_vs_vdn/` |
| `09_fetch_macpf.py` | MACPF (FOP and QMIX mixers) vs MAPPO / IPPO / VDN on fetch `open` and `single` → `results/macpf_vs_baselines/` |

`VDN_QUICK=1` makes every example finish in seconds (nothing is learned; it checks the plumbing).

## Indicative results (single seed, so treat as a sanity check, not a replication)

Fetch `open`, 150k training steps, 200-step episodes, lr 5e-4. "Final" = mean return of 3 greedy sessions.

| agent | train return (last 10%) | final sessions |
|---|---|---|
| `vdn` | 113.7 | 131 |
| `vdn` + shared weights + role id | 123.6 | 131 |
| `independent` | 99.4 | 60.7 ± 36 |
| `centralized` (joint actions) | 83.4 | 112 |

All four learn this easy map, and the ordering matches the paper's, but with one seed the gaps are small
and not significant. The corridor maps (`--map single|double`) are where the paper reports the biggest
differences; run `examples/01_fetch_vdn_vs_baselines.py` (3 seeds) to get error bars. In the plain `vdn`
run all reward came from one agent (per-agent split `[0, 131]`), the "lazy agent" pattern the paper
describes; with shared weights + role id both agents took part (`[8, 123]`).
