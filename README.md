# vdn_lab

A small, StarCraft-free testbed for **Value-Decomposition Networks** (Sunehag et al., 2017,
[arXiv:1706.05296](https://arxiv.org/abs/1706.05296)) and the baselines it is compared with, plus the
mixing-network successors **QMIX** (Rashid et al., 2018, [arXiv:1803.11485](https://arxiv.org/abs/1803.11485))
and **QPLEX** (Wang et al., 2021, [arXiv:2008.01062](https://arxiv.org/abs/2008.01062)).
Pure Python + PyTorch + NumPy; runs on one CPU core.

```
vdn_lab/
  envs/         base.py (grid world, egocentric FOV, rendering)  scene.py (full-map view)  switch.py  fetch.py  checkers.py
                matrix.py (matrix / two-stage matrix games)
  agents/       base.py (Agent interface)  networks.py  recurrent_dqn.py  vdn.py
                independent.py  centralized.py  baselines.py  replay.py
                mixers.py (QMIX / QPLEX mixing networks)  mixing.py (qmix / qplex agents)
  experiment/   config.py  runner.py  benchmark.py  render.py
  cli.py        python -m vdn_lab ...
examples/       seven runnable examples     tests/   pytest suite (75 tests)
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
