# vdn_lab

A small, StarCraft-free testbed for **Value-Decomposition Networks** (Sunehag et al., 2017,
[arXiv:1706.05296](https://arxiv.org/abs/1706.05296)) and the baselines it is compared with.
Pure Python + PyTorch + NumPy; runs on one CPU core.

```
vdn_lab/
  envs/         base.py (grid world, egocentric FOV, rendering)  switch.py  fetch.py  checkers.py
  agents/       base.py (Agent interface)  networks.py  recurrent_dqn.py  vdn.py
                independent.py  centralized.py  baselines.py  replay.py
  experiment/   config.py  runner.py  benchmark.py  render.py
  cli.py        python -m vdn_lab ...
examples/       five runnable examples      tests/   pytest suite (40 tests)
```

## Install / run

```bash
pip install torch numpy pillow matplotlib pytest       # CPU torch is enough
python -m vdn_lab list
python -m vdn_lab watch --env switch --map single --episode-len 30     # random agents, ASCII view
python -m vdn_lab train --env fetch --map open --agent vdn --steps 150000 \
       --episode-len 200 --lr 5e-4 --agent-kw eps_decay_steps=75000 warmup_steps=1000 target_update_every=100 \
       --final-sessions 3 --eval-visual text
python -m vdn_lab play runs/<run_dir> --sessions 3 --visual gif
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

Not implemented: the information-channel variants (low/high-level communication) from Table 1.

**Games (sec. 4.2):** 8 actions (forward/back/left/right, rotate ×2, beam, noop); observation is an
egocentric RGB window, default 3×5×5 (4 ahead, 2 each side).

* `switch` – reach your own goal at the other end; maps `open`, `single`, `double`; +1 per goal, reset when both arrive.
* `fetch` – pick up (+3) and drop at base (+5), pick-up reappears; maps `open`, `single`, `double`.
* `checkers` – apples/lemons; player 0 scores ±10, player 1 ±1; resets when all apples are eaten.

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
