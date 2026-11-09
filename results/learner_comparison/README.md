# VDN vs independent vs centralised learners

Produced by `python examples/06_compare_learners.py` (re-plot from saved runs: `--replot`).

| game | map | training steps | seeds |
|---|---|---|---|
| fetch | `open` | 200k | 0, 1, 2 |
| switch | `single` (one corridor) | 200k | 0, 1, 2 |
| checkers | `default` | 150k | 0, 1, 2 |

The settings are the same for every game: 200-step episodes, lr 5e-4, ε 1→0.05 over half the budget,
1k warm-up, target update every 100 gradient steps. Greedy evaluation runs every 1/8 of the budget, and
the final behaviour stats come from 5 greedy sessions.

**Checkers role information:** VDN and the independent learners use one shared network with a one-hot
role id; the centralised learner has a separate input encoder per player and picks the joint action,
so the role is built into its architecture.

Each `<game>/` folder contains:

* `returns.png`: training return (left), greedy evaluation return (middle), per-agent reward of the final greedy policy (right)
* `summary.md` / `summary.json`: tables; the JSON also has idle/beam/noop rates and cells visited per agent and seed
* `<learner>_best_seed_000.gif`: 120 greedy steps of each learner's best seed
* `runs/<learner>_s<seed>_*/`: config, full result, trained `agent.pt` (`python -m vdn_lab play <dir>`)

## Results (greedy return of the final policy, mean ± std over 3 seeds)

| learner | fetch | switch | checkers |
|---|---|---|---|
| vdn | 139.7 ± 19.6 | **12.7 ± 6.9** | **619 ± 52** |
| independent | **140.9 ± 10.9** | 12.3 ± 8.6 | 409 ± 64 |
| centralized | 89.0 ± 65.3 | 0.3 ± 0.5 | 154 ± 156 |

The training-curve area (AUC) puts VDN first in all three games (fetch 74 vs 65 vs 22, switch 6.6 vs 1.8 vs 0.5,
checkers 217 vs 173 vs 50). With 3 seeds the VDN and independent learners are not separable on fetch or on final switch performance.
