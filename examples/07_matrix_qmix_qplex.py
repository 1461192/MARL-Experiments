"""Example 7 - matrix games that separate VDN, QMIX and QPLEX.

1. QMIX's two-step game (``two_stage_matrix``, map ``two_step``): agent 0 picks matrix 0 (always
   7) or matrix 1 (8 only if both play B). VDN's additive fit of matrix 1 gives at most 6.5, so it
   settles for 7; QMIX's monotonic, state-conditioned mixer fits matrix 1 exactly and gets 8.
2. The non-monotonic matrix (``matrix``, map ``nonmonotonic``; QTRAN / QPLEX's didactic game):
   optimum (A, A) = 8 is surrounded by -12. No monotonic mixer can represent it, so QMIX goes for
   the safe 0; QPLEX's duplex dueling mixer can represent it.

Every learner sees uniformly random joint actions (epsilon = 1) and 1-step targets, as in both
papers, and the greedy policy is played once afterwards. Also shown: the representation-only
check (free utilities + mixer regressed straight onto the payoff table).

    python examples/07_matrix_qmix_qplex.py            # 8 seeds, ~5 minutes on 4 cores
    VDN_QUICK=1 python examples/07_matrix_qmix_qplex.py
"""
from __future__ import annotations

from _common import QUICK  # noqa: E402

from vdn_lab.envs.matrix import MATRIX_GAMES
from vdn_lab.experiment.matrix_study import fit_payoffs, run_seeds, summarize, train_uniform

SEEDS = range(2 if QUICK else 8)
TWO_STEP = dict(steps=200 if QUICK else 3000, gamma=0.5, lr=2e-3)
NONMONO = dict(steps=200 if QUICK else 5000, gamma=0.0)


def main():
    print("== two-step game (optimum 8; matrix 0 always pays 7)")
    res = {a: run_seeds(train_uniform, SEEDS, a, "two_stage_matrix", "two_step", TWO_STEP["steps"],
                        gamma=TWO_STEP["gamma"], lr=TWO_STEP["lr"]) for a in ("vdn", "qmix")}
    print(summarize(res, 8.0))

    print("\n== non-monotonic matrix [[8,-12,-12],[-12,0,0],[-12,0,0]] (optimum 8)")
    res = {a: run_seeds(train_uniform, SEEDS, a, "matrix", "nonmonotonic", NONMONO["steps"],
                        gamma=NONMONO["gamma"]) for a in ("vdn", "qmix", "qplex")}
    print(summarize(res, 8.0))

    print("\n== representation only: regress Q_tot onto the non-monotonic table "
          "(MSE, greedy joint action; (0, 0) is optimal)")
    R = MATRIX_GAMES["nonmonotonic"]["payoffs"]
    for m, kw in [("vdn", {}), ("qmix", {}), ("qplex", dict(detach_advantage=False))]:
        fits = run_seeds(fit_payoffs, SEEDS, m, R, steps=100 if QUICK else 1500, **kw)
        print(f"{m:<6} " + "  ".join(f"{l:.2f}{a}" for l, a in fits))


if __name__ == "__main__":
    main()
