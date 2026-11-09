"""Shared hyper-parameters for the examples.

The paper trains for 50,000 *episodes* of 5,000 steps; these examples use ~1e-3 of that,
so they raise the learning rate and shorten the epsilon schedule. Set VDN_QUICK=1 to
smoke-test any example in seconds (it will not learn anything)."""
import os
import sys
from pathlib import Path

# make `python examples/<file>.py` work from anywhere without installing the package
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

QUICK = bool(os.environ.get("VDN_QUICK"))


def hp(steps: int, **extra):
    d = dict(lr=5e-4, eps_decay_steps=steps // 2, warmup_steps=min(1000, steps // 3), target_update_every=100)
    d.update(extra)
    return d


def budget(full: int) -> int:
    return 1_500 if QUICK else full
