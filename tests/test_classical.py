import numpy as np
import pytest

from portfolio_generator import generate
from qportfolio.classical import brute_force, miqp, multistart, repair


@pytest.mark.parametrize("n", [10, 20, 50])
@pytest.mark.parametrize("seed", range(5))
def test_solvers_agree_with_brute_force(n, seed):
    inst = generate(n, seed)
    xb, _ = brute_force(inst)
    xm, _ = multistart(inst, restarts=10)
    xq, info = miqp(inst, time_limit=30)
    assert inst.is_feasible(xb)
    assert inst.objective(xm) == pytest.approx(inst.objective(xb), abs=1e-12)
    assert inst.objective(xq) == pytest.approx(inst.objective(xb), abs=1e-12)
    assert info["extra"]["status"] == "optimal"


@pytest.mark.parametrize("n", [10, 50, 200, 1000])
def test_repair_always_feasible(n):
    inst = generate(n, 0)
    rng = np.random.default_rng(0)
    for p in (0.0, 0.05, 0.3, 1.0):          # empty, sparse, dense, everything selected
        x = (rng.random(n) < p).astype(int)
        assert inst.is_feasible(repair(inst, x))
