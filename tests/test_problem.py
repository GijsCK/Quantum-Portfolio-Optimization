import numpy as np
import pytest

from portfolio_generator import generate
from qportfolio.problem import build_qubo


def test_objective_only_qubo_is_official_objective():
    inst = generate(20, 3)
    qb = build_qubo(inst, cardinality=False, budget=None, sector=None)
    X = np.random.default_rng(0).integers(0, 2, (100, 20))
    assert np.allclose(qb.energy(X), [inst.objective(x) for x in X])


def test_cardinality_penalty_vanishes_on_weight_K():
    inst = generate(20, 1)
    with_pen = build_qubo(inst, budget=None, sector=None)
    without = build_qubo(inst, cardinality=False, budget=None, sector=None)
    x = np.zeros(20, int)
    x[[3, 11]] = 1
    assert with_pen.energy(x) == pytest.approx(without.energy(x))


@pytest.mark.parametrize("budget", [("unbalanced", 0.1, 0.03), ("slack", 4, 3), None])
def test_ising_diagonal_matches_qubo(budget):
    qb = build_qubo(generate(10, 0), budget=budget)
    op, offset = qb.to_ising()
    diag = np.real(op.to_matrix(sparse=True).diagonal()) + offset
    assert np.allclose(diag, qb.all_energies())
