"""Small exhaustive checks for the classical optimizer and its certificates."""
from itertools import combinations
import unittest

import numpy as np

from classical import BOUND_GUARD, FactorModel, _enumerate, solve
from portfolio_generator import GeneratorParams, generate


def all_portfolios(inst):
    for indices in combinations(range(inst.n), inst.K):
        x = np.zeros(inst.n, int)
        x[list(indices)] = 1
        if inst.is_feasible(x):
            yield x, inst.objective(x)


class TestClassical(unittest.TestCase):
    def test_official_and_enumerated_objective(self):
        # Include asymmetric correlations, a binding price cap and custom K.
        for seed in range(3):
            inst = generate(13, seed, GeneratorParams(card_frac=.25, sector_cap_mult=1.2))
            truth = min(score for _, score in all_portfolios(inst))
            result = solve(inst)
            x = np.zeros(inst.n, int)
            x[result["selected"]] = 1
            self.assertTrue(inst.is_feasible(x))
            self.assertAlmostEqual(result["objective"], inst.objective(x), places=14)
            self.assertAlmostEqual(result["exact_objective"], truth, places=14)
            self.assertLessEqual(result["tangent_bound"], truth)

    def test_dual_bound_and_flipping_cost_are_global(self):
        inst = generate(14, 5, GeneratorParams(card_frac=.3, sector_cap_mult=1.1))
        model = FactorModel(inst)
        feasible = list(all_portfolios(inst))
        true_x, optimum = min(feasible, key=lambda p: p[1])
        trial_x, trial_cost = sorted(feasible, key=lambda p: p[1])[3]
        for multiplier in [0., .00008, .001]:
            z = inst.B.T @ trial_x
            lb, reduced, _, _ = model.certificate(z, multiplier)
            for x, score in feasible:
                # Paying for every reversed sign remains a lower bound.
                reversals = np.abs(reduced[x != (reduced < 0)]).sum()
                self.assertLessEqual(lb + reversals, score + 1e-14)
            gap = trial_cost - lb + BOUND_GUARD
            fixed = np.flatnonzero(reduced < -gap)
            free = np.flatnonzero(np.abs(reduced) <= gap)
            self.assertTrue(np.all(true_x[fixed] == 1))
            ix = _enumerate(model, fixed, free, inst.K - len(fixed), np.flatnonzero(trial_x))
            self.assertAlmostEqual(model.score(ix), optimum, places=14)

    def test_residual_certificate_independently_replays(self):
        inst = generate(100, 0)
        result = solve(inst)
        cert = result["certificate"]
        z = np.array(cert["exposure"])
        model = FactorModel(inst)
        reduced = (model.c + 2 * model.a * (inst.B @ z)
                   + cert["price_multiplier"] * inst.price
                   + np.asarray(cert["sector_multipliers"])[inst.sector] - cert["threshold"])
        lb = (-model.a * np.dot(z, z) - cert["price_multiplier"] * (inst.budget + 1e-9)
              - inst.sector_cap * sum(cert["sector_multipliers"]) + cert["threshold"] * inst.K
              + np.minimum(reduced, 0).sum() - BOUND_GUARD)
        self.assertAlmostEqual(lb, result["tangent_bound"], places=14)
        self.assertEqual(result["exact_method"], "reduced-cost enumeration")
        self.assertLessEqual(result["exact_objective"], result["objective"] + 1e-14)


if __name__ == "__main__":
    unittest.main()
