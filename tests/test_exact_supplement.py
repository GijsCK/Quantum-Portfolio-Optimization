"""Independent exhaustive checks of total-reversal pruning and the size join."""
from itertools import combinations
import unittest

import numpy as np

from classical import BOUND_GUARD, FactorModel, solve
from exact_supplement import affordable_groups, verify
from portfolio_generator import generate


class ReversalCertificateTests(unittest.TestCase):
    def test_affordable_subsets_include_ties_at_the_limit(self):
        prices = np.array([0.0, 2.0, 1.0, 1.0, 3.0])
        groups = affordable_groups(np.arange(5), prices, 2.0)
        got = {tuple(sorted(selection)) for group in groups.values() for _, selection in group}
        expected = {s for k in range(6) for s in combinations(range(5), k)
                    if sum(prices[i] for i in s) <= 2.0}
        self.assertEqual(got, expected)

    def test_signed_net_additions_match_full_direct_enumeration(self):
        inst = generate(12, 85)
        options = []
        for assets in combinations(range(inst.n), inst.K):
            portfolio = np.zeros(inst.n, dtype=int)
            portfolio[list(assets)] = 1
            if inst.is_feasible(portfolio):
                options.append((inst.objective(portfolio), list(assets)))
        expected = min(score for score, _ in options)
        # Use the worst feasible upper bound, so recovering the global optimum
        # requires actually visiting other holdings. Test both net directions.
        worst, start = max(options)
        model = FactorModel(inst)
        for threshold in (model.c.min() - 0.001, model.c.max() + 0.001):
            with self.subTest(threshold=threshold):
                cert = {"exposure": np.zeros(inst.B.shape[1]).tolist(),
                        "sector_multipliers": np.zeros(inst.n_sectors).tolist(),
                        "price_multiplier": 0., "threshold": threshold}
                lower = (threshold * inst.K + np.minimum(model.c - threshold, 0).sum()
                         - BOUND_GUARD)
                output = verify(inst, {"selected": start, "objective": worst,
                                       "certificate": cert, "tangent_bound": lower})
                self.assertAlmostEqual(expected, output["exact_objective"], places=13)

    def test_real_saved_type_certificate_matches_exhaustive_score(self):
        for seed in (75, 76, 77):
            inst = generate(20, seed)
            baseline = solve(inst)
            self.assertAlmostEqual(verify(inst, baseline)["exact_objective"],
                                   baseline["exact_objective"], places=13)


if __name__ == "__main__":
    unittest.main()
