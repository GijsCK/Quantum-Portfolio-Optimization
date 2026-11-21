"""Size-independent quality measures, relative to the exact optimum and to random guessing."""
import numpy as np

from .classical import repair


def random_feasible_mean(inst, samples=200, seed=0, q=0.5):
    """Mean objective of random K-portfolios made feasible by the objective-blind repair.

    Rejection sampling of feasible portfolios does not work at large n: the budget is
    K x median price while lognormal prices average ~1.38 x median, so at n = 1000 only
    ~1 in 20 000 random K-subsets is affordable. The repaired-random reference is also exactly
    what the "random" control method produces, so that control scores ~0 by construction.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        x = np.zeros(inst.n, int)
        x[rng.choice(inst.n, inst.K, replace=False)] = 1
        vals.append(inst.objective(repair(inst, x, q, use_objective=False), q))
    return float(np.mean(vals))


def normalised_score(f, f_opt, f_rand):
    """1 = optimal, 0 = as good as a repaired random portfolio, < 0 = worse than that."""
    return (f_rand - f) / (f_rand - f_opt)
