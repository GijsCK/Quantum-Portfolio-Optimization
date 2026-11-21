"""Size-independent quality measures, relative to the exact optimum and to random guessing."""
import numpy as np

from .problem import feasible_mask


def random_feasible_mean(inst, samples=2000, seed=0, q=0.5):
    """Mean objective of uniformly random *feasible* K-portfolios (rejection sampling)."""
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < samples:
        X = np.zeros((4 * samples, inst.n), int)
        rows = np.repeat(np.arange(4 * samples), inst.K)
        X[rows, np.argsort(rng.random((4 * samples, inst.n)), 1)[:, :inst.K].ravel()] = 1
        X = X[feasible_mask(inst, X)]
        vals += [inst.objective(x, q) for x in X[: samples - len(vals)]]
    return float(np.mean(vals))


def normalised_score(f, f_opt, f_rand):
    """1 = optimal, 0 = as good as a random feasible portfolio, < 0 = worse than random."""
    return (f_rand - f) / (f_rand - f_opt)
