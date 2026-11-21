"""Independent feasible classical solver and global optimality checks.

Only exact binary portfolios enter the returned primal benchmark. Fractional
portfolios help find good directions and lower bounds in the relaxation.
"""
from itertools import combinations, islice
from math import comb
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist


# The first tolerance is the organizer's. The second guards floating-point
# lower bounds and variable fixing conservatively in official objective units.
BUDGET_TOL = 1e-9
BOUND_GUARD = 1e-11


def _smallest(indices, costs, count):
    if count >= len(indices):
        return indices
    return indices[np.argpartition(costs[indices], count - 1)[:count]]


class FactorModel:
    def __init__(self, inst):
        self.inst = inst
        self.a = 0.5 / inst.K**2
        # On a binary portfolio x_i squared equals x_i. Folding the diagonal
        # into the linear term makes the relaxation tighter.
        self.c = self.a * inst.d - 0.5 * inst.mu / inst.K
        self.groups = [np.flatnonzero(inst.sector == s) for s in range(inst.n_sectors)]
        self.budget = inst.budget + BUDGET_TOL

    def cap_oracle(self, cost):
        # Anything ranked below the cap in its own sector has an alternative
        # at least as cheap. Choose the K cheapest among all surviving assets.
        inst = self.inst
        pool = np.concatenate([_smallest(g, cost, inst.sector_cap) for g in self.groups])
        if len(pool) < inst.K:
            raise ValueError("Sector capacity cannot accommodate K holdings.")
        return _smallest(pool, cost, inst.K)

    def score(self, ix):
        exposure = self.inst.B[ix].sum(axis=0)
        return float(self.c[ix].sum() + self.a * (exposure @ exposure))

    def budget_oracle(self, cost, cheapest):
        """Bracket the price cap between two integer, sector-feasible portfolios.

        The lower envelope of their linear costs is concave in the budget
        multiplier. Testing their line intersection usually locates its
        maximum in a handful of calls. Any tested price gives a valid bound.
        """
        price = self.inst.price
        high_price = self.cap_oracle(cost)
        high_total = price[high_price].sum()
        if high_total <= self.budget:
            return high_price, high_price, 0., 0.
        low_price = cheapest
        low_total = price[low_price].sum()
        if low_total > self.budget:
            raise ValueError("The cheapest sector-feasible K holdings exceed the budget.")
        for _ in range(50):
            multiplier = max(0., (cost[low_price].sum() - cost[high_price].sum())
                             / (high_total - low_total))
            penalized = cost + multiplier * price
            ix = self.cap_oracle(penalized)
            test_total = price[ix].sum()
            # A third line below the endpoint intersection tightens the
            # bracket. Matching its height certifies the breakpoint.
            if penalized[ix].sum() >= penalized[low_price].sum() - 1e-15:
                break
            if test_total <= self.budget:
                low_price, low_total = ix, test_total
            else:
                high_price, high_total = ix, test_total
        fraction_high = float((self.budget - low_total) / (high_total - low_total))
        return low_price, high_price, fraction_high, float(multiplier)

    def certificate(self, z, price_multiplier):
        """Return a scalar lower bound and a signed cost of reversing each bit."""
        inst = self.inst
        qcost = self.c + 2 * self.a * (inst.B @ z) + price_multiplier * inst.price
        pool = np.concatenate([_smallest(g, qcost, inst.sector_cap) for g in self.groups])
        costs = np.sort(qcost[pool])
        threshold = float(0.5 * (costs[inst.K - 1] + costs[min(inst.K, len(costs) - 1)]))
        sector_prices = np.zeros(inst.n_sectors)
        for s, group in enumerate(self.groups):
            if len(group) > inst.sector_cap:
                u = np.partition(qcost[group], [inst.sector_cap - 1, inst.sector_cap])
                middle = 0.5 * (u[inst.sector_cap - 1] + u[inst.sector_cap])
                sector_prices[s] = max(0., threshold - middle)
        # Relax all three constraints with their multipliers. Every allowed
        # integer holding has energy at least this constant + r @ x.
        r = qcost + sector_prices[inst.sector] - threshold
        constant = (-self.a * (z @ z) - price_multiplier * self.budget
                    - inst.sector_cap * sector_prices.sum() + threshold * inst.K)
        lower = float(constant + np.minimum(r, 0.).sum() - BOUND_GUARD)
        return lower, r, threshold, sector_prices


def _exchange(model, current, multiplier):
    """Take strictly improving feasible one-out, one-in steps."""
    inst = model.inst
    chosen = np.zeros(inst.n, bool)
    chosen[current] = True
    for step in range(100):
        held, other = np.flatnonzero(chosen), np.flatnonzero(~chosen)
        if not len(other):
            break
        z = inst.B[held].sum(axis=0)
        gradient = model.c + 2 * model.a * (inst.B @ z)
        cost = inst.price[held].sum()
        counts = np.bincount(inst.sector[held], minlength=inst.n_sectors)
        if inst.n > 1500:
            # Keep the large extension's candidate matrix small. Two rankings
            # retain both cheap exchanges and attractive unconstrained returns.
            adjusted = gradient + multiplier * inst.price
            held = np.union1d(_smallest(held, -gradient, 96), _smallest(held, -adjusted, 96))
            other = np.union1d(_smallest(other, gradient, 128), _smallest(other, adjusted, 128))
        # Exact, including the change in factor covariance between the assets.
        change = (gradient[other][None, :] - gradient[held][:, None]
                  + model.a * cdist(inst.B[held], inst.B[other], "sqeuclidean"))
        feasible = ((cost - inst.price[held, None] + inst.price[other]) <= model.budget)
        feasible &= ((counts[inst.sector[other]] < inst.sector_cap)[None, :]
                     | (inst.sector[held, None] == inst.sector[other]))
        change[~feasible] = np.inf
        i, j = np.unravel_index(np.argmin(change), change.shape)
        if change[i, j] >= -1e-14:
            break
        chosen[held[i]] = False
        chosen[other[j]] = True
    return np.flatnonzero(chosen)


def _optimize(model):
    inst = model.inst
    cheapest = model.cap_oracle(inst.price)
    if inst.price[cheapest].sum() > model.budget:
        raise ValueError("No admissible portfolio.")
    best = cheapest
    best_score = model.score(best)
    z = inst.B[best].sum(axis=0)
    current_linear = model.c[best].sum()
    best_lower = -np.inf
    for _ in range(36):
        gradient = model.c + 2 * model.a * (inst.B @ z)
        lo, hi, t, multiplier = model.budget_oracle(gradient, cheapest)
        candidate_score = model.score(lo)
        if candidate_score < best_score:
            best_score, best = candidate_score, lo
        lower, _, _, _ = model.certificate(z, multiplier)
        if lower > best_lower:
            best_lower, best_z, best_price = lower, z.copy(), multiplier
        current_value = current_linear + model.a * (z @ z)
        if current_value - lower <= 1e-9:
            break
        # This convex combination has the requested price and satisfies the
        # relaxed sector/count constraints. Line search is exact for a quadratic.
        target_z = (1 - t) * inst.B[lo].sum(axis=0) + t * inst.B[hi].sum(axis=0)
        target_linear = (1 - t) * model.c[lo].sum() + t * model.c[hi].sum()
        dz = target_z - z
        dc = target_linear - current_linear
        descent = -(dc + 2 * model.a * (z @ dz))
        curvature = 2 * model.a * (dz @ dz)
        alpha = np.clip(descent / curvature, 0., 1.) if curvature > 0 else float(descent > 0)
        if alpha < 1e-12:
            break
        z = z + alpha * dz
        current_linear += alpha * dc
    return _exchange(model, best, best_price), best_z, best_price


def _enumerate(model, fixed, free, picks, incumbent):
    """Exhaust a lexicographic subset stream; use the official score at the end."""
    inst = model.inst
    best = incumbent.copy()
    best_score = model.score(best)
    if picks == 0:
        return fixed.copy()
    fixed_price = inst.price[fixed].sum()
    fixed_counts = np.bincount(inst.sector[fixed], minlength=inst.n_sectors)
    fixed_z = inst.B[fixed].sum(axis=0)
    fixed_linear = model.c[fixed].sum()
    stream = combinations(free.tolist(), picks)
    # Bound the working memory even if many of the 1,000 holdings are free.
    batch_size = max(1, min(8192, 150000 // max(1, picks)))
    while chunk := list(islice(stream, batch_size)):
        rows = np.asarray(chunk, dtype=int)
        totals = fixed_price + inst.price[rows].sum(axis=1)
        valid = totals <= model.budget
        # The generator sums a dense dot product. Match its summation order
        # whenever numerical roundoff could change a boundary decision.
        for k in np.flatnonzero(np.abs(totals - model.budget) <= 1e-12 * max(1., model.budget)):
            bits = np.zeros(inst.n, int)
            bits[fixed] = bits[rows[k]] = 1
            valid[k] = inst.is_feasible(bits)
        for sector, available in enumerate(inst.sector_cap - fixed_counts):
            if available < picks:
                valid &= ((inst.sector[rows] == sector).sum(axis=1) <= available)
        rows = rows[valid]
        if len(rows):
            z = fixed_z + inst.B[rows].sum(axis=1)
            scores = fixed_linear + model.c[rows].sum(axis=1) + model.a * np.einsum("ij,ij->i", z, z)
            i = int(np.argmin(scores))
            if scores[i] < best_score:
                best_score = float(scores[i])
                best = np.concatenate((fixed, rows[i]))
    return np.sort(best)


def _official(inst, selected):
    x = np.zeros(inst.n, int)
    x[selected] = 1
    if len(selected) != inst.K or not inst.is_feasible(x):
        raise ArithmeticError("Classical audit produced a portfolio outside the official rules.")
    return inst.objective(x)


def solve(inst):
    """Benchmark and audit a default-q instance. All times are CPU-host wall time."""
    start = perf_counter()
    model = FactorModel(inst)
    selected, z, multiplier = _optimize(model)
    objective = _official(inst, selected)
    seconds = perf_counter() - start

    audit = perf_counter()
    lower, reduced, threshold, sector_prices = model.certificate(z, multiplier)
    gap = objective - lower + BOUND_GUARD
    if gap < 0:
        raise ArithmeticError("Primal objective fell below its global dual bound.")
    # At any feasible optimum, paying more than the full gap for one reversed
    # reduced-cost sign is impossible. Equality is conservatively left free.
    fixed = np.flatnonzero(reduced < -gap)
    free = np.flatnonzero(np.abs(reduced) <= gap)
    picks = inst.K - len(fixed)
    if not 0 <= picks <= len(free):
        raise ArithmeticError("The numerical fixing certificate excluded the incumbent.")
    unrestricted = inst.n <= 50 and comb(inst.n, inst.K) <= 2200000
    if unrestricted:
        fixed, free, picks = np.empty(0, int), np.arange(inst.n), inst.K
    count = comb(len(free), picks)
    bound_seconds = perf_counter() - audit
    exact_start = perf_counter()
    exact = None
    if unrestricted or count <= 150000:
        exact = _enumerate(model, fixed, free, picks, selected)
    exact_seconds = perf_counter() - exact_start
    exact_score = None if exact is None else _official(inst, exact)
    return {
        "selected": sorted(selected.tolist()),
        "objective": objective,
        "seconds": seconds,
        "lower_bound": lower if exact is None else exact_score,
        "tangent_bound": lower,
        "bound_seconds": bound_seconds,
        "exact_objective": exact_score,
        "exact_selected": None if exact is None else exact.tolist(),
        "exact_seconds": exact_seconds,
        "exact_method": ("full enumeration" if unrestricted else "reduced-cost enumeration")
                        if exact is not None else None,
        "free_assets": len(free),
        "residual_combinations": count,
        "certificate": {"exposure": z.tolist(), "price_multiplier": multiplier,
                        "sector_multipliers": sector_prices.tolist(), "threshold": threshold},
    }
