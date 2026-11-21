"""Classical solvers: exact brute force, SCIP MIQP benchmark, and swap local search / repair.

Everything works in the factor form Sigma = B B^T + diag(d) and never builds the dense
covariance, so it scales to n = 10 000. Internally we minimise the *scaled* objective

    F(x) = ||B^T x||^2 + d.x - c * mu.x,      c = (1 - q) K / q

which is the official f(x) = q/K^2 * F(x) (binary x gives d_i x_i^2 = d_i x_i). Scaling
by K^2/q keeps numbers O(1..100) instead of O(0.01), which is friendlier to SCIP's
tolerances and makes the local-search improvement threshold size-independent.
"""
import itertools

import numpy as np

_TOL = 1e-10


def _lin(inst, q):
    return inst.d - (1 - q) * inst.K / q * inst.mu


def scaled_objective(inst, x, q=0.5):
    x = np.asarray(x, float)
    y = inst.B.T @ x
    return float(y @ y + _lin(inst, q) @ x)


# --------------------------------------------------------------------------- exact

def brute_force(inst, q=0.5, chunk=200_000):
    """Enumerate every K-subset. Exact; practical up to C(n, K) of a few million (n <= 50)."""
    n, K = inst.n, inst.K
    onehot = np.eye(inst.n_sectors, dtype=np.int16)[inst.sector]
    lin = _lin(inst, q)
    combos = itertools.combinations(range(n), K)
    best_F, best_idx, n_feasible = np.inf, None, 0
    while True:
        flat = np.fromiter(itertools.chain.from_iterable(itertools.islice(combos, chunk)), np.int32)
        if flat.size == 0:
            break
        idx = flat.reshape(-1, K)
        y = inst.B[idx].sum(1)
        F = (y * y).sum(1) + lin[idx].sum(1)
        ok = (inst.price[idx].sum(1) <= inst.budget) & (onehot[idx].sum(1) <= inst.sector_cap).all(1)
        n_feasible += int(ok.sum())
        F[~ok] = np.inf
        k = int(F.argmin())
        if F[k] < best_F:
            best_F, best_idx = F[k], idx[k].copy()
    x = np.zeros(n, int)
    x[best_idx] = 1
    return x, {"extra": {"n_feasible": n_feasible, "n_subsets": _comb(n, K)}}


def _comb(n, k):
    from math import comb
    return comb(n, k)


def miqp(inst, q=0.5, time_limit=60.0, warm_start=None):
    """Convex MIQP solved by SCIP; returns the incumbent plus a proven optimality gap.

    Factor form keeps the quadratic part in m <= 15 continuous variables y = B^T x
    (epigraph z >= ||y||^2), instead of an n x n dense matrix.
    """
    from pyscipopt import Model, quicksum

    n, m_f = inst.B.shape
    lin = _lin(inst, q)
    mdl = Model("portfolio")
    mdl.hideOutput()
    x = [mdl.addVar(vtype="B", name=f"x{i}") for i in range(n)]
    y = [mdl.addVar(lb=None, name=f"y{k}") for k in range(m_f)]
    z = mdl.addVar(lb=0, name="z")
    for k in range(m_f):
        mdl.addCons(y[k] == quicksum(float(inst.B[i, k]) * x[i] for i in range(n)))
    mdl.addCons(z >= quicksum(y[k] * y[k] for k in range(m_f)))
    mdl.addCons(quicksum(x) == inst.K)
    mdl.addCons(quicksum(float(inst.price[i]) * x[i] for i in range(n)) <= inst.budget)
    for s in range(inst.n_sectors):
        idx = np.flatnonzero(inst.sector == s)
        if len(idx) > inst.sector_cap:
            mdl.addCons(quicksum(x[i] for i in idx) <= inst.sector_cap)
    mdl.setObjective(z + quicksum(float(lin[i]) * x[i] for i in range(n)), "minimize")
    mdl.setParam("limits/time", float(time_limit))

    if warm_start is not None:
        sol = mdl.createSol()
        yv = inst.B.T @ warm_start
        for i in range(n):
            mdl.setSolVal(sol, x[i], float(warm_start[i]))
        for k in range(m_f):
            mdl.setSolVal(sol, y[k], float(yv[k]))
        mdl.setSolVal(sol, z, float(yv @ yv))
        mdl.addSol(sol)

    mdl.optimize()
    xs = np.array([round(mdl.getVal(v)) for v in x], int)
    if not inst.is_feasible(xs):   # SCIP tolerance (1e-6) is looser than is_feasible's
        xs = local_search(inst, repair(inst, xs, q), q)
    scale = q / inst.K**2
    return xs, {"extra": {"status": mdl.getStatus(), "gap": mdl.getGap(),
                          "lower_bound": scale * mdl.getDualbound()}}


# ------------------------------------------------------------- swap local search

class _Moves:
    """Objective deltas and violation after every single add / remove / swap move."""

    def __init__(self, inst, q):
        self.inst = inst
        self.lin = _lin(inst, q)
        self.nrm = (inst.B ** 2).sum(1)

    def _base(self, x):
        inst = self.inst
        S, U = np.flatnonzero(x), np.flatnonzero(x == 0)
        y = inst.B.T @ x
        a = 2 * inst.B @ y + self.lin          # gradient-like term per asset
        cnt = np.bincount(inst.sector[S], minlength=inst.n_sectors)
        return S, U, a, cnt, float(inst.price @ x)

    def violation(self, x):
        inst = self.inst
        cnt = np.bincount(inst.sector[x == 1], minlength=inst.n_sectors)
        return (max(0.0, inst.price @ x - inst.budget) / inst.budget
                + float(np.maximum(cnt - inst.sector_cap, 0).sum()))

    def swaps(self, x):
        """dF[i, j] and violation after swapping selected S[i] out for unselected U[j] in."""
        inst = self.inst
        S, U, a, cnt, P = self._base(x)
        dF = (a[U][None, :] - a[S][:, None] + self.nrm[U][None, :] + self.nrm[S][:, None]
              - 2 * inst.B[S] @ inst.B[U].T)
        sS, sU = inst.sector[S][:, None], inst.sector[U][None, :]
        cap = inst.sector_cap
        exc = float(np.maximum(cnt - cap, 0).sum())
        d_exc = (sS != sU) * (-(cnt[sS] > cap).astype(int) + (cnt[sU] >= cap).astype(int))
        P_new = P - inst.price[S][:, None] + inst.price[U][None, :]
        V = np.maximum(P_new - inst.budget, 0) / inst.budget + exc + d_exc
        return S, U, dF, V

    def adds(self, x):
        inst = self.inst
        S, U, a, cnt, P = self._base(x)
        exc = float(np.maximum(cnt - inst.sector_cap, 0).sum())
        V = (np.maximum(P + inst.price[U] - inst.budget, 0) / inst.budget
             + exc + (cnt[inst.sector[U]] >= inst.sector_cap))
        return U, a[U] + self.nrm[U], V

    def removes(self, x):
        inst = self.inst
        S, U, a, cnt, P = self._base(x)
        exc = float(np.maximum(cnt - inst.sector_cap, 0).sum())
        V = (np.maximum(P - inst.price[S] - inst.budget, 0) / inst.budget
             + exc - (cnt[inst.sector[S]] > inst.sector_cap))
        return S, -a[S] + self.nrm[S], V


def repair(inst, x, q=0.5, max_iter=10_000, use_objective=True):
    """Turn any 0/1 vector into a feasible portfolio.

    1. Fix the count to exactly K (greedy add/remove, violation first, then the tie-break).
    2. While budget or sector caps are violated, apply a swap that strictly reduces the
       violation. Tie-break: the best objective change (use_objective=True), or the largest
       violation reduction (use_objective=False). The objective-blind variant is the fair
       choice when measuring the quality of an upstream solver: the objective-aware one
       already optimises while it repairs.
    """
    mv = _Moves(inst, q)
    x = np.asarray(x).astype(int).copy()
    while x.sum() > inst.K:
        S, dF, V = mv.removes(x)
        x[S[np.argmin(1e6 * V + (dF if use_objective else -inst.price[S]))]] = 0
    while x.sum() < inst.K:
        U, dF, V = mv.adds(x)
        x[U[np.argmin(1e6 * V + (dF if use_objective else inst.price[U]))]] = 1
    for _ in range(max_iter):
        V0 = mv.violation(x)
        if V0 <= 0:
            return x
        S, U, dF, V = mv.swaps(x)
        if not use_objective:
            dF = V
        dF = np.where(V < V0 - 1e-12, dF, np.inf)
        if not np.isfinite(dF).any():
            break
        i, j = np.unravel_index(np.argmin(dF), dF.shape)
        x[S[i]], x[U[j]] = 0, 1
    return _cheapest_feasible(inst)


def _cheapest_feasible(inst):
    """Constructive fallback: cheapest assets that respect the sector caps."""
    x = np.zeros(inst.n, int)
    cnt = np.zeros(inst.n_sectors, int)
    for i in np.argsort(inst.price):
        if cnt[inst.sector[i]] < inst.sector_cap:
            x[i], cnt[inst.sector[i]] = 1, cnt[inst.sector[i]] + 1
            if x.sum() == inst.K:
                break
    if not inst.is_feasible(x):
        raise RuntimeError("no feasible portfolio found by the constructive fallback")
    return x


def local_search(inst, x, q=0.5, max_iter=100_000):
    """Steepest-descent over feasibility-preserving swaps. Input must be feasible."""
    mv = _Moves(inst, q)
    x = np.asarray(x).astype(int).copy()
    for _ in range(max_iter):
        S, U, dF, V = mv.swaps(x)
        dF = np.where(V <= 0, dF, np.inf)
        k = np.argmin(dF)
        if dF.flat[k] >= -_TOL:
            break
        i, j = np.unravel_index(k, dF.shape)
        x[S[i]], x[U[j]] = 0, 1
    return x


def multistart(inst, q=0.5, restarts=20, seed=0, starts=None):
    """Classical heuristic: repair + local search from random (or supplied) starting points."""
    rng = np.random.default_rng(seed)
    if starts is None:
        starts = []
        for _ in range(restarts):
            s = np.zeros(inst.n, int)
            s[rng.choice(inst.n, inst.K, replace=False)] = 1
            starts.append(s)
    best, best_F = None, np.inf
    for s in starts:
        x = local_search(inst, repair(inst, s, q), q)
        F = scaled_objective(inst, x, q)
        if F < best_F:
            best, best_F = x, F
    return best, {"evaluations": len(starts)}
