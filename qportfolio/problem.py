"""Hand-built QUBO and Ising Hamiltonian for the constrained portfolio problem.

QUBO energy:  E(x) = x^T Q x + b.x + c   over x in {0,1}^n,  Q symmetric.

    objective   (q/K^2) x^T Sigma x - ((1-q)/K) mu.x          (the official f(x))
  + cardinality lam_card * (sum x - K)^2                        (equality -> quadratic penalty)
  + budget      unbalanced penalty on  h = (budget - price.x) / unit_price
  + sectors     unbalanced penalty on  h_s = cap - sum_{i in s} x_i  (binding sectors only)

The unbalanced penalty  -l1*h + l2*h^2  (Montanez-Barrera et al., 2024) encodes the
inequality h >= 0 without slack qubits: it is positive when h < 0 (violation) and grows
quadratically, at the cost of slightly tilting the feasible landscape. Penalty strengths are
given in units of ``delta``, the largest change one asset can make to the objective, so the
same multipliers mean the same thing at every n.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Qubo:
    Q: np.ndarray     # (n, n) symmetric
    b: np.ndarray     # (n,)
    c: float
    delta: float      # objective scale used for the penalties
    n_assets: int     # the first n_assets variables are the portfolio; the rest are slack

    @property
    def n(self):
        return len(self.b)

    def portfolio(self, k):
        """Asset bits of basis-state index k (or an array of indices)."""
        k = np.asarray(k)
        return (k[..., None] >> np.arange(self.n_assets)) & 1

    def best_per_portfolio(self, E):
        """Minimise energies over the slack bits: entry k = min_s E(x = bits of k, s)."""
        return E.reshape(-1, 1 << self.n_assets).min(0)

    def energy(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return ((X @ self.Q) * X).sum(1) + X @ self.b + self.c

    def all_energies(self, chunk=1 << 18):
        """Energy of every bitstring; index k has x_i = bit i of k (Qiskit qubit order)."""
        n = self.n
        out = np.empty(1 << n)
        shifts = np.arange(n)
        for start in range(0, 1 << n, chunk):
            k = np.arange(start, min(start + chunk, 1 << n))
            out[k] = self.energy((k[:, None] >> shifts) & 1)
        return out

    def to_ising(self):
        """Ising form via x_i = (1 - Z_i)/2.  Returns (SparsePauliOp, constant offset).

        Folding x_i^2 = x_i:  E = sum_{i<j} J_ij x_i x_j + sum_i h_i x_i + c,
        with J_ij = 2 Q_ij and h_i = Q_ii + b_i.
        """
        from qiskit.quantum_info import SparsePauliOp

        n = self.n
        J = 2 * np.triu(self.Q, 1)
        h = np.diag(self.Q) + self.b
        Jsym = J + J.T
        z_coef = -h / 2 - Jsym.sum(1) / 4
        offset = self.c + h.sum() / 2 + J.sum() / 4
        terms = [("Z", [i], z_coef[i]) for i in range(n) if z_coef[i] != 0]
        iu, ju = np.nonzero(J)
        terms += [("ZZ", [i, j], J[i, j] / 4) for i, j in zip(iu, ju)]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(), float(offset)


def feasible_mask(inst, X):
    """Vectorised ``inst.is_feasible`` over the rows of X (same rules, same tolerance)."""
    X = np.atleast_2d(X)
    counts = X @ (np.eye(inst.n_sectors, dtype=int)[inst.sector])
    return ((X.sum(1) == inst.K) & (X @ inst.price <= inst.budget + 1e-9)
            & (counts <= inst.sector_cap).all(1))


def objective_scale(inst, q=0.5):
    """Largest objective change from adding one asset to a K-portfolio (upper bound)."""
    S = inst.Sigma()
    K = inst.K
    off = np.abs(S - np.diag(np.diag(S)))
    topk = -np.sort(-off, axis=1)[:, :K].sum(1)
    return float(np.max(q / K**2 * (np.diag(S) + 2 * topk) + (1 - q) / K * np.abs(inst.mu)))


def _add_square(Q, b, a, rhs, lam):
    """lam * (a.x - rhs)^2"""
    Q += lam * np.outer(a, a)
    b -= 2 * lam * rhs * a
    return lam * rhs**2


def _add_unbalanced(Q, b, a, rhs, unit, l1, l2):
    """-l1*h + l2*h^2 with h = (rhs - a.x)/unit, which encodes a.x <= rhs."""
    a, r = a / unit, rhs / unit
    Q += l2 * np.outer(a, a)
    b += l1 * a - 2 * l2 * r * a
    return -l1 * r + l2 * r**2


def build_qubo(inst, q=0.5, lam_card=1.0, budget=("unbalanced", 0.1, 0.03),
               sector=(1.0, 0.5), cardinality=True):
    """Penalty QUBO. Multipliers are in units of ``objective_scale(inst)``.

    budget: ("unbalanced", l1, l2)  -l1*h + l2*h^2, no extra qubits, distorts feasible states
            ("slack", lam, bits)    lam*(price.x + s - budget)^2 with s >= 0 on ``bits`` extra
                                    qubits; feasible states keep their order up to the slack
                                    resolution (budget / (2^bits - 1))
            None                    no budget term (leave it to post-selection / repair)
    cardinality=False drops the (sum x - K)^2 term, for use with a mixer that already
    preserves the number of selected assets. sector=None drops the sector caps.
    Slack qubits, if any, come after the n asset qubits.
    """
    n, K = inst.n, inst.K
    delta = objective_scale(inst, q)
    unit = inst.budget / K                  # one "median asset price"
    n_slack = budget[2] if budget is not None and budget[0] == "slack" else 0
    N = n + n_slack
    Q = np.zeros((N, N))
    b = np.zeros(N)
    Q[:n, :n] = q / K**2 * inst.Sigma()
    b[:n] = -(1 - q) / K * inst.mu
    c = 0.0
    if cardinality:
        a = np.r_[np.ones(n), np.zeros(n_slack)]
        c += _add_square(Q, b, a, K, lam_card * delta)
    if budget is not None and budget[0] == "unbalanced":
        _, l1, l2 = budget
        a = np.r_[inst.price, np.zeros(n_slack)]
        c += _add_unbalanced(Q, b, a, inst.budget, unit, l1 * delta, l2 * delta)
    elif budget is not None and budget[0] == "slack":
        _, lam, bits = budget
        step = inst.budget / (2**bits - 1)  # s = step * sum_k 2^k s_k covers [0, budget]
        a = np.r_[inst.price, step * 2.0 ** np.arange(bits)] / unit
        c += _add_square(Q, b, a, inst.budget / unit, lam * delta)
    if sector is not None:
        l1, l2 = sector
        for s in range(inst.n_sectors):
            mask = np.r_[(inst.sector == s).astype(float), np.zeros(n_slack)]
            if mask.sum() > inst.sector_cap:   # cap can only bind if the sector is bigger
                c += _add_unbalanced(Q, b, mask, inst.sector_cap, 1.0, l1 * delta, l2 * delta)
    return Qubo(Q, b, c, delta, n)
