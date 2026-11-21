"""Pauli correlation encoding (PCE), Sciorilli et al., Nat. Commun. 16:476 (2025).

n binary decisions live on N << n qubits:  x_i = [<Pi_i> > 0],  where Pi_i are k-body Pauli
strings X^{(x)k}, Y^{(x)k} or Z^{(x)k} on a k-subset of qubits. 3 C(N, k) such strings exist, so
n = 1000 fits on N = 14 qubits with k = 3. Each family commutes, so three measurement settings
(all-X, all-Y, all-Z) give every correlator.

Training minimises a relaxed loss of t_i = tanh(alpha <Pi_i>), x~_i = (1 + t_i)/2 (see
``relaxed_loss``). Unlike a QAOA Hamiltonian, this loss is computed classically from measured
correlators, so it need not be quadratic: budget and sector caps become exact hinge penalties
that vanish on every feasible portfolio, with no slack qubits.

Simulation is an exact (noise-free) statevector in NumPy:
- correlators of a basis = Walsh-Hadamard transform of that basis' outcome probabilities;
- gradients by the adjoint method: d<O>/dtheta for O = sum_i (dL/d<Pi_i>) Pi_i, which equals
  dL/dtheta by the chain rule. Cost ~3 forward passes regardless of the parameter count.
Both are cross-checked against Qiskit's Statevector in tests/test_pce.py.
"""
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import minimize

from .classical import local_search, repair, scaled_objective

BASES = ("Z", "X", "Y")


# ------------------------------------------------------------------ encoding

def qubits_needed(n, k):
    N = k
    while 3 * comb(N, k) < n:
        N += 1
    return N


def layout(n, N, k):
    """Assign asset i -> (basis, qubit subset). Round-robin over bases keeps them balanced."""
    out = []
    for subset in combinations(range(N), k):
        for basis in BASES:
            out.append((basis, subset))
            if len(out) == n:
                return out
    raise ValueError("not enough correlators")


def fwht(v):
    """Unnormalised Walsh-Hadamard transform: out[m] = sum_b v[b] (-1)^popcount(b & m)."""
    v = np.array(v, copy=True)
    n = v.size
    h = 1
    while h < n:
        v = v.reshape(-1, 2, h)
        a, b = v[:, 0, :].copy(), v[:, 1, :]
        v[:, 0, :] += b
        v[:, 1, :] = a - b
        v = v.reshape(n)
        h *= 2
    return v


# ------------------------------------------------------------------ ansatz + simulator

class Ansatz:
    """Brickwork: [RY RZ on every qubit] then L x ([CZ brick layer] [RY RZ on every qubit]).

    entangle=False drops the CZ layers: the same parameters on a product state, the control
    for "does entanglement matter?".
    """

    def __init__(self, N, layers, entangle=True):
        self.N, self.L, self.entangle = N, layers, entangle
        self.n_params = 2 * N * (layers + 1)
        idx = np.arange(1 << N)
        bits = (idx[:, None] >> np.arange(N)) & 1
        self.cz = []
        for l in range(layers):
            sign = np.ones(1 << N)
            for a in range(l % 2, N - 1, 2):
                sign *= 1 - 2 * (bits[:, a] & bits[:, a + 1])
            self.cz.append(sign)
        self.phase_y = (-1j) ** np.bitwise_count(idx)       # S^dagger on every qubit

    def gates(self):
        p = 0
        for block in range(self.L + 1):
            if block > 0 and self.entangle:
                yield ("cz", block - 1, None)
            for q in range(self.N):
                yield ("ry", q, p)
                yield ("rz", q, p + 1)
                p += 2

    # single-qubit helpers on a flat state, qubit q = bit q of the index (Qiskit order)
    def _view(self, psi, q):
        return psi.reshape(1 << (self.N - 1 - q), 2, 1 << q)

    def _apply(self, psi, kind, q, theta, inverse=False):
        if kind == "cz":
            return psi * self.cz[q]
        th = -theta if inverse else theta
        v = self._view(psi, q)
        out = np.empty_like(v)
        if kind == "ry":
            c, s = np.cos(th / 2), np.sin(th / 2)
            out[:, 0, :] = c * v[:, 0, :] - s * v[:, 1, :]
            out[:, 1, :] = s * v[:, 0, :] + c * v[:, 1, :]
        else:  # rz
            out[:, 0, :] = np.exp(-0.5j * th) * v[:, 0, :]
            out[:, 1, :] = np.exp(0.5j * th) * v[:, 1, :]
        return out.reshape(-1)

    def _generator(self, psi, kind, q):
        v = self._view(psi, q)
        out = np.empty_like(v)
        if kind == "ry":
            out[:, 0, :], out[:, 1, :] = -1j * v[:, 1, :], 1j * v[:, 0, :]
        else:
            out[:, 0, :], out[:, 1, :] = v[:, 0, :], -v[:, 1, :]
        return out.reshape(-1)

    def to_qiskit(self, theta):
        """The same circuit as a Qiskit QuantumCircuit (for gate counts and Aer simulation)."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.N)
        for kind, q, p in self.gates():
            if kind == "cz":
                for a in range(q % 2, self.N - 1, 2):
                    qc.cz(a, a + 1)
            elif kind == "ry":
                qc.ry(theta[p], q)
            else:
                qc.rz(theta[p], q)
        return qc

    def state(self, theta):
        psi = np.zeros(1 << self.N, complex)
        psi[0] = 1.0
        for kind, q, p in self.gates():
            psi = self._apply(psi, kind, q, None if p is None else theta[p])
        return psi

    def rotate(self, psi, basis):
        """State in the measurement frame of ``basis`` (R psi, with R Pauli R^dag = Z)."""
        scale = 2.0 ** (-self.N / 2)
        if basis == "Z":
            return psi
        if basis == "X":
            return fwht(psi) * scale
        return fwht(self.phase_y * psi) * scale

    def unrotate(self, phi, basis):
        scale = 2.0 ** (-self.N / 2)
        if basis == "Z":
            return phi
        if basis == "X":
            return fwht(phi) * scale
        return np.conj(self.phase_y) * fwht(phi) * scale

    def gradient(self, theta, psi, O_psi):
        """d<psi|O|psi>/dtheta by the adjoint method, given psi = U(theta)|0> and O|psi>."""
        gates = list(self.gates())
        grad = np.zeros(self.n_params)
        lam = O_psi
        for kind, q, p in reversed(gates):
            if p is not None:
                grad[p] = np.imag(np.vdot(lam, self._generator(psi, kind, q)))
            th = None if p is None else theta[p]
            psi = self._apply(psi, kind, q, th, inverse=True)
            lam = self._apply(lam, kind, q, th, inverse=True)
        return grad


class Encoding:
    """Correlator <Pi_i> for every asset, and the observable sum_i g_i Pi_i applied to psi."""

    def __init__(self, n, N, k):
        self.n, self.N, self.k = n, N, k
        self.assign = layout(n, N, k)
        self.masks = {b: [] for b in BASES}
        self.items = {b: [] for b in BASES}
        for i, (basis, subset) in enumerate(self.assign):
            self.masks[basis].append(sum(1 << q for q in subset))
            self.items[basis].append(i)

    def correlators(self, ansatz, psi):
        c = np.empty(self.n)
        for b in BASES:
            if self.items[b]:
                p = np.abs(ansatz.rotate(psi, b)) ** 2
                c[self.items[b]] = fwht(p)[self.masks[b]]
        return c

    def apply_observable(self, ansatz, psi, g):
        out = np.zeros_like(psi)
        for b in BASES:
            if not self.items[b]:
                continue
            G = np.zeros(1 << self.N)
            G[self.masks[b]] = g[self.items[b]]
            D = fwht(G)                                   # diagonal of sum g_i Pi_i in frame b
            out += ansatz.unrotate(D * ansatz.rotate(psi, b), b)
        return out


# ------------------------------------------------------------------ loss

class RelaxedLoss:
    """L(x~) = F(x~)/scale + lam_card (sum x~ - K)^2 + lam_budget hinge(price) + lam_sector hinge.

    F is the scaled official objective ||B^T x||^2 + (d - c mu).x, exact on binary x. ``scale``
    is the spread of single-asset marginal costs times K, so one asset swap moves F/scale by
    O(1/K) at every size. Hinges are zero on feasible portfolios.
    """

    def __init__(self, inst, q=0.5, lam_card=1.0, lam_budget=1.0, lam_sector=1.0):
        self.inst = inst
        self.lin = inst.d - (1 - q) * inst.K / q * inst.mu
        y0 = inst.B.T @ np.full(inst.n, inst.K / inst.n)
        marg = 2 * inst.B @ y0 + (inst.B ** 2).sum(1) + self.lin
        self.scale = inst.K * marg.std()
        self.unit = inst.budget / inst.K
        self.onehot = np.eye(inst.n_sectors)[inst.sector]
        self.lam = (lam_card, lam_budget, lam_sector)

    def __call__(self, x):
        """Value and gradient with respect to x~."""
        inst, (lc, lb, ls) = self.inst, self.lam
        y = inst.B.T @ x
        val = (y @ y + self.lin @ x) / self.scale
        grad = (2 * inst.B @ y + self.lin) / self.scale
        e = x.sum() - inst.K
        val += lc * e * e
        grad += 2 * lc * e
        over = max(0.0, inst.price @ x - inst.budget) / self.unit
        val += lb * over * over
        grad += 2 * lb * over * inst.price / self.unit
        exc = np.maximum(x @ self.onehot - inst.sector_cap, 0)
        val += ls * exc @ exc
        grad += 2 * ls * self.onehot @ exc
        return val, grad


def decode(inst, score, q=0.5, polish=True):
    """Top-K assets by score -> repair budget/sector -> optional swap local search.

    Without polish the repair is objective-blind, so the result measures the upstream
    solver; with polish, repair and local search may use the objective freely.
    """
    x = np.zeros(inst.n, int)
    x[np.argsort(-score)[:inst.K]] = 1
    x = repair(inst, x, q, use_objective=polish)
    return local_search(inst, x, q) if polish else x


# ------------------------------------------------------------------ solvers

def _fit(fun, x0, maxiter):
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x, res.nfev


def train_pce(inst, k=None, layers=None, entangle=True, alpha=None, beta=0.0,
              restarts=4, maxiter=400, seed=0, q=0.5, loss=None):
    """Train PCE circuits from ``restarts`` random angles; return correlators of each run."""
    n = inst.n
    k = k or (2 if n <= 100 else 3)
    N = qubits_needed(n, k)
    enc = Encoding(n, N, k)
    layers = layers if layers is not None else max(2, int(np.ceil(n / (2 * N) / 2)))
    ans = Ansatz(N, layers, entangle)
    alpha = alpha or 1.5 * N
    loss = loss or RelaxedLoss(inst, q)

    def fun(theta):
        psi = ans.state(theta)
        c = enc.correlators(ans, psi)
        t = np.tanh(alpha * c)
        val, gx = loss((1 + t) / 2)
        reg = np.mean(t * t)
        val += beta * reg * reg
        dt = gx / 2 + beta * 2 * reg * 2 * t / n
        g = dt * alpha * (1 - t * t)                    # dL/dc_i
        return val, ans.gradient(theta, psi, enc.apply_observable(ans, psi, g))

    rng = np.random.default_rng(seed)
    runs, thetas, evals = [], [], 0
    for _ in range(restarts):
        theta, nfev = _fit(fun, rng.uniform(0, 2 * np.pi, ans.n_params), maxiter)
        evals += nfev
        thetas.append(theta)
        runs.append(enc.correlators(ans, ans.state(theta)))
    return runs, {"qubits": N, "k": k, "layers": layers, "params": ans.n_params,
                  "evaluations": evals, "circuits": 3 * evals, "thetas": thetas,
                  "ansatz": ans, "encoding": enc}


def train_relaxation(inst, restarts=4, maxiter=400, seed=0, q=0.5, loss=None):
    """Control: the same relaxed loss optimised directly over n classical variables."""
    loss = loss or RelaxedLoss(inst, q)

    def fun(w):
        t = np.tanh(w)
        val, gx = loss((1 + t) / 2)
        return val, gx / 2 * (1 - t * t)

    rng = np.random.default_rng(seed)
    runs, evals = [], 0
    for _ in range(restarts):
        w, nfev = _fit(fun, rng.normal(0, 0.1, inst.n), maxiter)
        evals += nfev
        runs.append(np.tanh(w))
    return runs, {"params": inst.n, "evaluations": evals}


def best_decoded(inst, runs, polish, q=0.5):
    """Decode every restart and keep the best portfolio (the same rule for every method)."""
    best, best_F = None, np.inf
    for score in runs:
        x = decode(inst, score, q, polish)
        F = scaled_objective(inst, x, q)
        if F < best_F:
            best, best_F = x, F
    return best
