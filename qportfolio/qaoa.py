"""QAOA on the hand-built Ising Hamiltonian, one qubit per asset (n <= ~22).

Circuit, repeated p times after the initial state:
    cost layer   exp(-i gamma H_C)  = RZ(2 gamma h_i) on every qubit, RZZ(2 gamma J_ij) on every pair
    mixer layer  "x":  exp(-i beta sum X_i)                    = RX(2 beta) on every qubit
                 "xy": exp(-i beta/2 sum_ring (X_i X_j + Y_i Y_j)) = RXX(beta) RYY(beta) on ring pairs
                 "xy-full": the same on all pairs (complete graph)

The X mixer starts in |+>^n and explores all 2^n bitstrings, so the cardinality rule needs a
penalty. The XY mixers only swap a |1> with a |0>; starting from the Dicke state (uniform over
weight K) they never leave the C(n, K) portfolios with exactly K assets, so cardinality is
enforced by the circuit itself, with no penalty and no extra qubits.

Training uses exact expectations from a noise-free statevector (Aer). Because H_C is diagonal,
<H_C> = sum_k |psi_k|^2 E_k with E precomputed for all 2^n bitstrings. Readout draws a finite
number of shots from the final state, as hardware would.
"""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .classical import local_search, repair, scaled_objective
from .problem import build_qubo, feasible_mask


def _scs(qc, q, k):
    """Split-and-cyclic-shift block of Bartschi & Eidenbenz (2019)."""
    m = len(q)
    qc.cx(q[m - 2], q[m - 1])
    qc.cry(2 * np.arccos(np.sqrt(1 / m)), q[m - 1], q[m - 2])
    qc.cx(q[m - 2], q[m - 1])
    for l in range(2, k + 1):
        qc.cx(q[m - 1 - l], q[m - 1])
        qc.append(RYGate(2 * np.arccos(np.sqrt(l / m))).control(2), [q[m - 1], q[m - l], q[m - 1 - l]])
        qc.cx(q[m - 1 - l], q[m - 1])


def dicke_state(n, k):
    """|D^n_k>: equal superposition of all C(n, k) bitstrings with k ones, O(n k) gates.

    This is the XY-mixer analogue of |+>^n: an unbiased start over every valid portfolio.
    """
    qc = QuantumCircuit(n)
    q = list(range(n))
    qc.x(q[n - k:])
    for l in range(n, k, -1):
        _scs(qc, q[:l], k)
    for l in range(k, 1, -1):
        _scs(qc, q[:l], l - 1)
    return qc


def _xy_pairs(n, mixer):
    if mixer == "xy":                       # ring: n pairs per layer
        return [(i, (i + 1) % n) for i in range(n)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]   # "xy-full": all pairs


def build_circuit(op, n, reps, mixer, K=None):
    gammas, betas = ParameterVector("g", reps), ParameterVector("b", reps)
    qc = QuantumCircuit(n)
    if mixer == "x":
        qc.h(range(n))
    else:
        qc.compose(dicke_state(n, K), inplace=True)
    z = [(p.to_label()[::-1], c.real) for p, c in zip(op.paulis, op.coeffs)]
    for layer in range(reps):
        g, b = gammas[layer], betas[layer]
        for label, c in z:
            qs = [i for i, ch in enumerate(label) if ch == "Z"]
            if len(qs) == 1:
                qc.rz(2 * c * g, qs[0])
            elif len(qs) == 2:
                qc.rzz(2 * c * g, *qs)
        if mixer == "x":
            qc.rx(2 * b, range(n))
        else:
            for i, j in _xy_pairs(n, mixer):
                qc.rxx(b, i, j)
                qc.ryy(b, i, j)
    return qc, gammas, betas


class _Sim:
    """Exact probabilities |psi(gamma, beta)|^2 from Aer, transpiled once."""

    def __init__(self, qc, gammas, betas):
        qc = qc.copy()
        qc.save_statevector()
        self.sim = AerSimulator(method="statevector")
        self.tqc = transpile(qc, self.sim, optimization_level=1)
        self.gammas, self.betas = gammas, betas
        self.calls = 0

    def probs(self, params):
        p = len(self.gammas)
        binds = {g: [v] for g, v in zip(self.gammas, params[:p])}
        binds |= {b: [v] for b, v in zip(self.betas, params[p:])}
        self.calls += 1
        sv = self.sim.run(self.tqc, parameter_binds=[binds]).result().get_statevector()
        return np.abs(np.asarray(sv)) ** 2


def _interp(v):
    """INTERP (Zhou et al. 2020): depth-p angles -> good starting depth-(p+1) angles."""
    p = len(v)
    ext = np.r_[0.0, v, 0.0]
    return np.array([(i / p) * ext[i] + ((p - i) / p) * ext[i + 1] for i in range(p + 1)])


def train(qubo, inst, reps=3, mixer="xy", maxiter=300, seed=0):
    """Layer-by-layer QAOA optimisation. Returns final probabilities and bookkeeping."""
    n = qubo.n
    E = qubo.all_energies()
    space = np.ones(1 << n, bool)
    if mixer != "x":                                    # XY mixers never leave weight K
        space = np.bitwise_count(np.arange(1 << n)) == inst.K
    # Rescale so the energy spread over the reachable states is 1: then gamma ~ O(1) is a
    # meaningful step regardless of n or of how strong the penalties are.
    scale = float(E[space].std())
    op, _ = qubo.to_ising()
    op = op / scale
    En = (E - E[space].mean()) / scale

    rng = np.random.default_rng(seed)
    params = None
    sims, evals = [], 0
    for p in range(1, reps + 1):
        qc, g, b = build_circuit(op, n, p, mixer, inst.K)
        sim = _Sim(qc, g, b)
        cost = lambda th: float(sim.probs(th) @ En)
        if params is None:      # depth 1: coarse grid, then refine
            grid = [(gg, bb) for gg in np.linspace(0.05, 2.0, 12) for bb in np.linspace(0.05, 1.5, 8)]
            vals = [cost(np.array(t)) for t in grid]
            start = np.array(grid[int(np.argmin(vals))])
        else:
            start = np.r_[_interp(params[:p - 1]), _interp(params[p - 1:])]
            start = start + rng.normal(0, 0.01, start.size)
        res = minimize(cost, start, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.1})
        params = res.x
        evals += sim.calls
        sims.append(sim)
    probs = sims[-1].probs(params)
    return {"probs": probs, "params": params, "evaluations": evals + 1,
            "cost": float(probs @ En), "depth": sims[-1].tqc.depth(),
            "two_qubit_gates": sum(v for k, v in sims[-1].tqc.count_ops().items() if k in ("rzz", "rxx", "ryy", "cx"))}


def readout(inst, qubo, probs, shots=1024, seed=0, polish=False):
    """Sample bitstrings, keep the best feasible one (post-selection).

    polish=False: if no sample is feasible, repair the lowest-energy sample (classical step).
    polish=True: additionally run repair + local search from the 10 best distinct samples.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(probs.size, size=shots, p=probs / probs.sum())
    X = qubo.portfolio(idx)
    feas = feasible_mask(inst, X)
    uniq = np.unique(idx)
    Xu = qubo.portfolio(uniq)
    Eu = qubo.energy((uniq[:, None] >> np.arange(qubo.n)) & 1)   # includes slack bits, if any
    fu = feasible_mask(inst, Xu)
    obj = np.array([scaled_objective(inst, x) for x in Xu])
    if fu.any():
        best = Xu[np.flatnonzero(fu)[np.argmin(obj[fu])]]
    else:
        best = repair(inst, Xu[np.argmin(Eu)])
    if polish:
        for k in np.argsort(Eu)[:10]:
            x = local_search(inst, repair(inst, Xu[k]))
            if scaled_objective(inst, x) < scaled_objective(inst, best):
                best = x
    return best, {"shots": shots, "raw_feasible": float(feas.mean()), "distinct": int(uniq.size)}


def solver(mixer="xy", reps=3, shots=1024, budget=("unbalanced", 0.1, 0.03), polish=False, seed=0):
    """Harness-compatible solver: inst -> (x, info)."""
    def solve(inst):
        qubo = build_qubo(inst, budget=budget, sector=None, cardinality=(mixer == "x"))
        out = train(qubo, inst, reps=reps, mixer=mixer, seed=seed)
        x, info = readout(inst, qubo, out["probs"], shots, seed, polish)
        info |= {"qubits": qubo.n, "evaluations": out["evaluations"],
                 "depth": out["depth"], "two_qubit_gates": out["two_qubit_gates"]}
        return x, info
    return solve
