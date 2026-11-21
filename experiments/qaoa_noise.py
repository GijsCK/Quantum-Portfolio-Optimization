"""Unscored bonus: the same depolarizing noise on a trained n=10 XY-QAOA circuit (Dicke start).

QAOA's answer *is* its output distribution, so noise that pushes the state toward the
maximally mixed state directly lowers the probability of the optimum and of feasible baskets.

    python -m experiments.qaoa_noise
"""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from portfolio_generator import generate
from qportfolio.classical import brute_force
from qportfolio.problem import build_qubo, feasible_mask
from qportfolio.qaoa import build_circuit, train

n, seed, reps = 10, 0, 3
inst = generate(n, seed)
x_opt, _ = brute_force(inst)
k_opt = int(x_opt @ (1 << np.arange(n)))
qubo = build_qubo(inst, sector=None, cardinality=False)
out = train(qubo, inst, reps=reps, mixer="xy", seed=seed)
E = qubo.all_energies()
space = np.bitwise_count(np.arange(1 << n)) == inst.K
op, _ = qubo.to_ising()
qc, g, b = build_circuit(op / float(E[space].std()), n, reps, "xy", inst.K)
qc = qc.assign_parameters(dict(zip(list(g) + list(b), out["params"])))
qc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
qc.save_probabilities()
feas = feasible_mask(inst, (np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
print(f"n={n} CX gates={qc.count_ops().get('cx', 0)}  uniform-over-K: P(opt)={1/space.sum():.4f}")
for p2 in (0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
    nm = NoiseModel()
    if p2:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    P = np.asarray(sim.run(qc).result().data()["probabilities"])
    print(f"p_cx={p2:<7} P(opt)={P[k_opt]:.4f}  P(feasible)={P[feas].sum():.3f}  P(K assets)={P[space].sum():.3f}")
