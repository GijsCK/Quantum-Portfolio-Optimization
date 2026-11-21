"""Unscored bonus study: how finite shots and gate noise affect a trained PCE circuit.

Train once noise-free (as the rules require), then read the trained circuit out
  (a) with a finite number of shots per measurement basis (X, Y, Z), and
  (b) under depolarizing gate noise (Aer density-matrix simulator), exact expectations.
Decoding is top-K by correlator value + objective-blind repair, i.e. the "raw" score.

    python -m experiments.pce_noise [--n 100] [--seed 0]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from portfolio_generator import generate
from qportfolio.classical import miqp
from qportfolio.metrics import normalised_score, random_feasible_mean
from qportfolio.pce import BASES, decode, fwht, qubits_needed, train_pce


def sampled_correlators(ans, enc, psi, shots, rng):
    c = np.empty(enc.n)
    for b in BASES:
        p = np.abs(ans.rotate(psi, b)) ** 2
        counts = rng.multinomial(shots, p / p.sum())
        c[enc.items[b]] = fwht(counts / shots)[enc.masks[b]]
    return c


def noisy_correlators(ans, enc, theta, p2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["ry", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    qc = ans.to_qiskit(theta)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    # The circuit is already in the ry/rz/cz basis the noise model targets: no transpile.
    rho = DensityMatrix(sim.run(qc).result().data()["density_matrix"])
    ops = [SparsePauliOp.from_sparse_list([(b * enc.k, list(s), 1.0)], num_qubits=enc.N)
           for b, s in enc.assign]
    return np.array([rho.expectation_value(o).real for o in ops])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/pce_noise.csv")
    a = ap.parse_args()

    inst = generate(a.n, a.seed)
    x_opt, _ = miqp(inst)
    f_opt, f_rand = inst.objective(x_opt), random_feasible_mean(inst)
    k = 2 if a.n <= 100 else 3
    N = qubits_needed(a.n, k)
    layers = int(min(32, np.ceil(3 * a.n / (2 * N))))
    runs, info = train_pce(inst, k=k, layers=layers, alpha=5.0 * N, restarts=2,
                           maxiter=1500, seed=a.seed)
    best = int(np.argmax([normalised_score(inst.objective(decode(inst, c, polish=False)),
                                           f_opt, f_rand) for c in runs]))
    ans, enc, theta = info["ansatz"], info["encoding"], info["thetas"][best]
    psi = ans.state(theta)
    c_exact = enc.correlators(ans, psi)

    def score(c):
        return normalised_score(inst.objective(decode(inst, c, polish=False)), f_opt, f_rand)

    def overlap(c):   # fraction of the noise-free top-K kept
        top = lambda v: set(np.argsort(-v)[:inst.K])
        return len(top(c) & top(c_exact)) / inst.K

    rows = [{"kind": "exact", "level": 0, "score": score(c_exact), "topk_overlap": 1.0,
             "mean_abs_corr": float(np.abs(c_exact).mean()), "reps": 1}]
    rng = np.random.default_rng(0)
    for shots in (100, 300, 1000, 3000, 10_000, 30_000, 100_000):
        sc, ov = [], []
        for _ in range(20):
            c = sampled_correlators(ans, enc, psi, shots, rng)
            sc.append(score(c))
            ov.append(overlap(c))
        rows.append({"kind": "shots_per_basis", "level": shots, "score": float(np.mean(sc)),
                     "score_sd": float(np.std(sc)), "topk_overlap": float(np.mean(ov)),
                     "mean_abs_corr": float(np.abs(c_exact).mean()), "reps": 20})
    for p2 in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        c = noisy_correlators(ans, enc, theta, p2)
        rows.append({"kind": "depolarizing_cz", "level": p2, "score": score(c),
                     "topk_overlap": overlap(c), "mean_abs_corr": float(np.abs(c).mean()),
                     "shrink": float(np.abs(c).mean() / np.abs(c_exact).mean()), "reps": 1})
    for r in rows:
        print(r, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"n={a.n} N={N} layers={layers} params={info['params']} "
          f"CZ gates={ans.to_qiskit(theta).count_ops().get('cz', 0)}")


if __name__ == "__main__":
    main()
