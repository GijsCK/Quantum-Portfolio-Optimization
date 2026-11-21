"""Independent small-circuit experiments for the portfolio challenge.

Run from the project root with `python quantum_audit.py`. The address experiment
explicitly enumerates every K-asset set. Its gate count and lookup cost belong to
this small reference experiment, not to the scalable solver.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import ceil, log2
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from portfolio_generator import generate


def portfolio_dictionary(inst):
    """Use the organizer's scalar evaluator and original floating-point rules."""
    holdings = np.array(list(combinations(range(inst.n), inst.K)), dtype=int)
    scores, valid, penalties = [], [], []
    variances = np.einsum("ij,ij->i", inst.B, inst.B) + inst.d
    # A convex quadratic over normalized nonnegative weights is at most the
    # largest single-asset variance. Returns lie between the extreme returns.
    width = 0.5 * (variances.max() + np.ptp(inst.mu))
    barrier = width + max(width * 0.01, 1e-9)
    for assets in holdings:
        x = np.zeros(inst.n, dtype=int)
        x[assets] = 1
        scores.append(inst.objective(x))
        feasible = inst.is_feasible(x)
        valid.append(feasible)
        sector_counts = np.bincount(inst.sector[assets], minlength=inst.n_sectors)
        sector_excess = np.maximum(sector_counts - inst.sector_cap, 0).sum()
        budget_excess = max(0.0, inst.price[assets].sum() - inst.budget)
        excess = sector_excess / inst.K + budget_excess / inst.budget
        penalties.append(0.0 if feasible else barrier * (1.0 + excess))
    return holdings, np.array(scores), np.array(valid), np.array(penalties)


def amplitudes(angles, normalized_energy):
    """Apply cost phases, then rotate the amplitude along the uniform state."""
    state = np.full(len(normalized_energy), 1 / np.sqrt(len(normalized_energy)), complex)
    for gamma, beta in np.reshape(angles, (-1, 2)):
        state *= np.exp(-1j * gamma * normalized_energy)
        # The projector |s><s| replaces every component by their mean.
        state += np.expm1(-1j * beta) * state.mean()
    return state


def train(energy):
    """Optimize the exact expected penalized energy with a fixed search budget."""
    span = np.ptp(energy)
    normalized = (energy - energy.min()) / span if span > 0 else energy * 0
    bound = [(-6 * np.pi, 6 * np.pi), (-np.pi, np.pi)] * 2
    # Identical angles and caps for all encodings. Include uniform sampling as
    # the zero-angle variational candidate. Keep the best point ever evaluated.
    starts = qmc.Sobol(d=4, scramble=True, seed=1701).random_base2(7)
    starts = qmc.scale(starts, *np.asarray(bound).T)
    starts[0] = 0
    best_value, best_angles, evaluations = np.inf, None, 0

    def expectation(angles):
        nonlocal best_value, best_angles, evaluations
        evaluations += 1
        probability = np.abs(amplitudes(angles, normalized)) ** 2
        value = float(probability @ normalized)
        if value < best_value:
            best_value, best_angles = value, np.array(angles, copy=True)
        return value

    values = [expectation(point) for point in starts]
    for i in np.argsort(values)[:6]:
        minimize(expectation, starts[i], method="Powell", bounds=bound,
                 options={"maxfev": 250, "xtol": 1e-6, "ftol": 1e-8})
    return normalized, best_angles, evaluations


def qiskit_grover(normalized, angles):
    """Build the physical-order circuit, with zero initial weight on padding."""
    count = len(normalized)
    qubits = max(1, ceil(log2(count)))
    dimension = 1 << qubits
    uniform = np.zeros(dimension)
    uniform[:count] = 1 / np.sqrt(count)
    prep = StatePreparation(uniform)
    qc = QuantumCircuit(qubits)
    wires = list(range(qubits))
    qc.append(prep, wires)
    for gamma, beta in np.reshape(angles, (-1, 2)):
        phases = np.ones(dimension, complex)
        phases[:count] = np.exp(-1j * gamma * normalized)
        qc.append(DiagonalGate(phases), wires)
        # U (I + (exp(-i beta)-1)|0><0|) U† implements the mixer, with
        # U|0> = |s>. Qiskit appends gates in their chronological order.
        qc.append(prep.inverse(), wires)
        phases = np.ones(dimension, complex)
        phases[0] = np.exp(-1j * beta)
        qc.append(DiagonalGate(phases), wires)
        qc.append(prep, wires)
    return qc


def quality(probability, scores, feasible, rng, shots=256):
    opt = float(scores[feasible].min())
    is_opt = feasible & np.isclose(scores, opt, rtol=0, atol=1e-12)
    probability = probability / probability.sum()
    mass = float(probability[feasible].sum())
    p_opt = float(np.clip(probability[is_opt].sum(), 0, 1))
    counts = rng.multinomial(shots, probability)
    observed = (counts > 0) & feasible
    return {
        "feasible_mass": mass,
        "optimal_mass": p_opt,
        "optimum_hit_16_shots": float(1 - (1 - p_opt) ** 16),
        "conditional_mean_regret_bp": float(
            1e4 * ((probability[feasible] / mass) @ (scores[feasible] - opt))),
        "feasible_shots": int(counts[feasible].sum()),
        "optimal_shots": int(counts[is_opt].sum()),
        "best_sampled_objective": float(scores[observed].min()) if observed.any() else None,
        "address_counts": counts.tolist(),
    }


def address_case(n, seed):
    inst = generate(n, seed)
    tick = perf_counter()
    holdings, scores, valid, penalty = portfolio_dictionary(inst)
    prep_time = perf_counter() - tick
    if not valid.any():
        raise ValueError(f"The instance {(n, seed)} has no feasible reference.")
    if (~valid).any() and (scores + penalty)[~valid].min() <= scores[valid].max():
        raise AssertionError("The invalid subspace overlaps the feasible energy band.")
    record = {
        "n": n, "seed": seed, "k": inst.K, "fixed_k_sets": len(holdings),
        "feasible_sets": int(valid.sum()), "dictionary_seconds": prep_time,
        "exact_feasible_objective": float(scores[valid].min()),
        "cases": {},
    }
    for case, mask in (("hard", valid), ("penalty", np.ones(len(valid), bool))):
        # The hard dictionary orders only valid combinations. The penalty
        # dictionary uses all combinations. Padding never duplicates a portfolio.
        energy = (scores + penalty)[mask]
        tick = perf_counter()
        scaled, angles, calls = train(energy)
        state = amplitudes(angles, scaled)
        training_time = perf_counter() - tick
        p = abs(state) ** 2
        tick = perf_counter()
        qc = qiskit_grover(scaled, angles)
        actual = Statevector.from_instruction(qc).data
        compiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                             optimization_level=1, seed_transpiler=1701,
                             num_processes=1)
        expected = np.pad(state, (0, len(actual) - len(state)))
        difference = np.max(abs(abs(actual) ** 2 - abs(expected) ** 2))
        fidelity = min(1.0, float(abs(np.vdot(expected, actual)) ** 2))
        if difference > 1e-9 or 1 - fidelity > 1e-9:
            raise AssertionError("Qiskit and the analytic Grover evolution disagree.")
        # Each control gets its own reproducible shot draws. Exact probabilities
        # above do not have Monte Carlo uncertainty.
        rows = []
        for control, prob in enumerate((np.ones(len(p)) / len(p), p)):
            rng = np.random.default_rng([95241, n, seed, int(case == "penalty"), control])
            rows.append(quality(prob, scores[mask], valid[mask], rng))
        record["cases"][case] = {
            "addresses": len(energy), "qubits": qc.num_qubits,
            "padding": len(actual) - len(energy), "evaluations": calls,
            "angles_gamma_beta": angles.tolist(), "training_seconds": training_time,
            "qiskit_audit_seconds": perf_counter() - tick,
            "cx_all_to_all": int(compiled.count_ops().get("cx", 0)),
            "compiled_depth_all_to_all": compiled.depth(),
            "state_fidelity": fidelity, "max_probability_error": float(difference),
            "padding_probability": float((abs(actual[len(energy):]) ** 2).sum()),
            "uniform": rows[0], "trained": rows[1],
        }
    return record


def _branch_angle(theta, table, asset, remaining):
    """A conditional rotation with sin²(angle/2) equal to the take probability."""
    available = len(theta) - asset
    if remaining <= 0 or remaining > available:
        return 0.0
    if remaining == available:
        return np.pi
    # Log odds compare the summed weights of both futures. Evaluating both
    # sigmoid signs avoids subtracting a probability very close to one.
    log_odds = theta[asset] + table[asset + 1, remaining - 1] - table[asset + 1, remaining]
    return float(2 * np.arctan2(np.sqrt(expit(log_odds)), np.sqrt(expit(-log_odds))))


def weighted_dicke_audit(theta, k, sample_seed):
    """Prepare |S> with conditional Ry rotations controlled by the prefix bits.

    The direct enumeration oracle normalizes all portfolio weights independently.
    The circuit instead uses the scalable solver's suffix table to choose local
    conditional rotations. This includes the endian convention of the asset bits.
    """
    from variational import conditional_table, sample_fixed

    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    table = conditional_table(theta, k)
    dimension = 1 << n
    addresses = np.arange(dimension)
    binary = (addresses[:, None] >> np.arange(n)) & 1
    valid = binary.sum(axis=1) == k
    raw = binary[valid] @ theta
    expected = np.zeros(dimension)
    expected[valid] = np.exp(raw - logsumexp(raw))
    qc = QuantumCircuit(n)
    for i in range(n):
        # Entry j is the rotation angle given low-order prefix j. Prefixes that
        # cannot arise from a K-asset set can receive any angle; choose zero.
        angles = [_branch_angle(theta, table, i, k - prefix.bit_count())
                  for prefix in range(1 << i)]
        if i == 0:
            qc.ry(angles[0], i)
        else:
            qc.append(UCRYGate(angles), [i, *range(i)])
    actual = Statevector.from_instruction(qc).data
    max_error = np.max(abs(abs(actual) ** 2 - expected))
    fidelity = min(1.0, float(abs(np.vdot(np.sqrt(expected), actual)) ** 2))
    if max_error > 1e-9 or 1 - fidelity > 1e-9:
        raise AssertionError("The prepared state disagrees with enumeration.")
    draws = 32768
    raw_samples = sample_fixed(theta, k, draws, np.random.default_rng(sample_seed))
    if raw_samples.shape != (draws, n) or np.any(raw_samples.sum(axis=1) != k):
        raise AssertionError("The primary sampler returned a malformed portfolio.")
    sampled = raw_samples @ (1 << np.arange(n))
    frequencies = np.bincount(sampled, minlength=dimension) / draws
    variation = np.sum(abs(frequencies - expected)) / 2
    # E[TV] <= 1/2 sqrt((d-1)/N) for d outcomes. McDiarmid's inequality
    # bounds fluctuations because changing one of N shots changes TV by 1/N.
    tv_limit = 0.5 * np.sqrt((valid.sum() - 1) / draws) + np.sqrt(np.log(1e6) / (2 * draws))
    if variation > tv_limit:
        raise AssertionError("Monte Carlo sampling fails the finite-shot TV bound.")
    return {
        "n": n, "k": k, "theta": theta.tolist(), "state_fidelity": fidelity,
        "max_probability_error": float(max_error),
        "off_cardinality_mass": float((abs(actual[~valid]) ** 2).sum()),
        "shots": draws, "sample_seed": int(sample_seed), "sample_total_variation": float(variation),
        "tv_bound_at_failure_probability_1e_minus_6": float(tv_limit),
    }


def counter_circuit(theta, k):
    """Coherently prepare the weighted state using a reversible holding counter.

    Data bits occupy wires 0..n-1. The little-endian counter occupies the top
    wires, starts at K, and records how many assets remain to be selected. Each
    selected asset subtracts one. At the end all branches share counter zero.
    """
    from variational import conditional_table

    theta = np.asarray(theta, dtype=float)
    table = conditional_table(theta, k)
    n = len(theta)
    width = int(k).bit_length()
    counter = list(range(n, n + width))
    circuit = QuantumCircuit(n + width)
    if k == 0:
        return circuit
    for bit, wire in enumerate(counter):
        if k & (1 << bit):
            circuit.x(wire)
    for i in range(n):
        angles = [_branch_angle(theta, table, i, left) if left <= k else 0.0
                  for left in range(1 << width)]
        circuit.append(UCRYGate(angles), [i, *counter])
        # Decrement from low to high. A borrow continues precisely when all
        # already-updated lower bits are one. The data qubit enables each gate.
        circuit.cx(i, counter[0])
        for bit in range(1, width):
            circuit.mcx([i, *counter[:bit]], counter[bit])
    return circuit


def counter_audit(theta, k):
    n = len(theta)
    if n + int(k).bit_length() > 16:
        raise ValueError("Full-state validation is restricted to small reference circuits.")
    qc = counter_circuit(theta, k)
    actual = Statevector.from_instruction(qc).data.reshape((-1, 1 << n))
    # Row zero means the whole final counter is zero. Each column corresponds
    # to the asset bitmask, so this test also checks disentanglement of ancillas.
    binary = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    valid = binary.sum(axis=1) == k
    raw = binary[valid] @ theta
    oracle = np.zeros(1 << n)
    oracle[valid] = np.exp((raw - logsumexp(raw)) / 2)
    error = float(np.max(abs(abs(actual[0]) ** 2 - oracle ** 2)))
    leakage = float(np.sum(abs(actual[1:]) ** 2))
    fidelity = min(1.0, float(abs(np.vdot(oracle, actual[0])) ** 2))
    if error > 1e-9 or leakage > 1e-9 or 1 - fidelity > 1e-9:
        raise AssertionError("The coherent counter preparation failed.")
    compiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                         optimization_level=1, seed_transpiler=1701, num_processes=1)
    return {
        "n": n, "k": k, "data_qubits": n, "counter_qubits": qc.num_qubits - n,
        "total_qubits": qc.num_qubits, "theta": np.asarray(theta).tolist(),
        "max_probability_error": error, "nonzero_counter_mass": leakage,
        "joint_state_fidelity": fidelity,
        "all_to_all_counts": {str(k): int(v) for k, v in compiled.count_ops().items()},
        "all_to_all_depth": compiled.depth(),
    }


def write_json(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Path to results/variational.json from the main sweep")
    parser.add_argument("--counter", action="store_true",
                        help="Check the coherent counter circuit on a generic n=10, K=2 case")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.counter:
        theta = np.random.default_rng(10571).normal(size=10)
        record = counter_audit(theta, 2)
        record.update(
            parameter_seed=10571,
            basis="all-to-all rz, sx, x, cx; optimization_level=1; seed_transpiler=1701",
            scalable_logical_operations=(
                "Per asset: one Ry multiplexed by b counter bits (2**b angles), "
                "then b controlled-X gates with 1 through b controls. "
                "b=ceil(log2(K+1)). The final counter is zero."
            ),
        )
        write_json(args.output_dir / "counter_circuit.json", record)
        return
    if args.weights:
        with args.weights.open(encoding="utf-8") as f:
            primary = json.load(f)
        rows = [{"seed": r["seed"], "theta": r["trained"]["final_theta"]}
                for r in primary["instances"] if r["n"] == 10]
        seeds = [row["seed"] for row in rows]
        if sorted(seeds) != list(range(5)) or any(len(row["theta"]) != 10 for row in rows):
            raise ValueError("Expected one 10-asset vector for each official seed.")
        audit = []
        for row in rows:
            seed = row["seed"]
            audit.append({"seed": seed, **weighted_dicke_audit(
                row["theta"], generate(10, seed).K, 7100 + seed)})
        write_json(args.output_dir / "weighted_circuit_audit.json", audit)
        return
    with threadpool_limits(limits=1):
        rows = [address_case(n, seed) for n in (10, 20) for seed in range(5)]
    write_json(args.output_dir / "quantum_audit.json", {
        "protocol": {
            "p": 2, "shots_per_distribution": 256, "sobol_seed": 1701,
            "sobol_starts": 128, "powell_refinements": 6, "max_calls_per_refinement": 250,
            "cost_angle_bounds": [-6 * np.pi, 6 * np.pi], "mixer_angle_bounds": [-np.pi, np.pi],
            "angles_in_radians": True, "objective": "mean energy, min-max normalized independently per dictionary",
            "address_order": "lexicographic K-subsets; hard encoding retains feasible sets in order",
            "resource_basis": "all-to-all rz, sx, x, cx, Qiskit transpile optimization_level=1",
            "grover_mixer_reference": "https://arxiv.org/abs/2006.00354",
            "statevector_reference": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector"
        },
        "instances": rows,
    })


if __name__ == "__main__":
    main()
