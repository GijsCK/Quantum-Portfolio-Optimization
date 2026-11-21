"""Ideal measurement of a number-projected variational quantum state.

Each data qubit names an asset. For a basis state x containing exactly K ones,
the probability is proportional to exp(theta @ x). No circuit amplitude from a
different cardinality can contribute. The sampler computes the Born law exactly
up to floating-point arithmetic, without allocating 2**n complex amplitudes.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


TEMPERATURE_INVERSE = (1.5, 2.5, 4., 7., 12., 20., 32., 50., 72.)
TRAIN_SHOTS = 128
FINAL_SHOTS = 256
MODES = ("trained", "schedule_only", "frozen")


def conditional_table(theta, k):
    """Entry [i,r] is log(total weight of r-element subsets of assets i:)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or not np.isfinite(theta).all() or not 0 <= k <= len(theta):
        raise ValueError("Need finite one-dimensional logits and 0 <= k <= n")
    n = len(theta)
    table = np.full((n + 1, k + 1), -np.inf)
    table[:, 0] = 0.
    for i in range(n - 1, -1, -1):
        stop = min(k, n - i)
        # The two alternatives are to skip i, or to take it and fill r-1 places.
        table[i, 1:stop + 1] = np.logaddexp(
            table[i + 1, 1:stop + 1], theta[i] + table[i + 1, :stop])
    return table


def _draw(theta, k, shots, rng, table):
    n = len(theta)
    samples = np.zeros((shots, n), dtype=bool)
    remaining = np.full(shots, k, dtype=int)
    for i in range(n):
        # exp(log numerator - log denominator) is the next Ry gate's sin²(angle/2).
        prev = np.maximum(remaining - 1, 0)
        log_p = theta[i] + table[i + 1, prev] - table[i, remaining]
        take = (remaining > 0) & (rng.random(shots) < np.exp(np.minimum(log_p, 0.)))
        # The deterministic final branches also guard against last-bit roundoff.
        take |= remaining == n - i
        samples[:, i] = take
        remaining -= take
    if remaining.any():
        raise ArithmeticError("Conditional measurement lost cardinality")
    return samples


def sample_fixed(theta, k, shots, rng):
    """Return directly measured portfolios as a (shots, assets) boolean array."""
    theta = np.asarray(theta, dtype=float)
    return _draw(theta, k, shots, rng, conditional_table(theta, k))


def measure_energy(inst, samples):
    """Official objective and constrained diagonal Hamiltonian on fixed-K states."""
    bits = np.asarray(samples, dtype=float)
    if not np.all((bits == 0) | (bits == 1)) or not np.all(bits.sum(axis=1) == inst.K):
        raise ValueError("Hamiltonian evaluation requires binary, fixed-K portfolios")
    exposure = bits @ inst.B
    risk = (np.einsum("ij,ij->i", exposure, exposure) + bits @ inst.d) / inst.K**2
    score = 0.5 * (risk - (bits @ inst.mu) / inst.K)
    spend = bits @ inst.price
    counts = np.column_stack([bits[:, inst.sector == s].sum(axis=1)
                              for s in range(inst.n_sectors)])
    extra_spend = np.maximum(spend - (inst.budget + 1e-9), 0.)
    extra_slots = np.maximum(counts - inst.sector_cap, 0.).sum(axis=1)
    valid = (extra_spend == 0.) & (extra_slots == 0.)
    worst_variance = np.max(np.einsum("ij,ij->i", inst.B, inst.B) + inst.d)
    score_span = 0.5 * (worst_variance + np.ptp(inst.mu)) + 1e-9
    energy = score + score_span * (~valid + extra_spend / inst.budget + extra_slots / inst.K)
    return score, energy, valid


def _asset_utility(inst, exposure):
    # Price the risk of adding an asset to the current average factor exposure.
    # The idiosyncratic variance is linear in a binary holding.
    return 0.5 * inst.mu - 0.5 * (2 * (inst.B @ exposure) + inst.d / inst.K)


def _center_fields(inst, preferences):
    """Fit continuous price/sector fields for the trial state without selecting holdings.

    Independent Bernoulli means are an inexpensive guide for the fields. The
    actual quantum state reconditions on K. Only its measured bits are scored.
    """
    price = inst.price / (inst.budget / inst.K)
    spend_target = 1. - min(.05, .50 / np.sqrt(inst.K))
    sector_target = (inst.sector_cap - min(.3, .04 * inst.sector_cap)) / inst.K

    def dual(fields):
        logits = preferences - fields[0] - fields[1] * price - fields[2:][inst.sector]
        prob = expit(logits)
        value = (np.logaddexp(0., logits).sum() / inst.K + fields[0]
                 + fields[1] * spend_target + fields[2:].sum() * sector_target)
        counts = np.bincount(inst.sector, weights=prob, minlength=inst.n_sectors)
        gradient = np.r_[1. - prob.sum() / inst.K,
                         spend_target - prob @ price / inst.K,
                         sector_target - counts / inst.K]
        return value, gradient

    start = np.zeros(2 + inst.n_sectors)
    start[0] = np.quantile(preferences, 1 - inst.K / inst.n)
    result = minimize(dual, start, jac=True, method="L-BFGS-B",
                      bounds=[(None, None)] + [(0., None)] * (len(start) - 1),
                      options={"maxiter": 160, "ftol": 1e-12, "gtol": 1e-7})
    if not np.all(np.isfinite(result.x)):
        raise ArithmeticError("Variational field optimization returned nonfinite values")
    fields = result.x
    theta = preferences - fields[0] - fields[1] * price - fields[2:][inst.sector]
    return theta, bool(result.success)


def solve(inst, mode="trained", replicate=0):
    """Return the lowest-energy feasible measurement and independent shot diagnostics."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    start = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([1841, inst.n, inst.seed, replicate]))
    exposure = inst.B.mean(axis=0)
    scale = max(_asset_utility(inst, exposure).std(), 1e-12)
    feedback = np.zeros(inst.n)
    best_score = np.inf
    best = None
    traces = []
    total_feasible = 0
    for step, concentration in enumerate(TEMPERATURE_INVERSE):
        shots = FINAL_SHOTS if step == len(TEMPERATURE_INVERSE) - 1 else TRAIN_SHOTS
        if step == 0 or mode != "frozen":
            utility = _asset_utility(inst, exposure)
            preferences = concentration * (utility - utility.mean()) / scale + feedback
            theta, converged = _center_fields(inst, preferences)
            table = conditional_table(theta, inst.K)
        samples = _draw(theta, inst.K, shots, rng, table)
        scores, energies, valid = measure_energy(inst, samples)
        ranked = np.argsort(energies, kind="stable")
        elite = samples[ranked[:shots // 8]]
        if valid.any():
            idx = int(np.argmin(np.where(valid, scores, np.inf)))
            if scores[idx] < best_score:
                best = np.flatnonzero(samples[idx]).tolist()
                best_score = float(scores[idx])
        total_feasible += int(valid.sum())
        traces.append(dict(step=step, shots=shots, feasible=int(valid.sum()),
                           mean_energy=float(energies.mean()),
                           lower_eighth_mean_energy=float(energies[ranked[:shots // 8]].mean()),
                           mean_feasible_score=float(scores[valid].mean()) if valid.any() else None,
                           best_feasible_score=float(scores[valid].min()) if valid.any() else None,
                           field_converged=converged))
        if step + 1 == len(TEMPERATURE_INVERSE) or mode == "frozen":
            continue
        frequency = samples.mean(axis=0)
        exposure = .5 * exposure + .5 * (frequency @ inst.B) / inst.K
        if mode == "trained":
            # This is a likelihood-gradient step on elite measurements. In the
            # fixed-K exponential family the score function is x - E_theta[x].
            # A variance floor moderates updates to rarely measured assets.
            direction = (elite.mean(axis=0) - frequency) / (.10 + frequency * (1. - frequency))
            feedback = np.clip(.65 * feedback + .8 * direction, -4., 4.)
    if best is not None:
        mask = np.zeros(inst.n, dtype=int)
        mask[best] = 1
        if not inst.is_feasible(mask) or not np.isclose(inst.objective(mask), best_score,
                                                        atol=1e-12, rtol=0):
            raise ArithmeticError("Returned portfolio failed organizer verification")
        best_score = inst.objective(mask)
    record = dict(selected=best, objective=best_score if best is not None else None,
                  seconds=perf_counter() - start, data_qubits=inst.n, feasible_shots=total_feasible,
                  shots=sum(t["shots"] for t in traces), trace=traces)
    if inst.n == 10:
        record["final_theta"] = theta.tolist()
    return record
