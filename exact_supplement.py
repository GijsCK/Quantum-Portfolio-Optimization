"""Post-study exact checks using the classical solver's saved dual certificates.

A basket that improves the incumbent can reverse only a limited total reduced
cost. Enumerate affordable additions and removals separately and join them by
the exact holding count. This supplement does not modify the original run.
"""
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from classical import BOUND_GUARD, FactorModel
from portfolio_generator import generate


def affordable_groups(indices, prices, limit):
    """Each affordable subset appears once, in the bucket of its cardinality."""
    order = indices[np.argsort(prices[indices], kind="stable")]
    buckets = defaultdict(list)

    def extend(start, spent, chosen):
        buckets[len(chosen)].append((spent, chosen))
        for position in range(start, len(order)):
            asset = int(order[position])
            cost = spent + prices[asset]
            if cost > limit:
                # All later assets cost at least this much to reverse.
                break
            extend(position + 1, cost, (*chosen, asset))

    extend(0, 0.0, ())
    return buckets


def subset_arrays(records, count):
    records.sort(key=lambda item: item[0])
    cost = np.fromiter((item[0] for item in records), dtype=float, count=len(records))
    picks = np.asarray([item[1] for item in records], dtype=int).reshape(len(records), count)
    return cost, picks


def subset_totals(model, picks):
    """Precompute the change in price, sector counts and factor exposure."""
    inst = model.inst
    # Gather by one column at a time to keep memory proportional to the number
    # of affordable subsets even when several reversals belong to each subset.
    price = np.zeros(len(picks))
    sectors = np.zeros((len(picks), inst.n_sectors), dtype=np.int32)
    exposure = np.zeros((len(picks), inst.B.shape[1]))
    linear = np.zeros(len(picks))
    rows = np.arange(len(picks))
    for column in picks.T:
        price += inst.price[column]
        sectors[rows, inst.sector[column]] += 1
        exposure += inst.B[column]
        linear += model.c[column]
    return price, sectors, exposure, linear


def verify(inst, classical):
    started = perf_counter()
    model = FactorModel(inst)
    cert = classical["certificate"]
    z = np.asarray(cert["exposure"])
    sector = np.asarray(cert["sector_multipliers"])
    price = cert["price_multiplier"]
    threshold = cert["threshold"]
    if price < 0 or (sector < 0).any():
        raise ValueError("Inequality multipliers must be nonnegative.")
    reduced = model.c + 2 * model.a * (inst.B @ z) + price * inst.price
    reduced += sector[inst.sector] - threshold
    constant = (-model.a * (z @ z) - price * model.budget
                - inst.sector_cap * sector.sum() + threshold * inst.K)
    lower = float(constant + np.minimum(reduced, 0).sum() - BOUND_GUARD)
    if abs(lower - classical["tangent_bound"]) > BOUND_GUARD:
        raise ValueError("The saved certificate failed reconstruction.")
    best = np.zeros(inst.n, dtype=int)
    best[classical["selected"]] = 1
    if not inst.is_feasible(best):
        raise ValueError("Starting incumbent failed the original constraints.")
    best_value = inst.objective(best)
    limit = best_value - lower + BOUND_GUARD
    if limit < 0:
        raise ArithmeticError("Lower bound exceeds the feasible incumbent.")

    # The Lagrangian's unconstrained minimum selects all negative coefficients.
    # Changing one sign increases its lower bound by exactly |reduced[i]|.
    base = reduced < 0
    remove = affordable_groups(np.flatnonzero(base), abs(reduced), limit)
    add = affordable_groups(np.flatnonzero(~base), abs(reduced), limit)
    base_price = inst.price[base].sum()
    base_sectors = np.bincount(inst.sector[base], minlength=inst.n_sectors)
    base_exposure = inst.B[base].sum(axis=0)
    base_linear = model.c[base].sum()
    required_net_additions = inst.K - int(base.sum())
    affordable_pairs = 0
    official_checks = 0
    # Loosen the screening budget slightly. Full official checks decide each
    # possible improvement, including candidates near the floating-price limit.
    budget_screen = model.budget + 1e-12 * max(1.0, inst.budget)
    for count, outgoing in remove.items():
        incoming_count = count + required_net_additions
        if incoming_count not in add:
            continue
        out_cost, out_picks = subset_arrays(outgoing, count)
        in_cost, in_picks = subset_arrays(add[incoming_count], incoming_count)
        # For each outgoing subset, all compatible incoming subsets form a
        # prefix of the incoming cost order. This avoids a Cartesian product.
        prefix = np.searchsorted(in_cost, limit - out_cost, side="right")
        ends = np.cumsum(prefix)
        total = int(ends[-1])
        affordable_pairs += total
        if not total:
            continue
        start_of_row = ends - prefix
        pr_out, sc_out, ex_out, ln_out = subset_totals(model, out_picks)
        pr_in, sc_in, ex_in, ln_in = subset_totals(model, in_picks)
        for start in range(0, total, 8192):
            pairs = np.arange(start, min(start + 8192, total))
            out_id = np.searchsorted(ends, pairs, side="right")
            in_id = pairs - start_of_row[out_id]
            okay = base_price + pr_in[in_id] - pr_out[out_id] <= budget_screen
            okay &= np.all(base_sectors + sc_in[in_id] - sc_out[out_id]
                           <= inst.sector_cap, axis=1)
            out_id, in_id = out_id[okay], in_id[okay]
            if not len(out_id):
                continue
            exposure = base_exposure + ex_in[in_id] - ex_out[out_id]
            values = (base_linear + ln_in[in_id] - ln_out[out_id]
                      + model.a * np.einsum("ij,ij->i", exposure, exposure))
            finalists = np.flatnonzero(values <= best_value + BOUND_GUARD)
            for at in finalists:
                candidate = base.astype(int)
                candidate[out_picks[out_id[at]]] = 0
                candidate[in_picks[in_id[at]]] = 1
                official_checks += 1
                if inst.is_feasible(candidate):
                    official = inst.objective(candidate)
                    if official < best_value:
                        best, best_value = candidate, official
    return {
        "n": inst.n, "seed": inst.seed, "exact_selected": np.flatnonzero(best).tolist(),
        "exact_objective": best_value, "seconds": perf_counter() - started,
        "certificate_lower": lower, "original_incumbent": classical["objective"],
        "guarded_reversal_budget": limit, "affordable_pairs": affordable_pairs,
        "affordable_add_subsets": sum(map(len, add.values())),
        "affordable_drop_subsets": sum(map(len, remove.values())),
        "official_finalist_checks": official_checks,
    }


def audit_references(original_rows, extra_rows):
    """Re-evaluate every exact basket and the full dual identity including slacks."""
    extras = {(row["n"], row["seed"]): row for row in extra_rows}
    worst_residual = 0.0
    for row in original_rows:
        key = row["n"], row["seed"]
        classical = row["classical"]
        result = extras.get(key, classical)
        inst = generate(*key)
        model = FactorModel(inst)
        chosen = np.zeros(inst.n, int)
        chosen[result["exact_selected"]] = 1
        if not inst.is_feasible(chosen) or abs(inst.objective(chosen) - result["exact_objective"]) > BOUND_GUARD:
            raise ArithmeticError("A stored exact basket failed the organizer's check.")
        cert = classical["certificate"]
        z = np.asarray(cert["exposure"])
        price = cert["price_multiplier"]
        sector = np.asarray(cert["sector_multipliers"])
        reduced = (model.c + 2 * model.a * (inst.B @ z) + price * inst.price
                   + sector[inst.sector] - cert["threshold"])
        flips = chosen != (reduced < 0)
        counts = np.bincount(inst.sector[chosen == 1], minlength=inst.n_sectors)
        reconstructed = (classical["tangent_bound"] + BOUND_GUARD + abs(reduced[flips]).sum()
                         + price * (model.budget - inst.price @ chosen)
                         + sector @ (inst.sector_cap - counts)
                         + model.a * np.linalg.norm(inst.B.T @ chosen - z) ** 2)
        worst_residual = max(worst_residual, float(abs(inst.objective(chosen) - reconstructed)))
    if worst_residual > BOUND_GUARD:
        raise ArithmeticError("The full Lagrangian identity failed the roundoff guard.")
    return {"validated_portfolios": len(original_rows), "max_dual_identity_residual": worst_residual}


def main():
    root = Path(__file__).resolve().parent
    original = json.loads((root / "results/classical.json").read_text(encoding="utf-8"))
    rows = []
    with threadpool_limits(limits=1):
        for row in original["instances"]:
            if row["classical"]["exact_objective"] is not None:
                continue
            result = verify(generate(row["n"], row["seed"]), row["classical"])
            rows.append(result)
            print(f"n={result['n']} seed={result['seed']} pairs={result['affordable_pairs']:,} "
                  f"score={result['exact_objective']:.12g} time={result['seconds']:.2f}s", flush=True)
    output = {"method": "Post-study exhaustive joined reduced-cost reversals",
              "objective_guard": BOUND_GUARD, "instances": rows,
              "official_readback": audit_references(original["instances"], rows)}
    (root / "results/exact_supplement.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
