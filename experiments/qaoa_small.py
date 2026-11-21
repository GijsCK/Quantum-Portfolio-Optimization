"""Small-n QAOA study (one qubit per asset): mixers vs. the random-sampling control.

For every (n, seed, mixer) we record the exact output distribution's quality (probability on
the optimum and on feasible portfolios) and the result of a 1024-shot readout. The control
samples the same number of shots uniformly from the same search space (all 2^n bitstrings for
the X mixer, all weight-K bitstrings for the XY mixers) with identical post-selection, so any
difference is due to the circuit.

    python -m experiments.qaoa_small [--sizes 10 20] [--mixers x xy xy-full] [--reps 3]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from portfolio_generator import generate
from qportfolio.classical import miqp
from qportfolio.metrics import normalised_score, random_feasible_mean
from qportfolio.problem import build_qubo, feasible_mask
from qportfolio.qaoa import readout, train

SHOTS = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--mixers", nargs="+", default=["x", "xy", "xy-full"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/qaoa_small.csv")
    a = ap.parse_args()

    rows = []
    for n in a.sizes:
        for seed in a.seeds:
            inst = generate(n, seed)
            x_opt, _ = miqp(inst)
            f_opt, f_rand = inst.objective(x_opt), random_feasible_mean(inst)
            k_opt = int(x_opt @ (1 << np.arange(n)))
            feas_all = feasible_mask(inst, (np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
            for mixer in a.mixers:
                qubo = build_qubo(inst, sector=None, cardinality=(mixer == "x"))
                space = (np.ones(1 << n, bool) if mixer == "x"
                         else np.bitwise_count(np.arange(1 << n)) == inst.K)
                t0 = time.perf_counter()
                out = train(qubo, inst, reps=a.reps, mixer=mixer, seed=seed)
                secs = time.perf_counter() - t0
                P = out["probs"]
                uniform = space / space.sum()
                for method, probs in ((f"qaoa-{mixer}", P), (f"random-{mixer}", uniform)):
                    x, info = readout(inst, qubo, probs, SHOTS, seed)
                    f = inst.objective(x)
                    rows.append({
                        "method": method, "n": n, "seed": seed, "reps": a.reps,
                        "qubits": n, "feasible": inst.is_feasible(x), "objective": f,
                        "f_opt": f_opt, "score": normalised_score(f, f_opt, f_rand),
                        "found_opt": bool(np.isclose(f, f_opt)),
                        "p_opt": probs[k_opt], "p_feasible": probs[feas_all].sum(),
                        "raw_feasible": info["raw_feasible"], "distinct": info["distinct"],
                        "search_space": int(space.sum()),
                        "evaluations": out["evaluations"] if method.startswith("qaoa") else 0,
                        "two_qubit_gates": out["two_qubit_gates"] if method.startswith("qaoa") else 0,
                        "seconds": secs if method.startswith("qaoa") else 0.0,
                    })
                q, r = rows[-2], rows[-1]
                print(f"n={n} seed={seed} {mixer:7s} {secs:6.1f}s | P(opt) qaoa={q['p_opt']:.4f} "
                      f"uniform={r['p_opt']:.4f} | P(feas) qaoa={q['p_feasible']:.2f} "
                      f"uniform={r['p_feasible']:.2f} | found_opt qaoa={q['found_opt']} "
                      f"random={r['found_opt']}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
