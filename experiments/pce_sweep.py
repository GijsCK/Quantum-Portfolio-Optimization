"""PCE sweep with controls. For each (n, seed):

  pce          entangled brickwork PCE (k = 2 for n <= 100, else 3)
  pce-product  the same circuit without CZ gates (product state): does entanglement matter?
  relax        the same relaxed loss optimised directly over n classical variables
  random       random scores (uniform random K-portfolio)

Every method's restarts are decoded the same way (top-K, repair) and scored twice:
  raw    objective-blind repair only: measures the method itself
  ls     + swap local search: what a practitioner would ship

    python -m experiments.pce_sweep --sizes 50 100 200 500 [--seeds 0 1 2 3 4]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from portfolio_generator import generate
from qportfolio.classical import miqp
from qportfolio.metrics import normalised_score, random_feasible_mean
from qportfolio.pce import (RelaxedLoss, best_decoded, qubits_needed, train_pce,
                            train_relaxation)
from qportfolio.problem import feasible_mask

RESTARTS, MAXITER, ALPHA_MULT, MAX_LAYERS = 2, 1500, 5.0, 32


def layers_for(n, N):
    return int(min(MAX_LAYERS, np.ceil(3 * n / (2 * N))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--methods", nargs="+", default=["pce", "pce-product", "relax", "random"])
    ap.add_argument("--out", default="results/pce_sweep.csv")
    ap.add_argument("--k", type=int, help="correlator body-ness (default 2 if n <= 100 else 3)")
    ap.add_argument("--layers", type=int, help="override the depth rule")
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = not out.exists()
    with out.open("a", newline="") as fh:
        writer = None
        for n in a.sizes:
            k = a.k or (2 if n <= 100 else 3)
            N = qubits_needed(n, k)
            L = a.layers or layers_for(n, N)
            for seed in a.seeds:
                inst = generate(n, seed)
                x_opt, _ = miqp(inst)
                f_opt, f_rand = inst.objective(x_opt), random_feasible_mean(inst)
                loss = RelaxedLoss(inst)
                for method in a.methods:
                    t0 = time.perf_counter()
                    info = {}
                    if method in ("pce", "pce-product"):
                        runs, info = train_pce(inst, k=k, layers=L, alpha=ALPHA_MULT * N,
                                               entangle=(method == "pce"), restarts=a.restarts,
                                               maxiter=a.maxiter, seed=seed, loss=loss)
                        t = [np.tanh(ALPHA_MULT * N * c) for c in runs]
                    elif method == "relax":
                        runs, info = train_relaxation(inst, restarts=a.restarts,
                                                      maxiter=a.maxiter, seed=seed, loss=loss)
                        t = runs
                    else:
                        rng = np.random.default_rng(seed)
                        runs = t = [rng.normal(size=n) for _ in range(a.restarts)]
                    secs = time.perf_counter() - t0
                    raw = best_decoded(inst, runs, polish=False)
                    ls = best_decoded(inst, runs, polish=True)
                    topk = np.zeros(n, int)
                    topk[np.argsort(-runs[0])[:inst.K]] = 1
                    row = {
                        "method": method, "n": n, "seed": seed,
                        "qubits": info.get("qubits", n), "k": k if method.startswith("pce") else "",
                        "layers": L if method.startswith("pce") else "",
                        "params": info.get("params", 0),
                        "loss": min(loss((1 + ti) / 2)[0] for ti in t),
                        "loss_at_optimum": loss(x_opt.astype(float))[0],
                        "score_raw": normalised_score(inst.objective(raw), f_opt, f_rand),
                        "score_ls": normalised_score(inst.objective(ls), f_opt, f_rand),
                        "opt_raw": bool(np.isclose(inst.objective(raw), f_opt)),
                        "opt_ls": bool(np.isclose(inst.objective(ls), f_opt)),
                        "topk_feasible_before_repair": bool(feasible_mask(inst, topk)[0]),
                        "evaluations": info.get("evaluations", 0),
                        "seconds": secs,
                    }
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=row.keys())
                        if first:
                            writer.writeheader()
                    writer.writerow(row)
                    fh.flush()
                    print(f"n={n} seed={seed} {method:11s} N={row['qubits']} params={row['params']} "
                          f"loss={row['loss']:.3f} (opt {row['loss_at_optimum']:.3f}) "
                          f"raw={row['score_raw']:.3f} ls={row['score_ls']:.4f} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
