"""Run reproducible portfolio experiments; each completed instance is checkpointed."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from portfolio_generator import generate


ROOT = Path(__file__).resolve().parent
SIZES = (10, 20, 50, 100, 200, 500, 1000, 10000)
SEEDS = tuple(range(5))
GENERATOR_SHA256 = "e724347ebf8ed50c96a344f6dcc2148543bd10519a48d09c8e97447591fc86c0"


def sha(path):
    # Git may check out the identical supplied source with CRLF on Windows.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, record):
    # Replace only a fully written file. An interrupted run can then resume.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, allow_nan=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def verify_holdings(inst, record):
    if record["selected"] is None:
        return
    selected = record["selected"]
    if len(selected) != inst.K or len(set(selected)) != inst.K or not all(
            isinstance(i, (int, np.integer)) and 0 <= i < inst.n for i in selected):
        raise ValueError("Output must contain K unique, in-range integer indices")
    bits = np.zeros(inst.n, dtype=int)
    bits[selected] = 1
    if not inst.is_feasible(bits) or not np.isclose(
            inst.objective(bits), record["objective"], atol=1e-12, rtol=0):
        raise ValueError("Organizer verification failed")
    w = bits / inst.K
    record.update(annual_expected_return=float(inst.mu @ w), variance=inst.risk(w),
                  spend=float(inst.price @ bits),
                  sector_counts=np.bincount(inst.sector[selected], minlength=inst.n_sectors).tolist())


def run(method, sizes, seeds, out, resume):
    if method == "classical":
        from classical import solve
    else:
        from variational import MODES, solve
    source = ROOT / (method + ".py")
    path = out / (method + ".json")
    hashes = {p.name: sha(p) for p in (ROOT / "portfolio_generator.py", source, Path(__file__))}
    if hashes["portfolio_generator.py"] != GENERATOR_SHA256:
        raise RuntimeError("Organizer generator differs from the supplied challenge version")
    if resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["source_sha256"] != hashes:
            raise ValueError("Source changed since checkpoint. Start a fresh run.")
    else:
        payload = dict(method=method, source_sha256=hashes, python=sys.version,
                       platform=platform.platform(), processor=platform.processor(),
                       packages={p: version(p) for p in ("numpy", "scipy", "threadpoolctl")},
                       blas=[{k: v for k, v in lib.items() if k in (
                           "internal_api", "num_threads", "version", "architecture")}
                             for lib in threadpool_info()],
                       started_utc=datetime.now(timezone.utc).isoformat(), instances=[])
    done = {(r["n"], r["seed"]) for r in payload["instances"]}
    for n in sizes:
        for seed in seeds:
            if (n, seed) in done:
                continue
            inst = generate(n, seed)
            row = dict(n=n, seed=seed, scope="official" if n in SIZES[:-1] and seed in SEEDS
                       else "extension" if n == 10000 and seed in SEEDS else "additional",
                       k=inst.K, budget=inst.budget, sector_cap=inst.sector_cap)
            if method == "classical":
                record = solve(inst)
                verify_holdings(inst, record)
                row["classical"] = record
            else:
                # Cycling mode order avoids systematically awarding a warm CPU to one arm.
                for mode in MODES[seed % 3:] + MODES[:seed % 3]:
                    record = solve(inst, mode)
                    verify_holdings(inst, record)
                    row[mode] = record
            payload["instances"].append(row)
            write_json(path, payload)
            print(f"{method:11s} n={n:5d} seed={seed} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", "variational", "classical"], default="all")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        methods = ("variational", "classical") if args.method == "all" else (args.method,)
        for method in methods:
            run(method, args.sizes, args.seeds, args.output, args.resume)


if __name__ == "__main__":
    main()
