"""Run any solver on an official instance and score it with the generator's own methods.

A solver is a callable ``solver(inst) -> (x, info)``: ``x`` is the 0/1 portfolio and
``info`` holds optional resource counts (qubits, evaluations, shots, ...). Scoring always
goes through ``inst.objective`` and ``inst.is_feasible``, never through our own QUBO, so a
bug in a Hamiltonian can never make a result look better than it is.
"""
import csv
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from portfolio_generator import generate

Q = 0.5  # risk aversion fixed by the hackathon rules


@dataclass
class Result:
    method: str
    n: int
    seed: int
    objective: float
    feasible: bool
    seconds: float
    qubits: int = 0
    evaluations: int = 0          # circuit / loss evaluations used by the optimiser
    shots: int = 0
    raw_feasible: float | None = None   # fraction of quantum samples feasible before repair
    extra: dict = field(default_factory=dict)
    selected: str = ""            # space-separated indices of the chosen assets


def run(method: str, solver, n: int, seed: int) -> Result:
    inst = generate(n, seed)
    t0 = time.perf_counter()
    x, info = solver(inst)
    seconds = time.perf_counter() - t0
    x = np.asarray(x).astype(int)
    known = {f.name for f in fields(Result)}
    extra = {k: v for k, v in info.items() if k not in known}
    return Result(method, n, seed, inst.objective(x, Q), inst.is_feasible(x), seconds,
                  extra=extra, selected=" ".join(map(str, np.flatnonzero(x))),
                  **{k: v for k, v in info.items() if k in known})


def write_csv(results: list[Result], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
