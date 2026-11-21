"""Secondary reproducibility check: vary circuit sampling with market data held fixed."""
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from portfolio_generator import generate
from run_study import verify_holdings, write_json
from variational import solve


def main():
    root = Path(__file__).resolve().parent / "results"
    primary = json.loads((root / "variational.json").read_text())
    rows = []
    # These are the first two official sizes with K >= 20 and K >= 100,
    # respectively. The default seed is already recorded in the main experiment.
    with threadpool_limits(limits=1):
        for base in primary["instances"]:
            if base["n"] not in (200, 1000):
                continue
            inst = generate(base["n"], base["seed"])
            for replicate in range(5):
                line = dict(n=inst.n, seed=inst.seed, replicate=replicate)
                for mode in ("trained", "schedule_only"):
                    rec = base[mode] if replicate == 0 else solve(inst, mode, replicate)
                    verify_holdings(inst, rec)
                    line[mode] = dict(objective=rec["objective"], selected=rec["selected"],
                                      feasible_shots=rec["feasible_shots"])
                rows.append(line)
    write_json(root / "repeats.json", dict(instances=rows))


if __name__ == "__main__":
    main()
