"""Print the report tables from the result CSVs (mean +/- sd over seeds).

    python -m experiments.summarize
"""
from pathlib import Path

import pandas as pd

R = Path("results")


def md(df):
    """Markdown table without the optional ``tabulate`` dependency."""
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pm(s):
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def pce_tables():
    files = [f for f in ("pce_sweep.csv", "pce_sweep_1000.csv", "pce_sweep_1000b.csv", "pce_10k.csv")
             if (R / f).exists() and (R / f).stat().st_size > 0]
    if not files:
        return
    df = pd.concat([pd.read_csv(R / f) for f in files])
    print("\n## PCE vs controls: raw score (objective-blind repair), mean ± sd over seeds")
    order = ["pce", "pce-product", "relax", "random"]
    rows = []
    for n, g in df.groupby("n"):
        row = {"n": n, "seeds": g.seed.nunique(),
               "qubits (pce)": int(g[g.method == "pce"].qubits.iloc[0]) if (g.method == "pce").any() else "",
               "params (pce)": int(g[g.method == "pce"].params.iloc[0]) if (g.method == "pce").any() else ""}
        for m in order:
            h = g[g.method == m]
            if len(h):
                row[f"{m} raw"] = pm(h.score_raw)
        rows.append(row)
    print(md(pd.DataFrame(rows)))

    print("\n## After swap local search: fraction of instances reaching the exact optimum")
    rows = []
    for n, g in df.groupby("n"):
        row = {"n": n}
        for m in order:
            h = g[g.method == m]
            if len(h):
                row[f"{m} opt-hit"] = f"{h.opt_ls.mean() * 100:.0f}%"
                row[f"{m} ls score"] = pm(h.score_ls)
        rows.append(row)
    print(md(pd.DataFrame(rows)))

    print("\n## Loss reached vs loss of the true optimum (lower = better fit)")
    print(md(df.groupby(["n", "method"])[["loss", "loss_at_optimum", "seconds"]].mean()
          .round(3)))


def qaoa_table():
    f = R / "qaoa_small.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    print("\n## QAOA vs uniform sampling of the same search space (exact output distributions)")
    rows = []
    for (n, method), g in df.groupby(["n", "method"]):
        rows.append({"n": n, "method": method, "seeds": len(g),
                     "P(opt)": pm(g.p_opt), "P(feasible)": pm(g.p_feasible),
                     "found opt (1024 shots)": f"{g.found_opt.mean() * 100:.0f}%",
                     "2q gates": int(g.two_qubit_gates.max()), "sec": round(g.seconds.mean(), 1)})
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    pce_tables()
    qaoa_table()
