"""Recheck saved portfolios, export per-seed metrics, and draw the figures."""
import csv
import json
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_generator import generate
from run_study import ROOT, SEEDS, SIZES, verify_holdings


DIR = ROOT / "results"
FIG = DIR / "figures"
TEAL, ORANGE, INK, LIGHT = "#08776F", "#B44A21", "#18344B", "#B2C6CC"


def read(name):
    return json.loads((DIR / name).read_text(encoding="utf-8"))


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics():
    variational = read("variational.json")["instances"]
    klass = {(r["n"], r["seed"]): r for r in read("classical.json")["instances"]}
    audits = {(r["n"], r["seed"]): r for r in read("exact_supplement.json")["instances"]}
    pending = {key for key, r in klass.items() if r["classical"]["exact_objective"] is None}
    if set(audits) != pending:
        raise ValueError("Supplement must resolve exactly the remaining primary references")
    expected = {(n, s) for n in SIZES for s in SEEDS}
    assert set(klass) == expected and len(variational) == len(expected)
    assert {(r["n"], r["seed"]) for r in variational} == expected
    rows = []
    for rec in sorted(variational, key=lambda r: (r["n"], r["seed"])):
        n, seed = rec["n"], rec["seed"]
        inst = generate(n, seed)
        c = klass[n, seed]["classical"]
        t, frozen, schedule = (rec[k] for k in ("trained", "frozen", "schedule_only"))
        for case in (t, frozen, schedule, c):
            verify_holdings(inst, case)
            if case["objective"] is None:
                raise ValueError(f"No feasible output on {(n, seed)}; report the failure")
            if case["objective"] < c["lower_bound"] - 1e-12:
                raise ArithmeticError("A saved portfolio contradicts the certificate")
        extra = audits.get((n, seed))
        exact = extra if extra is not None else c
        ref = exact["exact_objective"]
        verify_holdings(inst, {"selected": exact["exact_selected"], "objective": ref})
        if ref < c["lower_bound"] - 1e-12 or min(t["objective"], c["objective"]) < ref - 1e-12:
            raise ArithmeticError("Exact audit contradicts a verified bound or feasible solution")
        rows.append(dict(scope=rec["scope"], n=n, seed=seed, k=inst.K,
                         trained_score=t["objective"], classical_score=c["objective"],
                         exact_objective=ref, lower_bound=c["lower_bound"], reference_exact=True,
                         exact_method="summed reduced-cost enumeration" if extra else c["exact_method"],
                         tangent_bound=c["tangent_bound"],
                         trained_gap_bp=1e4 * max(0., t["objective"] - ref),
                         classical_gap_bp=1e4 * max(0., c["objective"] - ref),
                         trained_minus_classical_bp=1e4 * (t["objective"] - c["objective"]),
                         frozen_minus_trained_bp=1e4 * (frozen["objective"] - t["objective"]),
                         schedule_minus_trained_bp=1e4 * (schedule["objective"] - t["objective"]),
                         final_valid_pct=100 * t["trace"][-1]["feasible"] / t["trace"][-1]["shots"],
                         all_valid_pct=100 * t["feasible_shots"] / t["shots"],
                         final_mean_feasible_score=t["trace"][-1]["mean_feasible_score"],
                         trained_seconds=t["seconds"], classical_seconds=c["seconds"],
                         supplement_seconds=extra["seconds"] if extra else 0.,
                         verification_seconds=c["exact_seconds"] + c["bound_seconds"]
                         + (extra["seconds"] if extra else 0.),
                         expected_return=t["annual_expected_return"], variance=t["variance"],
                         budget_use_pct=100 * t["spend"] / inst.budget))
    return rows


def summarize(rows):
    by_size = []
    for n in SIZES:
        group = [r for r in rows if r["n"] == n]
        d = dict(n=n, scope=group[0]["scope"], seeds=len(group),
                 exact_references=sum(r["reference_exact"] for r in group),
                 trained_hits=sum(r["reference_exact"] and r["trained_gap_bp"] < 1e-6 for r in group))
        for key in ("trained_score", "classical_score", "trained_gap_bp", "classical_gap_bp",
                    "trained_minus_classical_bp", "frozen_minus_trained_bp", "schedule_minus_trained_bp",
                    "final_valid_pct", "all_valid_pct", "trained_seconds", "classical_seconds",
                    "verification_seconds", "budget_use_pct"):
            d[key + "_mean"] = mean(r[key] for r in group)
            d[key + "_sd"] = stdev(r[key] for r in group)
        by_size.append(d)
    return by_size


def plot_figures(rows, summary):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#B5C5CA", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white"})
    labels = ["10", "20", "50", "100", "200", "500", "1,000", "10,000*"]
    fig, ax = plt.subplots(figsize=(8.4, 2.9), layout="constrained")
    for j, n in enumerate(SIZES):
        group = [r for r in rows if r["n"] == n]
        ax.scatter(j + np.linspace(-.13, .13, len(group)), [r["trained_gap_bp"] for r in group],
                   s=29, color=LIGHT, edgecolor=INK, linewidth=.35, zorder=3)
        ax.plot([j - .21, j + .21], [summary[j]["trained_gap_bp_mean"]] * 2,
                color=TEAL, lw=3.4, zorder=4)
    ax.set(xticks=np.arange(8), xticklabels=labels, ylabel="Gap to optimum (objective bp)",
           xlabel="Assets in the universe    (* separate stress test)")
    ax.axhline(0, lw=.6, color=INK)
    ax.grid(axis="y", alpha=.18)
    ax.text(.015, .96, "Each point is one instance. Green bars show five-seed means.",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.savefig(FIG / "certified_gaps.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 2.8), layout="constrained")
    for key, label, marker, color in [("trained_seconds", "Variational sampler", "o", TEAL),
                                      ("classical_seconds", "Classical solver", "s", ORANGE)]:
        ax.plot(SIZES, [d[key + "_mean"] for d in summary], marker=marker,
                color=color, lw=2, ms=5, label=label)
        for r in rows:
            ax.scatter(r["n"], r[key], s=10, color=color, alpha=.36)
    ax.set(xscale="log", yscale="log", xticks=SIZES, xticklabels=labels,
           xlabel="Assets (log scale)", ylabel="Wall time in seconds (log scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", which="major", alpha=.18)
    fig.savefig(FIG / "wall_time.png", dpi=200)
    plt.close(fig)

    quantum = read("quantum_audit.json")["instances"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), layout="constrained")
    for n, ax in zip((10, 20), axes):
        group = [r for r in quantum if r["n"] == n]
        y = [[100 * r["cases"][encoding][arm]["optimal_mass"] for encoding, arm in (
            ("hard", "uniform"), ("penalty", "trained"), ("hard", "trained"))] for r in group]
        ax.plot(np.arange(3), np.array(y).T, color=LIGHT, lw=1, marker="o", ms=3)
        ax.plot(np.arange(3), np.mean(y, axis=0), color=TEAL, lw=2.5, marker="o", ms=5)
        ax.set(xticks=[0, 1, 2], xticklabels=["Uniform\nfeasible", "QAOA\npenalty", "QAOA\nfeasible"],
               ylabel="Optimum probability (%)", title=f"{n} assets", xlim=(-.2, 2.2), ylim=(0, None))
        ax.grid(axis="y", alpha=.18)
    fig.savefig(FIG / "qaoa_optimum.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    all_rows = metrics()
    summary = summarize(all_rows)
    dump_csv(DIR / "instances.csv", all_rows)
    dump_csv(DIR / "by_size.csv", summary)
    plot_figures(all_rows, summary)
