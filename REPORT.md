# Where does quantum help in portfolio optimization?

Amsterdam Qiskit Fall Fest 2026 · Topic 1 · Gijs-Branch

## The answer in brief

On the official instances, no quantum method we built beats a classical solver on portfolio quality or speed, and the problem leaves no room for that. SCIP proves the optimum for every official instance in under 0.5 s, and at 10,000 assets in about 100 s.

The quantum part still does measurable, non-trivial work in two ways:

1. **Pauli correlation encoding (PCE) depends on entanglement.** PCE stores 1000 buy-or-skip decisions on 14 qubits. The same circuit without entangling gates scores no better than random. An exhaustive check shows that no product state can encode the optimal portfolio for n ≥ 100.
2. **PCE's readout tolerates noise and QAOA's does not.** The PCE portfolio stays the same at a depolarizing error rate that shrinks its signal by 81%. QAOA's output falls below random guessing at a similar error budget.

PCE's limit is that optimising the *same* relaxed loss directly over n classical variables is as good or better, and about 10,000 times faster.

## What we built, mapped to the four tasks

| Task in the prompt | What we did |
|---|---|
| 1. Build the Hamiltonian | A hand-built QUBO and Ising Hamiltonian: the official objective, a squared cardinality penalty, and an unbalanced or slack-qubit budget term. Verified on every bitstring for n = 10 (`qportfolio/problem.py`). |
| 2. Solve it variationally | **QAOA** with an X mixer, or an XY ring mixer with a Dicke-state start (`qaoa.py`). **PCE** with k-body Pauli correlators, a brickwork ansatz and exact adjoint gradients; the simulator is checked against Qiskit `Statevector` (`pce.py`). |
| 3. Scale up | PCE puts 1000 assets on 14 qubits and 10,000 on 19 qubits. |
| 4. Classical benchmark | The SCIP MIQP solver (a proven optimum on every instance), swap local search, and a classical relaxation of the PCE loss. |

### Encoding, qubits and constraints (required by the rules)

| Method | Encoding | Qubits at n = 20 / 1000 / 10,000 | Exactly K assets | Budget | Sector caps |
|---|---|---|---|---|---|
| QAOA, X mixer | 1 qubit per asset | 20 / – / – | Penalty (Σx − K)² | Unbalanced penalty, then keep the best feasible sample | Never binds at n ≤ 20 |
| QAOA, XY + Dicke | 1 qubit per asset | 20 / – / – | **Built into the circuit**: Dicke start plus a number-preserving mixer | Same as above | Same as above |
| PCE | xᵢ = [⟨Πᵢ⟩ > 0], k-body Paulis | 5 / **14** / **19** | Squared penalty in the loss, then top-K decoding | **Exact hinge** max(0, spend − budget)² in the loss, then repair | Exact hinge, then repair |

"Repair" is an objective-blind classical step that only fixes violations. It is the same for every method and every control.

## Findings

**1. The classical benchmark is exact and fast.**
- Brute force, SCIP and local search agree on every instance with n ≤ 50.
- SCIP proves the optimum up to n = 1000 in under 0.5 s, and at n = 10,000 in about 100 s.
- Swap local search from random starts reaches the exact optimum 89% of the time at n = 50, but 0% of the time at n = 1000, so exact answers at scale need a real solver.

Every quantum result below is scored against these proven optima.

**2. The budget rule cannot be written cleanly as a QAOA Hamiltonian.**
- A quadratic penalty on an inequality also punishes legal portfolios that leave budget unused. Objective differences between good portfolios (about 10⁻³) are small next to that penalty.
- With the best tuned unbalanced penalty, the ground state was still not the true optimum on 3 of 10 small instances. In the worst case the optimum ranked 101st by energy.
- Five slack qubits improved the worst rank only to about 45, because the slack has too coarse a resolution.
- PCE avoids the problem. Its loss is computed classically from measured correlators, so it can use an exact hinge penalty that is zero on every legal portfolio.

**3. QAOA concentrates probability, but not usefully at these sizes.**

All figures below are means over 5 seeds, from exact output distributions.

| | P(optimum) n = 10 | P(optimum) n = 20 | P(feasible) n = 20 | Two-qubit gates at n = 20 |
|---|---|---|---|---|
| QAOA, XY + Dicke, p = 3 | **0.169** | **0.042** | **0.68** | 800 |
| Uniform over K-asset portfolios | 0.022 | 0.005 | 0.41 | – |
| QAOA, X mixer, p = 3 | 0.015 | 0.003 | 0.22 | 570 |

- XY QAOA raises the chance of measuring the optimum about 8-fold.
- But with 1024 shots, uniform sampling of the same C(n, K) portfolios also finds the optimum in 100% of runs. At K = 2 there are only 45 or 190 candidate portfolios.
- The Dicke start is essential. With a single basis-state start, the ring mixer put 0% of its probability on feasible portfolios at n = 20.

**4. PCE needs entanglement.**

Raw scores before any local search. 1 = optimal, 0 = a random portfolio after the same repair. Mean ± sd over seeds 0–4.

| n | Qubits | Parameters | **PCE (entangled)** | Same circuit without CZ gates | Classical relaxation of the same loss | Random |
|---|---|---|---|---|---|---|
| 50 | 7 | 168 | 0.76 ± 0.23 | 0.18 ± 0.16 | 0.96 ± 0.06 | −0.01 ± 0.19 |
| 100 | 9 | 324 | 0.87 ± 0.07 | 0.16 ± 0.18 | 0.94 ± 0.09 | 0.12 ± 0.10 |
| 200 | 9 | 594 | **0.98 ± 0.01** | 0.05 ± 0.24 | 0.97 ± 0.04 | 0.02 ± 0.05 |
| 500 | 12 | 792 | 0.97 ± 0.01 | 0.08 ± 0.09 | 0.99 ± 0.01 | 0.05 ± 0.03 |
| 1000 | 14 | 924 | 0.88 ± 0.02 | 0.05 ± 0.03 | 0.99 ± 0.01 | 0.03 ± 0.04 |
| 10,000 (1 seed) | 19 | 342 | −0.02 (did not train) | – | 0.99 | −0.01 |

- Without entanglement, training never satisfies the constraints: the loss stays at 10³–10⁵. The decoded portfolios score no better than random.
- The cause is structural, not a training failure. In a product state every k-body correlator factorises into single-qubit values. We enumerated every sign pattern a product state can produce. For n ≥ 100, none of them encodes the optimal portfolio, and at n = 1000 the best one gets 89 of the 1000 decisions wrong.
- **Entanglement is what lets 14 qubits carry 1000 independent decisions.** This is the "quantum resource" question from the prompt, answered with a control.

**5. ...but the quantum circuit is a compressed classical relaxation, not a better optimiser.**
- Optimising the same loss directly over n classical variables matches or beats PCE at every size except n = 200.
- It takes 5–90 ms, against 9 s–14 min for the PCE simulation.
- After swap local search, PCE and relaxation starts score 0.994–1.000 and random starts 0.969–1.000. At n = 1000, PCE and product-state starts each reached the exact optimum on 1 of 5 seeds, against 0 of 5 for relaxation and random starts. The product state's raw quality is no better than random, so this is consistent with luck, not with an advantage.

**6. Noise and shots (bonus question).**

Trained circuits were read out under depolarizing noise in Aer's density-matrix simulator. Errors on 1-qubit gates are set to 1/10 of the two-qubit error rate. This is an unscored experiment.

| Expected two-qubit errors per run | PCE, n = 100 (68 CZ) | QAOA, XY, n = 10 (506 CX) |
|---|---|---|
| ≈ 0.05–0.07 | Portfolio unchanged | P(opt) 0.21 → 0.20 |
| ≈ 0.5–0.7 | Portfolio unchanged | P(opt) 0.14; 71% of samples keep K assets |
| ≈ 1.5–2 | **Portfolio unchanged**, correlators shrunk to 19% | P(opt) 0.06; 40% keep K assets |
| ≈ 5 | – | P(opt) 0.006, **below uniform** (0.022) |

- PCE decodes by *ranking* correlators, and depolarizing noise mostly shrinks all of them together.
- The cost is shots. About 1,000–3,000 shots per basis reproduce the noise-free decoding. Shrinking the signal by a factor s needs about 1/s² times more shots.
- QAOA's "exactly K assets" guarantee only holds without noise.

**7. Runtime scaling (bonus question).**
- Simulated PCE time grows from 9 s (n = 50) to about 14 min (n = 1000) on 14 qubits. Classical times stay in milliseconds: SCIP under 0.5 s.
- On hardware, PCE needs only N = O(n^{1/k}) qubits. But training with parameter-shift gradients costs about 3 bases × 3,000 shots × 2·(parameters) × 1,500 iterations, which is about **10¹⁰ shots**, or 10–30 days at 10 kHz. The qubit saving is real; the work moves into measurements.
- One-qubit-per-asset QAOA needs n qubits and O(n²) two-qubit gates per layer (190 ZZ terms per layer at n = 20).

## Where quantum helps, and where it doesn't

| Helps (measured) | Does not help |
|---|---|
| Qubit compression that depends on entanglement: 1000 assets on 14 qubits | Portfolio quality: classical solvers already prove the optimum |
| A noise-tolerant readout (PCE) | Wall-clock time, simulated or projected to hardware |
| About 8× concentration on the optimum over uniform sampling (QAOA) | Beating a classical relaxation of the same loss |
| Exact inequality penalties without slack qubits (PCE loss) | 10,000 assets within our training budget |

## Caveats

- Everything runs on noise-free statevector simulation, except the labelled noise study.
- PCE hyperparameters (α = 5N, depth, iterations) were tuned on seed 0 at n = 100 and n = 500. That tuning slightly favours PCE, and PCE still did not beat the relaxation.
- At n = 1000 we use 5 seeds with 2 training restarts each, and at 10,000 a single seed with a deliberately reduced budget.
- The QUBO penalty settings were chosen on all 10 small instances.

## Reproduce

```
.venv/bin/python -m pytest -q                                      # 29 tests
.venv/bin/python -m experiments.qaoa_small --sizes 10 20 --mixers x xy
.venv/bin/python -m experiments.pce_sweep --sizes 50 100 200 500
.venv/bin/python -m experiments.pce_sweep --sizes 1000 --seeds 0 1 2 --out results/pce_sweep_1000.csv
.venv/bin/python -m experiments.pce_sweep --sizes 1000 --seeds 3 4 --out results/pce_sweep_1000b.csv
.venv/bin/python -m experiments.pce_sweep --sizes 10000 --seeds 0 --k 4 --layers 8 --maxiter 300 \
    --restarts 1 --methods relax random pce --out results/pce_10k.csv
.venv/bin/python -m experiments.pce_noise --n 100 && .venv/bin/python -m experiments.qaoa_noise
.venv/bin/python -m experiments.summarize                          # tables above
```

Raw per-instance results are in `results/`, and the summary tables are in `results/summary.md`.

Three notes on the data:
- The 10,000-asset scores in `pce_10k.csv` were rescored against SCIP's proven optimum from a 300 s run. The sweep's 60 s SCIP limit was not enough at that size.
- The `score` column in `qaoa_small.csv` uses an earlier random-baseline definition. The QAOA findings above use only P(opt), P(feasible) and found-opt, which do not depend on it.
- Running the QAOA experiment at n = 20 takes about 1 h. A PCE run at n = 1000 takes about 14 min per seed.
