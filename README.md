# Quantum portfolio selection

How close can an ideal variational sampler get to the best affordable basket of exactly K stocks? This project runs the Amsterdam Qiskit Fall Fest portfolio challenge on all **35 official instances** and on **five additional universes of 10,000 assets**. All reported portfolios were produced by the solvers and verified with the unmodified organizer generator.

## Read the results

- **Research report.** The six-page [report](research_report.pdf) gives the findings, the worked example, uncertainty across seeds, controls, runtime and quantum-circuit checks.
- **Presentation.** Open the [editable slides](presentation/portfolio-study.pptx), [slide PDF](presentation/portfolio-study.pdf) or [five-minute talk script and judge Q&A](presentation/SPEAKER_NOTES.md).
- **Experimental design.** The [pre-run plan](RESEARCH_PLAN.md) explains the competing approaches and the decisions made before scoring. The [technical appendix](METHODS.md) derives the Hamiltonians, training rule, exact sampler and certificate.
- **Data.** [One row per instance](results/instances.csv) and [summary by universe size](results/by_size.csv) give the measured scores with spreads and verified reference gaps. The [raw results](results/) preserve actual holdings, timing and circuit probabilities.

### Main findings

Every best-of-run basket passed the original cardinality, price and sector rules. Independent exact verification covered all 40 instances. The variational sampler **matched 29 of 35 official global optima**. Its largest official gap was **0.9844 objective basis points**, and its mean gap at 10,000 assets was **0.2966 objective bp**. One such point is 0.0001 in the organizer's risk-return score. Each run used 1,280 raw simulated measurements. The 10,000-asset extension used the same settings and shot budget.

On the 1,000-asset cases the sampler gained **71.07 objective bp** over a frozen, informed starting state. A control using the same concentration schedule and factor-exposure update without elite feedback scored **0.084 bp better** on average. The well-matched classical solver was faster at every size on average. The scalable circuit's measurement law has an efficient exact classical sampler.

The separate coherent QAOA experiment isolates interference on small known problems. Among 20-asset cases, two layers on feasible portfolio addresses raised the mean single-shot probability of the exact optimum from **1.31% to 8.75%**. Uniform sampling already has access to the identical feasible dictionary. Enumerating that dictionary exposes the optimum during preparation, which limits what can be inferred from this experiment.

## Reproduce

The study ran on Windows with Python 3.13.5. Use Python 3.11–3.13 and the study's pinned dependencies. The supplied generator, participant guide, PDF and original environment file are preserved in this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-research.txt
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run_study.py
python quantum_audit.py
python quantum_audit.py --weights results/variational.json
python quantum_audit.py --counter
python repeat_study.py
python exact_supplement.py
python analyze.py
python build_report.py
```

On macOS or Linux, run `source .venv/bin/activate` after creating the environment, install with `python -m pip install -r requirements-research.txt`, then run the test and experiment commands above. `python run_study.py --resume` continues completed instance checkpoints after interruption and verifies source hashes. A fresh run defaults to replacing the saved numerical output. A separate output directory preserves the recorded study during a quick trial.

```sh
python run_study.py --sizes 10 20 --seeds 77 --output pilot-results
```

The primary driver constrains BLAS thread use. Wall times include state or benchmark construction, optimization and portfolio checking, and exclude imports, data generation and independent exact verification. CPU scheduling and installed math libraries can change timings. The raw records include package versions, hardware identification and source hashes. Gate counts use Qiskit 2.6.2 and an ideal all-to-all basis. No hardware credentials are required.

## Implementation and measurement

| File | Role |
| --- | --- |
| [variational.py](variational.py) | Number-projected quantum trial state, log-space measurement, Hamiltonian, data-informed field fitting and paired controls |
| [classical.py](classical.py) | Independent feasible baseline using a factor-risk relaxation, budget and sector multipliers, exchanges, lower bounds and bounded exact enumeration |
| [exact_supplement.py](exact_supplement.py) | Guards the total reduced cost of each changed set of holdings and exhaustively verifies the remaining eight optima after timing the solvers |
| [quantum_audit.py](quantum_audit.py) | Explicit Qiskit circuits, weighted-state validation, compact coherent counter and the two-layer feasible/penalty QAOA ablation |
| [run_study.py](run_study.py) | Official sweep, separate extension, score verification and atomic checkpoints |
| [repeat_study.py](repeat_study.py) | Secondary check of shot-stream variation at fixed data on 200 and 1,000 assets |
| [analyze.py](analyze.py) | Rechecks selected holdings, compares methods on matched seeds and draws the scientific figures |
| [build_report.py](build_report.py) | Typesets the PDF from the measured data, figures and audit outputs |

The PDF challenge asks to reach 10,000 assets. Its participant guide specifies seven official sizes through 1,000 and five seeds per size. The extension has its own rows in every table. All data are synthetic. The topic rules specify noise-free simulation, so the study uses independent statevector cross-validation and a separate analytical readout sensitivity calculation. The results establish benchmark accuracy and sampling effects on this generator.

The primary run followed the research plan. Two follow-up investigations were added after inspecting the original run. Independent shot streams separate sampling variation from variation across markets. The exhaustive reduced-cost check resolves eight initially bounded cases. Neither follow-up changes the original portfolios, the paired control results, or solver timing.
