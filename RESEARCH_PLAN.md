# Research protocol · Quantum portfolio selection

This protocol sets the design before implementation and scored runs. The question is how closely a variational quantum state can approach a constrained portfolio optimum, at what computational cost, and which part of the observed performance comes from the encoding.

## 1. Scope and scoring

Use the unmodified organizer generator, default parameters, binary holdings, equal weights, and risk preference q = 0.5. The participant guide defines the official sweep as 10, 20, 50, 100, 200, 500 and 1,000 assets, seeds 0–4. The PDF also asks to reach 10,000. Run those five larger instances as a separately labeled extension. Each size has independent synthetic data. Use ideal simulation throughout as specified in the topic guide. Address the rubric's validation criterion through independent circuit simulation, exhaustive references and mathematical bounds. Five instance seeds support descriptive spread, not claims about market populations.

## 2. Objective and constraints

For a selected set S of K holdings, use the generator's score

    f(S) = q ( ||sum B_i||² + sum d_i ) / K² − (1−q) sum μ_i / K.

The common factor exposure adds before it is squared. This captures the risk of buying assets with similar exposures. At exactly K holdings, the diagonal objective Hamiltonian assigns this number to each basis portfolio. Use original floating prices and the generator's tolerance, plus the per-sector cap. Add an indicator penalty larger than the entire possible fixed-K score range and a graded excess term to distinguish invalid outcomes. On fixed-K states, variance lies between zero and the maximum individual variance by convexity. That observation provides a safe penalty size. Read all final holdings back through the official functions. Record infeasible shots and failures explicitly.

## 3. Encodings and choices

**Main solver.** Use an n-data-qubit number-projected product state, also called a weighted Dicke state here. The squared amplitude of a K-asset set is proportional to the product of its asset weights. Every K-asset portfolio has support for finite parameters. The state can range from the uniform fixed-count superposition to nearly deterministic portfolios. Exact conditional sampling avoids a 2ⁿ statevector, preserves cardinality on every shot and requires no discarded assets or solution repair. Its efficient classical simulation also sets the limit on any advantage claim. The corresponding classical sampling law and the uniform Dicke state preparation literature give independent points of reference. [Chen, Dempster & Liu](https://doi.org/10.1093/biomet/81.3.457) · [Bärtschi & Eidenbenz](https://arxiv.org/abs/2207.09998)

Use the objective's marginal return and factor risk to initialize asset rotations. Use price and sector fields to center the sampling distribution near the allowed region. Train the variational parameters on measured portfolios with an increasing concentration schedule and elite feedback. Low-energy samples guide the next state, following the lower-tail motivation of CVaR variational optimization. Specify the exact update in code and the report. No baseline portfolio enters the variational search and no classical local search improves its outputs. [Barkoutsos et al.](https://quantum-journal.org/papers/q-2020-04-20-256/)

**Small controlled experiment.** Also run an interference-based QAOA with a Grover mixer, encoding portfolio addresses in binary. At 10 and 20 assets enumerate the feasible dictionary and charge its construction explicitly. Compare a feasible dictionary with a fixed-cardinality dictionary containing penalized infeasible addresses. Compare each trained distribution with its uniform initial state and verify the circuit probabilities independently in Qiskit. Count address qubits, explain padding and disclose the enumeration cost. This tests coherence and constraint handling directly. [Bärtschi & Eidenbenz](https://arxiv.org/abs/2006.00354) · [Hadfield et al.](https://arxiv.org/abs/1709.03489)

**Why these options.** A full statevector with one bit per asset becomes impractical early. Restricting to K holdings still leaves a combinatorial basis. A feasible address encoding saves logical qubits on enumerated cases but moves work into a dictionary. Pauli correlation encoding is an alternative, especially when qubit count is the limiting resource. Decoding many expectations and enforcing this exact floating-price budget introduce separate questions. The published portfolio application optimizes recursive clustering. The supplied sampling-VQA example includes local post-processing on another objective. Neither is a matched baseline here. [Sciorilli et al.](https://arxiv.org/abs/2401.09421) · [Soloviev & Krompiec](https://arxiv.org/html/2511.21305v1) · [Agliardi et al.](https://arxiv.org/abs/2508.13557)

## 4. Comparators and proof of quality

Build an independent classical solver with a price multiplier, exact sector-cap handling and feasible exchanges. Enumerate at 10 and 20, and at 50 if practical. For larger instances, use the identity ||z||² ≥ 2aᵀz−||a||² to lower-bound risk for every possible portfolio. Optimize or bound the resulting linear selection problem including price and sector constraints. Any feasible score and any global lower bound bracket the unknown optimum. Report the width in score units multiplied by 10,000, called objective bp. These units describe the risk-return score, rather than a realized investment return. Use exact references for small cases. Strong classical baselines are material to the interpretation. [Stopfer & Wagner](https://arxiv.org/abs/2509.17876)

## 5. Measurements and controls

- Save every instance's selected indices, objective, feasibility, cost, sector use, run time, sample counts and lower bound. Summarize means and sample standard deviations across five paired instance seeds. Log software, timing boundaries and random seeds.
- Fix the sampling budget in advance at eight training rounds of 128 shots and 256 final shots. Return the best feasible raw measurement over all 1,280 shots. Record final-batch quality and feasibility separately from best-of-run results. Use two controls with equal shot budgets: a frozen informed initial state, and the same concentration schedule without elite feedback. Use pilot development seeds outside 0–4 for numerical checks. Record any departures.
- Compare objective difference to the strong classical incumbent and the certified/exact gap to optimum. Report paired differences in score units with direction. Separate classical, variational, certificate, and cross-check wall times. Timings measure this CPU implementation.
- Verify the factor score against dense and official calculations, the conditional probabilities against independent enumeration and a small gate-model circuit, penalties against feasible score ranges, returned holdings against the original rules, and certificates against exhaustive optima. Retain per-instance data and reproducible commands.
- Report circuit data width separately from preparation workspace and from classical simulator storage. Analyze a simple independent readout-error model as a hypothetical sensitivity calculation. Noise-free measurements remain the only scored experiments.

## 6. Presentation

Use a five-minute story: what selecting K assets means, how a quantum state represents a basket, what independent checks establish, how much objective quality and runtime change with size, and what can be concluded. Quantify the compact circuit's contribution relative to uniform sampling and the scalable solver's performance relative to both controls. Include any negative finding. Put detailed derivations, raw evidence, method tradeoffs and sources alongside the implementation.
