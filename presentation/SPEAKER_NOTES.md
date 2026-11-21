# Five-minute presentation

Show slides 1–7. The final two slides support questions about the full score table and the proof of optimality.

## 0:00–0:25 · Quantum portfolio selection

Suppose a fund can hold only a fixed number of assets. Buying them must fit a budget, and too many holdings in one sector would break the rules. We ask how close a simulated quantum search can come to the best legal basket, and how we can tell when checking every possible basket becomes impractical.

## 0:25–1:05 · Which two assets should the fund hold?

Every dot is a different pair from the ten-asset instance with seed zero. Its horizontal position tells you what the pair costs. Its height shows how far its score is from the best legal pair. We give half the weight to portfolio variance and half to expected return. Both holdings carry equal portfolio weight, even when their purchase prices differ. The orange point is the best legal choice. Assets one and eight cost about 81 units and earn the lowest score among all 15 affordable pairs.

## 1:05–1:55 · A quantum state with exactly K holdings

Our main encoding assigns one logical data qubit to each asset and gives probability only to bitstrings with the required number of holdings. In this three-asset example, the weights are one, two and four. A and B have product two, while B and C have product eight. The second basket is four times as likely as the first. On the scored instances, the solver starts with market inputs and learns from the best eighth of each sample batch. An over-budget or over-concentrated basket receives an energy penalty. We return the best legal measured basket, with no repair. The structure lets us draw exact ideal measurements on a conventional computer.

## 1:55–2:45 · Interference raises the chance of an exact hit

We also tested a genuinely interference-based approach on universes small enough to enumerate. Each legal basket gets an address. Two QAOA layers change the phases and mix the amplitudes. At ten assets, the chance of drawing the exact optimum rises from about six percent to forty percent. At twenty assets, it rises from about one percent to nearly nine percent. Keeping illegal baskets in the address list and assigning them penalties performed worse. Independent Qiskit circuits reproduced the probabilities to within 2.3 × 10⁻¹⁴ across the tested encodings and seeds. Building the list already inspects every candidate, so this experiment isolates the circuit’s effect.

## 2:45–3:43 · All 35 portfolios finished within one objective bp

We checked the best measured holdings against the original rules in every official instance. The classical baseline accounts for factor risk, prices and sector limits and improves its choices through feasible exchanges. To judge both methods, we also compute a floor that every legal portfolio must stay above. Subtracting the floor from a returned score gives the largest possible error. The bound cuts down how many combinations we must check. We then enumerated the remaining possibilities and certified every optimum. The chart shows the true score gap averaged over five markets per size. Across all thirty-five, the largest variational gap was under one ten-thousandth of the score. The search matched the optimum in twenty-nine cases. At a thousand assets the classical baseline had the smaller mean gap.

## 3:43–4:25 · Removing elite feedback changed little

What produced these scores? We ran two controls with identical shot budgets and matched random draws. The first kept the informed initial state frozen. The second still concentrated the weights and updated market-factor exposures, but it did not learn from the elite draws. At a thousand assets the frozen state lost seventy-one objective basis points. The schedule-only control actually did slightly better than full training, by eight hundredths of a point. These five markets give little evidence for an additional gain from elite feedback.

## 4:25–5:00 · The 10,000-asset stress test

We extended the experiment to ten thousand assets on five additional generated markets, each with a thousand holdings. Every returned basket was legal. The largest observed gap from the exact optimum was about zero point three seven objective basis points. Mean simulator time was about eleven seconds per instance. The classical solver took about seven hundredths of a second and returned slightly better scores on average. These are synthetic inputs and ideal measurements on a CPU. We measured the gap to a proven optimum. Because this state has an efficient classical sampler, these results provide no evidence of a quantum speedup.

## Questions from judges

**What counts as a win?** We minimize the official equal-weight, risk-adjusted score. A hit matches the proven optimum within the study’s numerical tolerance. One objective basis point is 0.0001 of that score. The units describe the optimization objective and cannot be read as realized investment returns.

**Which constraints does the state enforce?** Every weighted-Dicke measurement contains exactly K assets. The energy penalizes budget and sector violations strongly enough to put every illegal holding above all legal scores. The solver only returns a measured legal holding. At 1,000 assets, 98.4% of the final 256-shot batches were legal on average.

**Where is the quantum contribution?** The small QAOA experiment includes coherent cost phases and a Grover mixer. It increased the one-shot probability of the true optimum against the same uniform feasible-address control. The full-size positive-amplitude state admits an efficient exact classical sampler. Its close scores quantify portfolio quality and give no evidence of a hardware speedup.

**How many qubits does it use?** The scalable state uses one logical data qubit per candidate asset. A coherent implementation may use an additional reversible counting register. The experiment simulated its measurement distribution directly. The compact QAOA circuit used 4–7 address qubits on the five 10-asset and five 20-asset feasible dictionaries. Enumerating and loading those dictionaries costs classical work.

**What provides the exact references?** Small exact searches enumerate all fixed-size combinations. At larger sizes, a tangent lower bound and dual prices certify that some decisions cannot change in an optimum. Exhausting the remaining candidates, including a supplementary exchange search on eight difficult cases, established the optimum in all 40 instances. The verification cost is reported separately from each solver’s time.

**Did all probabilistic draws hit the optimum?** The 29/35 figure uses the best feasible raw draw across 1,280 measurements per official instance. Raw feasibility and one-shot QAOA probability answer different questions. The final-batch feasibility percentage appears on slide 8.

**Did the comparison favor the variational solver?** Its informed initialization and concentration schedule use the data, and the controls retain them where specified. The classical method independently constructs and improves a legal solution. The methods have different computational budgets. The report gives measured CPU wall times and per-method quality on identical instances.

**What would noise change?** A single readout-bit flip changes the number of selected assets in the direct encoding. Multiple flips can cancel in the holding count while still violating the budget or sector cap. The topic guide specifies ideal simulation. The published numbers measure synthetic instances under those conditions.

## Evidence for follow-up

- Complete run summaries and sample spreads are in [the size-level table](../results/by_size.csv).
- Holdings, output validity and full training traces are in [the variational measurements](../results/variational.json).
- Exact reference records are in [the classical runs](../results/classical.json) and [the supplementary exhaustive search](../results/exact_supplement.json).
- The two encodings, circuit checks and per-instance probabilities are in [the small QAOA study](../results/quantum_audit.json).
