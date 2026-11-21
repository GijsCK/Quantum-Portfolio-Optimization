# Methods and optimality certificates

The experiment asks how close an ideal variational measurement can come to the best admissible basket. It uses three independent calculations. The main simulator draws baskets from a fixed-count quantum state. A classical optimizer constructs its own basket and bounds how much any solver could improve on it. Small gate-model circuits check the measurement law and test how much interference changes outcome probabilities.

## 1. What a portfolio earns and risks

An asset has an expected annual return μᵢ, a price pᵢ, a vector of factor loadings Bᵢ and an asset-specific variance dᵢ. Exactly K holdings receive equal weights. If xᵢ is one when asset i is held and zero otherwise, the score to minimize is

    z(x) = Bᵀx
    a    = 0.5 / K²
    cᵢ   = a dᵢ − 0.5 μᵢ / K
    f(x) = a ‖z(x)‖² + cᵀx.

Two companies exposed to the same market factor contribute in the same direction before the exposures are squared. The separate variances add because the generator puts them on the diagonal of its covariance matrix. The identity xᵢ² = xᵢ makes this last term linear for binary decisions. Factor arithmetic gives precisely the official score with q = 0.5 and avoids forming an n-by-n covariance matrix.

An admissible basket spends at most the official budget plus its numerical tolerance of 10⁻⁹ and respects every sector cap. The official evaluator checks each returned basket. A difference of 0.0001 in f is one *objective bp*. Its units describe the risk-adjusted score. Annual return and variance can also be inspected separately in the raw records. The statistical population here consists of portfolios from the supplied synthetic generator.

## 2. A state that always selects K holdings

Associate data qubit i with asset i. The main solver gives the following nonzero amplitudes to subsets S with |S| = K:

    ψθ(S) = exp(½ Σᵢ∈S θᵢ) / √Z
    Z     = Σ over K-subsets T of exp(Σᵢ∈T θᵢ).

This is a product of single-asset amplitudes projected onto the fixed-count subspace and normalized. Equal θ gives a uniform Dicke state. Finite unequal θ preserves support on every K-subset and changes its probability. Increasing one asset's θ by log 2 doubles the relative weight of any basket containing it against any basket omitting it. Adding the same constant to all θ leaves every probability unchanged.

The measurement distribution is also a classical conditional Bernoulli distribution. This equivalence makes an exact noiseless simulation possible and limits the interpretation of its measured speed. A separate portfolio study uses Dicke sectors for prescribed asset classes. Its circuit, constraints and optimizer differ from this experiment. [Chen, Dempster & Liu](https://doi.org/10.1093/biomet/81.3.457) · [Bärtschi & Eidenbenz](https://arxiv.org/abs/2207.09998) · [Scursulim et al.](https://www.nature.com/articles/s41598-026-36333-4)

**Sampling.** Let T(i,r) be the total weight of all ways to take r more holdings starting at asset i. Taking or skipping one asset gives

    T(i,r) = T(i+1,r) + exp(θᵢ) T(i+1,r−1)
    P(take i | r still needed) = exp(θᵢ) T(i+1,r−1) / T(i,r).

The boundary values are T(n,0) = 1 and T(n,r) = 0 for r > 0. The simulator stores logarithms and sums the two branches with log-add-exp. It draws one conditional decision for each asset and forces the unique completion when the number of remaining assets equals the number still needed. Multiplying the path probabilities cancels the intermediate T factors and yields |ψθ(S)|². Every measured basket already has K holdings. See [variational.py](variational.py).

**A diagonal Hamiltonian for the remaining limits.** Each fixed-K basis basket has energy f(x) when feasible. Define b as the positive amount of spending above the tolerated budget and h as the total number of holdings above sector caps. Put Δ = ½(maxᵢ Σᵢᵢ + max μ − min μ) + 10⁻⁹, with Σᵢᵢ = ‖Bᵢ‖² + dᵢ. The implemented energy is

    E(x) = f(x) + Δ [1{b+h>0} + b/budget + h/K].

Why the first penalty is sufficient follows from convexity. An equal-weight basket has nonnegative weights summing to one, so its nonnegative variance is at most the largest individual variance. Its return lies between the smallest and largest asset returns. Δ therefore exceeds the entire possible fixed-K score span. Every admissible basket ranks ahead of every inadmissible one. The other terms give direction when the first measurements contain violations. This diagonal operator uses exact floating-price comparisons and includes a discontinuous barrier. A two-body Ising implementation would need additional machinery to represent that barrier. The simulation evaluates it on each measured basket.

## 3. How the state learns

The initial guess uses the universe's mean factor exposure e. Asset i receives desirability uᵢ = ½μᵢ − ½(2 Bᵢ·e + dᵢ/K). This is the negative first-order change in the binary-extended score, multiplied by K. Thus a high-return asset looks less attractive when it adds exposure already present in the basket. The initial standard deviation of u sets a common scale throughout the run, with a 10⁻¹² floor.

At each round the solver centers u, divides by that scale, multiplies by the scheduled concentration and adds the current feedback Fᵢ. Call this preference vᵢ. It then sets

    θᵢ = vᵢ − α − λ pᵢ/(budget/K) − ν_sector(i).

The unrestricted α centers the expected count in a preliminary independent-Bernoulli model. The nonnegative λ and ν discourage expensive assets and crowded sectors. They solve a small convex fitting problem with L-BFGS-B. Its objective is Σ log(1+exp(θᵢ))/K + α + λ tᵦ + tₛ Σν. The targets are tᵦ = 1 − min(0.05, 0.5/√K) and tₛ = [cap − min(0.3, 0.04 cap)]/K. A modest buffer leaves room for sampling variability. These approximate means guide the state. Actual spending and sector counts are checked after each fixed-K measurement. Each trace records whether the field fit reported convergence.

**Round budget.** The nine concentration values are 1.5, 2.5, 4, 7, 12, 20, 32, 50 and 72. The first eight rounds each draw 128 shots, and the final round draws 256. The best feasible basket observed in all 1,280 raw measurements is the returned solution. Feasible frequency and the final batch are distinct diagnostics. A run with zero valid shots reports a missing objective.

**Updates.** Denote all-shot inclusion frequency by mᵢ and inclusion frequency in the lowest-energy eighth by mᵢ*. The trained arm updates

    e ← 0.5 e + 0.5 Bᵀm / K
    Fᵢ ← clip(0.65 Fᵢ + 0.8 (mᵢ* − mᵢ) / [0.10 + mᵢ(1−mᵢ)], −4, 4).

The derivative of log Pθ(x) with respect to θᵢ equals xᵢ − Eθ[xᵢ]. The difference of measured frequencies therefore estimates a likelihood gradient toward the elite baskets. The denominator dampens frequent assets' updates and limits the largest rescaling for rare assets. The changing exposure, prescribed concentration and constrained field fit also move the state. The program records its lower-eighth mean energy. Its parameter update fits elite holdings and includes these other changes, so a claim of exact CVaR minimization would overstate what was computed. [Barkoutsos et al.](https://arxiv.org/abs/1907.04769)

**Paired controls.** The schedule-only arm updates e and the fields through the same concentration schedule with F fixed at zero. The frozen arm draws all batches from the initial informed state. Each arm starts from the same seed for its instance and uses the same number of uniforms and shot opportunities. This makes the schedule-only comparison informative about the marginal contribution of elite feedback. The frozen comparison measures the joint effect of the changing fields, concentration and feedback. Equal shot counts can incur different wall times because frozen parameters reuse their lookup table.

## 4. The independent classical benchmark

The benchmark starts from the cheapest K holdings consistent with the sector caps. At a trial exposure z it assigns a marginal price cᵢ + 2a Bᵢ·z to each asset. For a fixed nonnegative budget multiplier λ it adds λpᵢ, retains the cap cheapest assets within each sector, then takes the cheapest K survivors. An asset ranked below the sector cap has enough cheaper replacements in that sector. This gives an exact linear minimizer for the prescribed multiplier.

When this choice is too expensive, the optimizer maintains one affordable and one expensive choice. Their costs are lines as functions of λ. At their intersection it calls the selection rule again. A cheaper line tightens the bracket. A convex combination of bracket endpoints spends exactly the available budget. This fractional combination supplies a direction for a Frank–Wolfe update of exposure and linear cost. A one-dimensional quadratic gives its optimal step length. The algorithm keeps the best affordable integer selection seen in up to 36 rounds. [Jaggi](https://proceedings.mlr.press/v28/jaggi13.html)

It then tries up to 100 improving swaps. For replacing held i with unheld j, the exact change is (cⱼ + 2aBⱼ·z) − (cᵢ + 2aBᵢ·z) + a‖Bⱼ−Bᵢ‖². A swap must preserve the original budget and sector rules. Through 1,500 assets each step examines all K(n−K) pairs. On the 10,000-asset extension it takes the union of candidates ranked by plain and price-adjusted marginal costs, at most 192 selling and 256 buying candidates. This trades exhaustive neighborhood search for a bounded working set. No candidate from this benchmark enters the variational algorithm. See [classical.py](classical.py).

### How much better could any portfolio be?

The basic identity ‖Bᵀx−z‖² ≥ 0 implies ‖Bᵀx‖² ≥ 2z·Bᵀx − ‖z‖². Take any nonnegative price multiplier λ, nonnegative sector multipliers ν and any count threshold τ. Define

    rᵢ = cᵢ + 2a Bᵢ·z + λpᵢ + ν_sector(i) − τ
    C  = −a‖z‖² − λ(budget + 10⁻⁹) − cap Σν + τK
    L  = C + Σ min(0,rᵢ) − 10⁻¹¹.

Adding λ(spend−budget−10⁻⁹) and ν(count−cap) can only decrease a feasible portfolio's cost. Subtracting τ(sum x−K) leaves it unchanged. Hence every feasible f(x) is at least C + Σrᵢxᵢ, whose unconstrained binary minimum gives L with a conservative roundoff margin. The best tangent found during the iterations is recorded with its exposure and multipliers. The argument requires no claim about the convergence of the relaxation.

For a positive rᵢ the minimizing binary choice is zero. For negative rᵢ it is one. Reversing this choice adds |rᵢ| to the lower bound for that basket. Given an affordable incumbent score U, any bit with |rᵢ| > U−L+10⁻¹¹ is fixed in every optimum. The verifier enumerates all cardinality-compatible completions when their count is at most 150,000. At the official 10-, 20- and 50-asset sizes it instead checks every K-subset, up to 2,118,760 at n = 50. The exhaustive check scores feasible portfolios in batches, rechecks near-budget cases against the official evaluator, and returns an exact optimum to the numerical precision of the supplied inputs. Certificates and scores use double precision and explicit guard margins. The exhaustive small-case tests compare pointwise bounds against the organizer's evaluator.

The original audit supplied exact references for 32 of the 40 instances. For eight others, the feasible classical incumbent U and the tangent bound L bracketed the optimum. A separate, post-study enumeration closed these intervals while preserving the original solver outputs and timing measurements. [Original certificate tests](tests/test_classical.py)

### Exact references for the eight remaining instances

Let bᵢ = 1 when rᵢ < 0, and zero otherwise. This preferred basket minimizes the tangent's unconstrained linear expression. A basket x pays the sum of |rᵢ| over every bit that differs from b. The full identity for any feasible x is

    f(x) = L + guard + Σ_{xᵢ≠bᵢ}|rᵢ|
           + λ (budget + 10⁻⁹ − spend)
           + Σₛ νₛ (cap − countₛ) + a ‖Bᵀx − z‖².

Every term after L is nonnegative. Any basket that could equal or improve the incumbent U must therefore have *total* reversal cost at most U−L. The implementation adds a 10⁻¹¹ rounding margin. Removing five individually cheap holdings can still exceed this total, which explains why summing their costs prunes far more cases than fixing expensive bits one at a time.

The verifier lists affordable additions and removals separately. For each removal set it retains the additions that restore exactly K holdings and fit the remaining reversal allowance. It screens these pairs for budget and sector feasibility, evaluates their exact factor scores, then checks every possible new incumbent against the organizer's functions. The budget and score screens both include margins for floating-point arithmetic. Differential tests include preferred baskets that are too large or too small, zero reversal costs, tied cutoffs and an independent exhaustive search. [Implementation](exact_supplement.py) · [Tests](tests/test_exact_supplement.py)

Across the eight instances, 4,520,477 cardinality-compatible pairs passed the reversal cutoff. The 1,000-asset instance with seed 4 accounted for 4,384,787 pairs and took 2.17 s. The eight checks together took 2.52 s on the recorded host. The supplement improved the original classical incumbents by between 0.00067 and 1.50095 objective bp. Rechecking all 40 exact portfolios with the official rules found every one feasible. Substituting each into the full identity above gave a maximum absolute residual of 2.09 × 10⁻¹⁷ or less. Every reported regret can now use the exact discrete optimum at the precision of the supplied inputs. [Post-study results](results/exact_supplement.json)

## 5. Independent circuits and interference

At n = 10 the saved trained angles specify the entire fixed-K probability law. The validation circuit prepares each qubit with a prefix-controlled Rᵧ having sin²(angle/2) equal to the corresponding conditional probability. Qiskit's ideal statevector is compared with direct weights normalized by enumerating all 2¹⁰ bit strings. The audit records probability discrepancies, state fidelity, off-cardinality mass and total variation for 32,768 independent simulated shots. Small statevector agreement tests bit order, rotation angles and normalization. A finite-sample concentration bound supplies a separate check on the sampler. [Qiskit Statevector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)

The scalable circuit builder controls each rotation by how many holdings remain. It initializes a b = ⌈log₂(K+1)⌉-qubit counter to K. For asset i, a multiplexed Rᵧ uses the conditional angle for each possible remaining count. A selected data qubit then enables a reversible binary decrement. A borrow propagates when all already-updated lower bits are one. When the number of open places equals the number of remaining assets, all must be selected. Every populated branch ends with the counter at zero. Each asset needs one multiplexed rotation with 2ᵇ entries and b controlled-X operations with one through b controls. This replaces the prefix audit's exponentially many possible control settings with at most 2(K+1) angle entries per asset. [Qiskit uniformly controlled rotations](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UCRYGate)

For one generic 10-asset, K = 2 state the complete preparation used 10 data and two counter wires. Compilation to an all-to-all rz/sx/x/cx basis gave 110 CX gates and depth 294. The largest portfolio-probability discrepancy against direct enumeration was 3.33 × 10⁻¹⁶ and the final probability of a nonzero counter was 5.16 × 10⁻³³ or less. This generic-state resource measurement is separate from the five trained-vector prefix checks. The latter agreed with enumeration to 2.54 × 10⁻¹⁴ or better. [Counter circuit and counts](results/counter_circuit.json) · [Trained-state checks](results/weighted_circuit_audit.json)

The second circuit tests a different mechanism at n = 10 and 20. A compact address names one entry in an enumerated dictionary of portfolios. The hard dictionary contains feasible baskets only. The penalty dictionary contains all fixed-K baskets and assigns each infeasible one a barrier of 1% above the theoretical score span (with an absolute 10⁻⁹ minimum margin), plus a term for excess spending and sector use. Unused addresses receive zero initial amplitude. The ideal mixer keeps them empty.

Both variants start uniformly and apply two rounds of a diagonal cost phase and a selective phase on the uniform state. This Grover mixer changes the amplitude of an address through interference with the mean amplitude. Each dictionary's energies are scaled to [0,1]. The search uses 128 deterministic scrambled Sobol starts and up to 250 evaluations at each of six Powell refinements. Its training loss is exact mean energy. Uniform and trained distributions are compared using exact single-shot probability of an optimum, conditional mean regret and a separate 256-shot draw. The probability of at least one optimum in 16 independent shots is 1−(1−p*)¹⁶. [Bärtschi & Eidenbenz](https://arxiv.org/abs/2006.00354)

The statevector check independently rebuilds the phase and mixer gates in Qiskit. Compilation targets all-to-all connectivity with rz, sx, x and cx gates. Reported depths and CX counts describe those logical compilations. Building either dictionary evaluates every candidate in advance, making the optimum accessible during preparation and normalization. That is an appropriate reference for investigating interference on known energies. It would be an expensive way to search a new large universe. See [quantum_audit.py](quantum_audit.py).

## 6. Resources, timing and interpretation

| Computation | Principal costs and what the recorded time includes |
| --- | --- |
| Weighted-state simulator | For R shots, n assets and m factors, a fresh suffix table needs O(nK) time and space. Drawing takes O(nR) time and scoring takes O(nRm), plus field fitting and sector accounting. The table at n = 10,000 and K = 1,000 occupies 80,088,008 bytes alone. Arrays for samples, scoring and table replacement increase peak memory. Per-arm time includes initialization, sampling, updating, feasibility filtering and the returned basket's official score check. |
| Data width | One logical data qubit per asset plus ⌈log₂(K+1)⌉ counter wires in the implemented reversible preparation. This gives 1,007 logical wires for 1,000 assets and 10,010 for 10,000 assets. Measurements at those sizes used the exact classical conditional sampler. The 12-wire resource audit includes angle loading and counter updates. Physical routing, fault tolerance and large-circuit execution times fall outside these counts. |
| Classical construction | The factor model takes O(nm) space. Each capped linear selection uses per-sector and global partial selection. Constructing a tangent and finding a certificate requires the factor multiplication and a sort of survivors. Exchange matrices grow as K(n−K) on the official sizes. The recorded solver time includes bounds calculated while producing the incumbent. |
| Independent audit | Bound time records the additional certificate and fixing calculation. Exact time records the enumeration loop. Dictionary construction, QAOA training and circuit cross-checks have separate fields. The driver records timings on one host and limits numeric-library thread pools to one for the main sweep. Generating an input, serialization, imports and interpreter startup sit outside per-solver times. |

**What a noisy measurement could change.** Under an illustrative independent readout flip probability ε, the probability of receiving the identical basket is (1−ε)ⁿ. Cardinality can also survive equal numbers of false additions and deletions. Its exact probability is the sum over j from 0 to min(K,n−K) of C(K,j)C(n−K,j) ε²ʲ (1−ε)ⁿ⁻²ʲ. Budget and sector feasibility depend on *which* assets flipped. Neither formula predicts gate noise or validates a physical device. The scored experiments use ideal simulation.

**Statistical scope.** The five seeds at a size represent five independently generated universes. Shared seeds across methods support paired differences. The 1,280 shots within one run share an adaptive training trajectory, so binomial intervals treating all those shots as independent repetitions of the whole optimizer would be misleading. The main sweep uses one fixed sampling stream per instance and mode. Descriptive sample spread over five instances captures variation of the observed runs. The small-circuit probabilities computed directly from statevectors have no shot error. The design measures solution quality and classical CPU cost on this generator and does not establish an investment outcome or a quantum speed advantage.
