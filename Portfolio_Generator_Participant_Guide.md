# Portfolio Optimization Data Generator: Participant Guide

Amsterdam Qiskit Fall Fest Hackathon, November 21st 2026. Topic 1: Portfolio Optimization with Quantum Computing. Facilitator: Ilia Fazeli, WIQD.

## 1. Setup

Requires Python 3.10 or newer and NumPy. Put `portfolio_generator.py` in your working folder.

```python
from portfolio_generator import generate

inst = generate(n=100, seed=0)   # same (n, seed) gives identical data on every machine
```

Qiskit is optional. The helper `inst.to_quadratic_program()` needs `qiskit-optimization`, and it is only practical for small n because it builds a dense matrix.

## 2. What an instance contains

| Field | Shape | Meaning |
|---|---|---|
| `mu` | (n,) | Expected return of each asset (annualised) |
| `B`, `d` | (n, m), (n,) | Risk in factor form: covariance = B Bᵀ + diag(d) |
| `sector` | (n,) | Integer sector label of each asset |
| `price` | (n,) | Cost of including each asset |
| `K` | int | Number of assets to select (exactly K) |
| `budget` | float | Maximum total price of the selected assets |
| `sector_cap` | int | Maximum number of selected assets per sector |
| `n_sectors` | int | Number of sectors |

Methods you can use:

- `inst.Sigma()` returns the dense n × n covariance matrix. It is built only on request (about 8 MB at n = 1000).
- `inst.risk(x)` returns xᵀΣx using the factor form, without building Σ.
- `inst.sigma_matvec(x)` returns Σx in O(n·m) time.
- `inst.objective(x, q=0.5)` returns the official score of a portfolio.
- `inst.is_feasible(x)` checks all constraints.

The covariance has a market factor, sector factors and three style factors, plus asset-specific noise. Assets in the same sector are therefore more correlated, and the average pairwise correlation is realistic (around 0.25).

## 3. The problem you must solve

A portfolio is a 0/1 vector x of length n, where x[i] = 1 means asset i is selected. Weights are equal: w = x / K.

**Objective (minimise):**

```
f(x) = q · wᵀ Σ w − (1 − q) · μᵀ w,    w = x / K,    q = 0.5
```

**Constraints (a portfolio is valid only if all hold):**

1. Cardinality: exactly K assets selected.
2. Budget: the total price of selected assets is at most `budget`.
3. Sector cap: at most `sector_cap` selected assets in any one sector.

`is_feasible(x)` is the single authoritative definition of these rules. How you enforce them (penalty terms, slack variables, constraint-preserving mixers, or classical repair) is part of your design and is explicitly something to experiment with.

## 4. Official sweep

- Universe sizes n: 10, 20, 50, 100, 200, 500, 1000
- Seeds: 0 to 4 for each size (report the mean and spread over seeds)
- Parameters: default `GeneratorParams()`, with q = 0.5
- The generator supports larger n, but the hackathon sweep stops at 1000.

## 5. Rules

- Use the generator unmodified, with the specified (n, seed) pairs. Do not change the parameters for scored results.
- Noise-free simulators only. No QPU run is expected, and no error mitigation.
- Variational methods are expected (QAOA, CVaR-VQE, VQEC, Pauli correlation encoding and similar).
- Include a classical benchmark of your own. Your solver is judged against it on the same instances.
- Report the encoding, the number of qubits used, and how each constraint is enforced.

## 6. Notes and caveats

- Data is synthetic. Parameters are plausible but not calibrated to real market data.
- Instances of different sizes are independent, so a 10-asset universe is not a subset of a 100-asset one.
- `QuadraticProgramToQubo().convert(...)` fails on the QuadraticProgram returned by 'to_quadrtic_program' because the budget constraint has float prices. Either build your penalty QUBO by hand, or round the prices to integers for the conversion and check the final portfolio with `inst.is_feasible`, which is the only authoritative rule.
- The Qiskit helper `to_quadratic_program()` has not been extensively tested against a Qiskit installation by the organizers. Verify it in your own environment, or build your QUBO by hand.
- Ask a facilitator if anything about the problem definition is unclear.
