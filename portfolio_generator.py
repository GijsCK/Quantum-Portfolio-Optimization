from dataclasses import dataclass, field
import numpy as np

@dataclass
class GeneratorParams:
    market_vol: float = 0.16        
    sector_vol: float = 0.08        
    style_vol: float = 0.04         
    n_styles: int = 3               
    idio_vol_median: float = 0.25   # median idiosyncratic vol
    idio_vol_sigma: float = 0.35    # lognormal dispersion of idio vol
    market_premium: float = 0.05
    style_premium: tuple = (0.02, 0.015, 0.01)
    alpha_vol: float = 0.02         # asset-specific alpha noise
    rf: float = 0.0
    corr_scale: float = 1.0         
    card_frac: float = 0.10         
    budget_slack: float = 1.0       
    sector_cap_mult: float = 2.0    
    price_median: float = 50.0
    price_sigma: float = 0.8

@dataclass
class PortfolioInstance:
    n: int
    seed: int
    params: GeneratorParams
    mu: np.ndarray                  # (n,) expected returns
    B: np.ndarray                   # (n, m) factor loadings, scaled so Sigma = B B^T + diag(d)
    d: np.ndarray                   # (n,) idiosyncratic variances
    sector: np.ndarray             
    n_sectors: int
    price: np.ndarray              
    budget: float
    K: int                          # cardinality (exactly K assets)
    sector_cap: int                 
    tickers: list = field(default_factory=list)

    def Sigma(self, dtype=np.float64) -> np.ndarray:
        # Dense covariance matrix 
        B = self.B.astype(dtype)
        S = B @ B.T
        S[np.diag_indices(self.n)] += self.d.astype(dtype)
        return S

    def sigma_matvec(self, x: np.ndarray) -> np.ndarray:
        return self.B @ (self.B.T @ x) + self.d * x

    def risk(self, x: np.ndarray) -> float:
        y = self.B.T @ x
        return float(y @ y + np.dot(self.d * x, x))

    def objective(self, x: np.ndarray, q: float = 0.5) -> float:
        w = np.asarray(x, float) / self.K
        return q * self.risk(w) - (1 - q) * float(self.mu @ w)

    def is_feasible(self, x: np.ndarray) -> bool:
        x = np.asarray(x).astype(int)
        if x.sum() != self.K or float(self.price @ x) > self.budget + 1e-9:
            return False
        counts = np.bincount(self.sector[x == 1], minlength=self.n_sectors)
        return bool((counts <= self.sector_cap).all())

    def to_quadratic_program(self, q: float = 0.5):
        """qiskit_optimization.QuadraticProgram with all constraints (n <= ~2000 advisable)."""
        from qiskit_optimization import QuadraticProgram
        qp = QuadraticProgram("portfolio")
        for i in range(self.n):
            qp.binary_var(f"x{i}")
        S = self.Sigma() / self.K**2
        lin = -(1 - q) * self.mu / self.K
        qp.minimize(linear=lin, quadratic=q * S)
        names = [f"x{i}" for i in range(self.n)]
        qp.linear_constraint(dict.fromkeys(names, 1), "==", self.K, "cardinality")
        qp.linear_constraint(dict(zip(names, self.price)), "<=", self.budget, "budget")
        for s in range(self.n_sectors):
            idx = np.where(self.sector == s)[0]
            if len(idx) > self.sector_cap:
                qp.linear_constraint({f"x{i}": 1 for i in idx}, "<=", self.sector_cap, f"sector{s}")
        return qp


def generate(n: int, seed: int = 0, params: GeneratorParams | None = None) -> PortfolioInstance:
    p = params or GeneratorParams()
    rng = np.random.default_rng(np.random.SeedSequence([seed, n]))

    n_sectors = int(np.clip(round(np.sqrt(n) / 2), 2, 11))   # 2 sectors at n=10, 11 (GICS-like) at n>=~500
    sector = rng.choice(n_sectors, size=n, p=_sector_weights(n_sectors, rng))
    # guarantee every sector is non-empty (matters at small n)
    sector[:n_sectors] = rng.permutation(n_sectors)

    S = p.n_styles
    m = 1 + n_sectors + S
    c = np.sqrt(p.corr_scale)

    # loadings (unit-variance factors), then scale columns by factor vol
    beta_m = np.clip(rng.normal(1.0, 0.25, n), 0.2, None)
    sec_load = np.zeros((n, n_sectors))
    sec_load[np.arange(n), sector] = np.clip(rng.normal(1.0, 0.3, n), 0.2, None)
    style_load = rng.normal(0, 1, (n, S))
    B = np.hstack([beta_m[:, None] * p.market_vol * c,
                   sec_load * p.sector_vol * c,
                   style_load * p.style_vol * c])

    idio = p.idio_vol_median * np.exp(rng.normal(0, p.idio_vol_sigma, n))
    d = idio**2

    prem = np.asarray(p.style_premium[:S] if S <= len(p.style_premium)
                      else list(p.style_premium) + [0.0] * (S - len(p.style_premium)))
    mu = (p.rf + beta_m * p.market_premium + style_load @ prem
          + rng.normal(0, p.alpha_vol, n)
          + 0.02 * (idio - idio.mean()))  # weak risk-return tilt
    price = p.price_median * np.exp(rng.normal(0, p.price_sigma, n))

    K = int(min(n, max(2, round(p.card_frac * n))))
    budget = float(p.budget_slack * K * np.median(price))
    sector_cap = int(np.ceil(p.sector_cap_mult * K / n_sectors))

    inst = PortfolioInstance(n, seed, p, mu, B, d, sector, n_sectors, price,
                             budget, K, max(1, sector_cap), [f"A{i:05d}" for i in range(n)])
    return inst


def _sector_weights(k, rng):
    w = rng.dirichlet(np.full(k, 4.0))
    return w / w.sum()


def sweep(sizes=(10, 20, 50, 100, 200, 500, 1000), seed=0, params=None):
    for n in sizes:
        yield generate(n, seed, params)


if __name__ == "__main__":
    import time
    for inst in sweep():
        t = time.time()
        x = np.zeros(inst.n, int); x[:inst.K] = 1
        print(f"n={inst.n:6d} K={inst.K:4d} sectors={inst.n_sectors:2d} cap={inst.sector_cap:3d} "
              f"mem={(inst.B.nbytes+inst.d.nbytes)/1e6:.1f}MB risk(equal first K)={inst.risk(x/inst.K):.4f} "
              f"[{time.time()-t:.2f}s]")