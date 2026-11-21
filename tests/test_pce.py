import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from portfolio_generator import generate
from qportfolio.pce import Ansatz, Encoding, RelaxedLoss, qubits_needed


@pytest.mark.parametrize("N,k,n,layers", [(5, 2, 30, 3), (6, 3, 60, 2)])
def test_state_and_correlators_match_qiskit(N, k, n, layers):
    ans, enc = Ansatz(N, layers), Encoding(n, N, k)
    theta = np.random.default_rng(1).uniform(0, 2 * np.pi, ans.n_params)
    psi = ans.state(theta)
    ref = Statevector(ans.to_qiskit(theta))
    assert np.allclose(psi, ref.data)
    c = enc.correlators(ans, psi)
    for i, (basis, subset) in enumerate(enc.assign):
        op = SparsePauliOp.from_sparse_list([(basis * k, list(subset), 1.0)], num_qubits=N)
        assert c[i] == pytest.approx(ref.expectation_value(op).real, abs=1e-12)


@pytest.mark.parametrize("entangle", [True, False])
def test_adjoint_gradient_matches_finite_differences(entangle):
    inst = generate(50, 0)
    N, k = qubits_needed(50, 2), 2
    ans, enc, loss = Ansatz(N, 2, entangle), Encoding(50, N, k), RelaxedLoss(inst)
    alpha = 1.5 * N

    def L(theta):
        psi = ans.state(theta)
        c = enc.correlators(ans, psi)
        t = np.tanh(alpha * c)
        val, gx = loss((1 + t) / 2)
        g = gx / 2 * alpha * (1 - t * t)
        return val, ans.gradient(theta, psi, enc.apply_observable(ans, psi, g))

    theta = np.random.default_rng(2).uniform(0, 2 * np.pi, ans.n_params)
    _, grad = L(theta)
    eps = 1e-6
    for p in np.random.default_rng(3).choice(ans.n_params, 12, replace=False):
        e = np.zeros_like(theta)
        e[p] = eps
        fd = (L(theta + e)[0] - L(theta - e)[0]) / (2 * eps)
        assert grad[p] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_relaxed_loss_gradient_and_feasible_hinges():
    inst = generate(100, 1)
    loss = RelaxedLoss(inst)
    x = np.random.default_rng(0).random(100) * 0.3
    _, g = loss(x)
    eps = 1e-7
    for i in range(0, 100, 17):
        e = np.zeros(100)
        e[i] = eps
        assert g[i] == pytest.approx((loss(x + e)[0] - loss(x - e)[0]) / (2 * eps), rel=1e-5)
