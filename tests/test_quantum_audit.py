"""Circuit regressions for padding, phase sign, and asset bit order."""
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from portfolio_generator import generate
from quantum_audit import (
    amplitudes, counter_audit, portfolio_dictionary, qiskit_grover, weighted_dicke_audit,
)


class QuantumCircuitTests(unittest.TestCase):
    def test_grover_mixer_on_non_power_of_two_dictionary(self):
        costs = np.array([0.0, 0.11, 0.19, 0.45, 0.68, 0.88, 1.0])
        angles = [1.7, -0.9, -2.1, 0.7]
        logical = np.pad(amplitudes(angles, costs), (0, 1))
        circuit = Statevector.from_instruction(qiskit_grover(costs, angles)).data
        self.assertAlmostEqual(abs(np.vdot(logical, circuit)) ** 2, 1.0, places=11)
        self.assertLess(abs(circuit[-1]) ** 2, 1e-20)

    def test_penalty_separates_every_infeasible_basis_state(self):
        _, scores, valid, penalties = portfolio_dictionary(generate(20, 77))
        self.assertTrue(np.any(valid) and np.any(~valid))
        self.assertTrue(np.all(penalties[valid] == 0))
        self.assertGreater(np.min((scores + penalties)[~valid]), scores[valid].max())

    def test_moderately_spread_weighted_state_and_shift_invariance(self):
        # Highly concentrated final states are easy to sample. This asymmetric,
        # broadly supported case catches reversed asset/wire conventions as well.
        theta = np.array([-1.2, 0.5, 1.0, 0.25, -0.75, 0.1])
        base = weighted_dicke_audit(theta, 3, 555)
        shifted = weighted_dicke_audit(theta + 10.0, 3, 555)
        self.assertLess(base["max_probability_error"], 1e-12)
        self.assertLess(shifted["max_probability_error"], 1e-12)
        self.assertAlmostEqual(base["sample_total_variation"],
                               shifted["sample_total_variation"], places=12)

    def test_reversible_counter_borrows_and_returns_to_zero(self):
        # K=4 makes the very first subtraction borrow across three counter bits.
        weights = np.array([-1.7, 0.2, -0.8, 0.5, 1.1, -0.3])
        for k in (0, 1, 2, 3, 4, 6):
            with self.subTest(k=k):
                result = counter_audit(weights, k)
                self.assertEqual(result["total_qubits"], 6 + k.bit_length())
                self.assertLess(result["nonzero_counter_mass"], 1e-18)
                self.assertGreater(result["joint_state_fidelity"], 1 - 1e-12)


if __name__ == "__main__":
    unittest.main()
