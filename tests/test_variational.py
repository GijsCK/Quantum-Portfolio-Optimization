import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from portfolio_generator import generate
from variational import conditional_table, measure_energy, sample_fixed, solve


def test_suffix_probabilities_equal_enumeration_and_survive_logit_shift():
    theta = np.array([-2., 1.5, .3, -.8, 2.])
    for k in range(len(theta) + 1):
        choices = list(itertools.combinations(range(len(theta)), k))
        logits = np.array([theta[list(s)].sum() for s in choices])
        table = conditional_table(theta, k)
        assert table[0, k] == pytest.approx(logsumexp(logits))
        for indices, true_logp in zip(choices, logits - logsumexp(logits)):
            r = k
            logp = 0.
            for i in range(len(theta)):
                if i in indices:
                    logp += theta[i] + table[i + 1, r - 1] - table[i, r]
                    r -= 1
                else:
                    logp += table[i + 1, r] - table[i, r]
            assert logp == pytest.approx(true_logp)
        for offset in (0., 300.):
            sample = sample_fixed(theta + offset, k, 30, np.random.default_rng(91))
            assert np.all(sample.sum(axis=1) == k)


def test_observed_distribution_is_consistent_with_analytic_born_law():
    theta = np.array([-1.1, -.7, .4, .6, 1.])
    pairs = list(itertools.combinations(range(5), 2))
    scores = np.array([theta[a] + theta[b] for a, b in pairs])
    prob = np.exp(scores - logsumexp(scores))
    shots = 40_000
    bits = sample_fixed(theta, 2, shots, np.random.default_rng(19))
    freq = np.array([(bits[:, a] & bits[:, b]).mean() for a, b in pairs])
    assert np.all(np.abs(freq - prob) < 6 * np.sqrt(prob * (1. - prob) / shots))


def test_hamiltonian_matches_official_dense_cost_and_enforces_constraints():
    inst = generate(10, 77)
    choices = np.array(list(itertools.combinations(range(inst.n), inst.K)))
    states = np.zeros((len(choices), inst.n), dtype=int)
    np.put_along_axis(states, choices, 1, axis=1)
    score, energy, valid = measure_energy(inst, states)
    dense = inst.Sigma()
    for i, state in enumerate(states):
        w = state / inst.K
        assert score[i] == pytest.approx(.5 * (w @ dense @ w - inst.mu @ w))
        assert score[i] == pytest.approx(inst.objective(state))
        assert valid[i] == inst.is_feasible(state)
    assert np.array_equal(energy[valid], score[valid])
    assert energy[~valid].min() > score[valid].max()
    with pytest.raises(ValueError):
        measure_energy(inst, states + 1)


def test_all_controls_share_a_prior_shot_budget_and_random_stream():
    inst = generate(20, 77)
    runs = [solve(inst, mode) for mode in ("trained", "schedule_only", "frozen")]
    assert all(r["trace"][0] == runs[0]["trace"][0] for r in runs)
    for r in runs:
        assert r["shots"] == 1280
        assert r["selected"] is not None
        assert len(set(r["selected"])) == inst.K
        assert r["objective"] <= r["trace"][-1]["best_feasible_score"] + 1e-14
