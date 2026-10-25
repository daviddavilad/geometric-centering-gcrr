"""Tests for the closed-form asymptotic coefficients in theory.py."""

import numpy as np
import pytest

from gcrr.theory import (
    rho_B,
    rho_S,
    A_eta,
    phase_gap,
    digital_strict_even_coef,
    digital_hn_coef,
)

# Test 1: Appendix A.3 coefficients
@pytest.mark.parametrize(
    "K, expected_B, expected_S",
    [
        (105, 0.005508, 0.011192),
        (95, -0.020894, -0.058557),
    ],
)
def test_rho_values(K, expected_B, expected_S):
    params = (100, K, 0.5, 0.06, 0.20)

    np.testing.assert_allclose(rho_B(*params), expected_B, atol=1e-6, rtol=0)

    np.testing.assert_allclose(rho_S(*params), expected_S, atol=1e-6, rtol=0)

# Test 2: Theorem 3.2 phase-wise coefficients
@pytest.mark.parametrize(
    "K, expected_A0, expected_A1",
    [
        (105, -1.099837, 1.715667),
        (95, -1.224196, 1.066169),
    ],
)
def test_A_eta_values(K, expected_A0, expected_A1):
    params = (100, K, 0.5, 0.06, 0.20)

    np.testing.assert_allclose(A_eta(*params, 0), expected_A0, atol=1e-6, rtol=0)

    np.testing.assert_allclose(A_eta(*params, 1), expected_A1, atol=1e-6, rtol=0)

# Primary ITM and OTM cases
PRIMARY_CASES = [
    (100, 105, 0.5, 0.06, 0.20),
    (100, 95, 0.5, 0.06, 0.20),
]

# Eight WAND benchmark cases:
# S=40, r=0.06, K in {35,45},
# sigma in {0.2,0.4}, T in {0.25,0.5}
WAND_CASES = [
    (40, K, T, 0.06, sigma)
    for K in (35, 45)
    for sigma in (0.2, 0.4)
    for T in (0.25, 0.5)
]

CASE_IDS = ["OTM-K105", "ITM-K95"] + [
    f"WAND-K{K}-s{sigma}-T{T}"
    for K in (35, 45)
    for sigma in (0.2, 0.4)
    for T in (0.25, 0.5)
]

# Test 3: Corollary 3.3 phase-gap identity
@pytest.mark.parametrize("params", PRIMARY_CASES + WAND_CASES, ids=CASE_IDS)
def test_phase_gap_identity(params):
    a0 = A_eta(*params, 0)
    a1 = A_eta(*params, 1)

    predicted_gap = phase_gap(*params)
    computed_gap = a1 - a0

    np.testing.assert_allclose(computed_gap, predicted_gap, atol=1e-12, rtol=0)

# Test 4: Digital option coefficients
@pytest.mark.parametrize(
    "K, expected_strict, expected_hn0, expected_hn1",
    [
        (105, -0.379212, 0.020589, -0.018011),
        (95, -0.340954, -0.039174, 0.046767),
    ],
)
def test_digital_coefs(K, expected_strict, expected_hn0, expected_hn1):
    params = (100, K, 0.5, 0.06, 0.20)

    np.testing.assert_allclose(digital_strict_even_coef(*params), expected_strict, atol=1e-6, rtol=0)

    np.testing.assert_allclose(digital_hn_coef(*params, 0), expected_hn0, atol=1e-6, rtol=0)

    np.testing.assert_allclose(digital_hn_coef(*params, 1), expected_hn1, atol=1e-6, rtol=0)

# Test 5: Invalid phase rejection
def test_A_eta_rejects_bad_eta():
    with pytest.raises(ValueError):
        A_eta(100, 105, 0.5, 0.06, 0.20, 2)