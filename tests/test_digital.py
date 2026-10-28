"""Tests for digital option pricing."""

import numpy as np
import pytest
from scipy.stats import norm

from gcrr import schemes
from gcrr.black_scholes import bs_digital_strict
from gcrr.digital import digital_strict_price
from gcrr.digital import digital_strict_price, digital_half_node_corrected_price


# Standard ATM-ish parameter set; matches the call test conventions.
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_digital_strict_in_unit_interval():
    """The strict digital is a discounted probability, so 0 <= price <= e^{-rT}."""
    params = schemes.gcrr_gc(S, K, T, r, sigma, 200)
    price = digital_strict_price(S, K, T, r, params)
    assert 0.0 <= price <= np.exp(-r * T) * 1.0 + 1e-12


def test_digital_strict_monotone_in_K():
    """Higher strike -> lower digital price (less likely S_T > K)."""
    n = 200
    K_values = [80.0, 90.0, 100.0, 110.0, 120.0]
    prices = []
    for K_i in K_values:
        params = schemes.gcrr_gc(S, K_i, T, r, sigma, n)
        prices.append(digital_strict_price(S, K_i, T, r, params))
    # Check strictly decreasing
    diffs = np.diff(prices)
    assert np.all(diffs < 0), f"Digital price not monotone in K: {prices}"


def test_digital_strict_converges_to_bs_at_sqrt_n_rate():
    """The strict digital converges to e^{-rT} Phi(d_2) at O(1/sqrt(n)).

    The leading error is the 'central-atom discrepancy' (Section 4):
    the strict digital's even-phase error is O(1/sqrt(n)) rather
    than O(1/n). This slow rate is the motivation for the
    half-node corrected digital, which removes the central-atom term.

    Test: error * sqrt(n) should be approximately constant across n.
    """
    bs_ref = bs_digital_strict(S, K, T, r, sigma)
    n_values = [200, 800, 3200]
    scaled_errors = []
    for n in n_values:
        params = schemes.gcrr_gc(S, K, T, r, sigma, n)
        price = digital_strict_price(S, K, T, r, params)
        scaled_errors.append((price - bs_ref) * np.sqrt(n))
    # The scaled errors should be roughly constant (same leading constant
    # across phase-matched n; here all three are even so same phase).
    # Sanity check: they should agree to ~1% relative.
    mean_scaled = np.mean(scaled_errors)
    max_dev = max(abs(s - mean_scaled) for s in scaled_errors)
    rel_dev = max_dev / abs(mean_scaled)
    assert rel_dev < 0.01, (
        f"Scaled errors not constant: error*sqrt(n) at n={n_values} were "
        f"{scaled_errors}, max relative deviation = {rel_dev:.4f}"
    )


def test_digital_strict_at_K_equals_S_with_zero_drift():
    """When K = S and r = sigma^2/2 (zero log-drift), the BS digital
    equals exactly e^{-rT}/2. The binomial strict digital approaches
    this at the O(1/sqrt(n)) rate of Section 4.

    This is a sanity check on the d2 sign and the discount factor.
    At n = 1000, the 1/sqrt(n) envelope allows error ~ 0.03.
    """
    sigma_test = 0.20
    r_test = 0.5 * sigma_test**2  # zero log-drift: r = sigma^2/2
    bs_value = np.exp(-r_test * T) * 0.5  # Phi(0) = 0.5
    params = schemes.gcrr_gc(S, S, T, r_test, sigma_test, 1000)
    price = digital_strict_price(S, S, T, r_test, params)
    # Tolerance set to accommodate the O(1/sqrt(n)) error rate
    assert abs(price - bs_value) < 0.03, (
        f"At K=S with zero log-drift: digital = {price:.6f}, expected ~{bs_value:.6f}"
    )


def test_bs_digital_strict_at_K_equals_S_zero_drift():
    """Sanity check on the BS digital: zero log-drift -> exactly e^{-rT}/2."""
    sigma_test = 0.20
    r_test = 0.5 * sigma_test**2
    bs_value = bs_digital_strict(S, S, T, r_test, sigma_test)
    expected = np.exp(-r_test * T) * 0.5
    assert abs(bs_value - expected) < 1e-12, (
        f"BS digital sanity check failed: got {bs_value}, expected {expected}"
    )


def test_half_node_corrected_digital_symmetry_at_p_half():
    """The half-node corrected digital satisfies the exact symmetry
    Psi^hn_{N,0}(1/2) = Psi^hn_{N,1}(1/2) = 1/2.

    This is the design property of the correction: at p = 1/2 (zero drift),
    the corrected upper tail equals exactly 1/2 in both phases.

    Implementation note: tested by directly checking the Psi^hn_{N,eta}(1/2)
    formula. We use n = 50 (even) and n = 51 (odd) as a representative
    parity-pair; the symmetry is exact in floating point for any N.
    """
    from scipy.stats import binom

    # Even phase: n = 2N, Psi^hn_{N,0}(1/2) = Psi_{N,0}(1/2) + (1/2) C(2N, N) (1/2)^(2N)
    for n in [50, 100, 200, 500]:
        N = n // 2
        psi = binom.sf(N, n, 0.5)
        central_atom = binom.pmf(N, n, 0.5)
        psi_star = psi + 0.5 * central_atom
        assert abs(psi_star - 0.5) < 1e-12, (
            f"Even-phase symmetry failed at n={n}: Psi*={psi_star}"
        )

    # Odd phase: Psi^hn_{N,1}(1/2) = Psi_{N,1}(1/2) since correction is zero
    for n in [51, 101, 201, 501]:
        N = n // 2  # = (n-1)/2 for odd n
        psi = binom.sf(N, n, 0.5)
        # No correction in odd phase
        assert abs(psi - 0.5) < 1e-12, f"Odd-phase symmetry failed at n={n}: Psi={psi}"


def test_half_node_corrected_digital_equals_strict_for_odd_n():
    """For odd n, the half-node correction is zero, so the corrected
    digital equals the strict digital exactly."""
    for n in [51, 101, 201, 501]:
        params = schemes.gcrr_gc(S, K, T, r, sigma, n)
        strict = digital_strict_price(S, K, T, r, params)
        corrected = digital_half_node_corrected_price(S, K, T, r, params)
        assert strict == corrected, (
            f"Strict and corrected disagree for odd n={n}: "
            f"strict={strict}, corrected={corrected}"
        )


def test_half_node_corrected_digital_faster_convergence_even_phase():
    """The half-node corrected digital should converge faster than the
    strict digital in the even phase. Specifically, the strict digital's
    error is O(1/sqrt(n)) and the corrected digital's error should be
    O(1/n) -- so error * n should be roughly constant for the corrected
    digital, while error * sqrt(n) is roughly constant for the strict.
    """
    bs_ref = bs_digital_strict(S, K, T, r, sigma)
    n_values = [200, 800, 3200]
    scaled_corrected = []
    for n in n_values:
        params = schemes.gcrr_gc(S, K, T, r, sigma, n)
        corrected = digital_half_node_corrected_price(S, K, T, r, params)
        # error * n should be roughly constant if O(1/n) convergence
        scaled_corrected.append((corrected - bs_ref) * n)
    # Check stability across n -- should be within ~5% relative
    mean_scaled = np.mean(scaled_corrected)
    max_dev = max(abs(s - mean_scaled) for s in scaled_corrected)
    rel_dev = max_dev / abs(mean_scaled) if mean_scaled != 0 else 0.0
    assert rel_dev < 0.05, (
        f"Corrected digital not O(1/n): error*n at n={n_values} were "
        f"{scaled_corrected}, max relative deviation = {rel_dev:.4f}"
    )


def test_digital_rejects_uncentered_tree():
    """CRR at K=105, n=200 (file parameters, T=1): gamma_n ~ 101.73, so
    m_n = 102 != N + 1 = 101, and the centered-tree guard must raise."""
    params = schemes.crr(S, 105.0, T, r, sigma, 200)
    with pytest.raises(ValueError):
        digital_strict_price(S, 105.0, T, r, params)
