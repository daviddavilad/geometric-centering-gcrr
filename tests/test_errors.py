"""Tests for phase-separated error infrastructure."""
import numpy as np
import pytest

from gcrr import schemes
from gcrr.errors import phase_separated_errors

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_phase_separation_partitions_n_values():
    """Even and odd n's should partition the input sweep cleanly."""
    n_values = np.arange(50, 100)
    out = phase_separated_errors(S, K, T, r, sigma, schemes.crr, n_values)
    assert len(out.n_even) + len(out.n_odd) == len(n_values)
    assert np.all(out.n_even % 2 == 0)
    assert np.all(out.n_odd % 2 == 1)


def test_errors_have_matching_shapes():
    """err and scaled_err arrays should have the same length as their n arrays."""
    n_values = np.arange(51, 201, 2)  # all odd
    out = phase_separated_errors(S, K, T, r, sigma, schemes.crr, n_values)
    assert len(out.n_odd) == len(out.err_odd) == len(out.scaled_err_odd)
    assert len(out.n_even) == len(out.err_even) == len(out.scaled_err_even)
    assert len(out.scaled_err_odd) == len(n_values)


def test_scaled_error_is_n_times_error():
    """Sanity: scaled_err_even = n_even * err_even, same for odd."""
    n_values = np.arange(100, 200)
    out = phase_separated_errors(S, K, T, r, sigma, schemes.crr, n_values)
    np.testing.assert_allclose(
        out.scaled_err_even, out.n_even * out.err_even, rtol=1e-12
    )
    np.testing.assert_allclose(
        out.scaled_err_odd, out.n_odd * out.err_odd, rtol=1e-12
    )


def test_gcrr_gc_scaled_error_stabilizes_phase_wise():
    """GCRR-GC: scaled error n*(C_n - C_BS) should approach a phase-wise
    constant A_eta as n grows. The standard deviation of the scaled errors
    within each phase should be small once n is large enough."""
    n_values = np.arange(500, 1001)
    out = phase_separated_errors(S, K, T, r, sigma, schemes.gcrr_gc, n_values)
    # Within each phase, scaled errors should cluster tightly.
    # (Allow some slack -- next-order correction is O(1/sqrt(n)).)
    assert np.std(out.scaled_err_even) < 0.05, (
        f"GCRR-GC even-phase scaled error not flat: std = {np.std(out.scaled_err_even):.4f}"
    )
    assert np.std(out.scaled_err_odd) < 0.05, (
        f"GCRR-GC odd-phase scaled error not flat: std = {np.std(out.scaled_err_odd):.4f}"
    )


def test_crr_scaled_error_does_not_stabilize():
    """CRR: scaled error oscillates with n at non-ATM strikes. The standard
    deviation should be substantially larger than for a centered scheme.

    Note: at exactly S = K, CRR's strike-location coordinate is parity-locked
    (gamma_n = n/2 exactly for all n), so no oscillation appears. This test
    therefore uses K = 95 (in-the-money), where log(K/S0) != 0 produces the
    irrational fractional part that drives the CRR sawtooth.
    """
    K_itm = 95.0  # in-the-money, log(K/S0) != 0
    n_values = np.arange(500, 1001)
    out_crr = phase_separated_errors(S, K_itm, T, r, sigma, schemes.crr, n_values)
    crr_std = max(np.std(out_crr.scaled_err_even), np.std(out_crr.scaled_err_odd))
    assert crr_std > 0.5, (
        f"CRR scaled error didn't scatter as expected: max-std = {crr_std:.4f}"
    )
