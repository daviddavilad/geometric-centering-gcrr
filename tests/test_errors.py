"""Tests for phase-separated error infrastructure."""

import numpy as np

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
    np.testing.assert_allclose(out.scaled_err_odd, out.n_odd * out.err_odd, rtol=1e-12)


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


def test_tian_1999_has_step_discontinuities_otm():
    """Tian (1999) strike-aligned scheme has discontinuities in n where the
    nearest-integer j_0 = round(eta) jumps. The scaled-error sequence is
    therefore piecewise-smooth but not continuous in n.

    GCRR-GC avoids this because its lambda formula contains no integer
    rounding -- lambda_n = (kappa_n + sqrt(kappa_n^2 + 4))/2 is continuous in n.

    Test: compute successive scaled-error differences. Tian's series
    should have at least a few large jumps in [50, 500]; GCRR-GC's
    should not.
    """
    K_otm = 105.0  # OTM, where Tian's j_0 transitions are visible
    n_values = np.arange(50, 501)

    out_tian = phase_separated_errors(
        S, K_otm, T, r, sigma, schemes.tian_1999, n_values
    )
    out_gc = phase_separated_errors(S, K_otm, T, r, sigma, schemes.gcrr_gc, n_values)

    # Successive differences in scaled error, within the even-n phase.
    # Tian: should see at least 2 jumps of size > 0.05 (the visible steps).
    # GCRR-GC: max successive diff should be much smaller.
    tian_diffs = np.abs(np.diff(out_tian.scaled_err_even))
    gc_diffs = np.abs(np.diff(out_gc.scaled_err_even))

    large_jumps_tian = np.sum(tian_diffs > 0.05)
    assert large_jumps_tian >= 2, (
        f"Expected >=2 large jumps in Tian's even-n scaled error at K=105, "
        f"found {large_jumps_tian}"
    )

    # GCRR-GC should be smooth -- no jumps above 0.05.
    max_gc_jump = np.max(gc_diffs)
    assert max_gc_jump < 0.05, (
        f"GCRR-GC's even-n scaled error should be smooth in n at K=105, "
        f"but max successive diff = {max_gc_jump:.4f}"
    )
