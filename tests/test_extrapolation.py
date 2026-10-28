"""Tests for phase-aware Richardson extrapolation (Section 3.3).

Synthetic tests verify exact algebraic identities.
Real-tree tests verify the observed convergence behavior of GCRR-GC.
"""

import numpy as np
import pytest

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.extrapolation import (
    phase_adjust,
    richardson_adjacent,
    richardson_same_phase,
)
from gcrr.pricer import two_tail_call_price
from gcrr.theory import phase_gap

# Parameters and helpers
S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20


def gc_price(n: int) -> float:
    """GCRR-GC call via two-tail pricing at m = N + 1."""
    params = schemes.gcrr_gc(S, K, T, r, sigma, n)
    return two_tail_call_price(S, K, T, r, params, m=n // 2 + 1)


def synthetic(n: int, CBS=4.79, A0=-1.1, A1=1.7) -> float:
    """Exact leading terms of Theorem 3.2, no remainder."""
    A = A0 if n % 2 == 0 else A1
    return CBS + A / n


# ============================================================
# Synthetic tests: exact algebraic identities
# ============================================================


@pytest.mark.parametrize("n", [100, 101, 1000])
def test_same_phase_cancels_leading_term(n):
    result = richardson_same_phase(synthetic(n), synthetic(n + 2), n)

    np.testing.assert_allclose(result, 4.79, atol=1e-10, rtol=0)


@pytest.mark.parametrize("n", [100, 101])
def test_adjacent_error_is_signed_gap(n):
    gap = 1.7 - (-1.1)  # 2.8

    result = richardson_adjacent(synthetic(n), synthetic(n + 1), n)

    expected = 4.79 + gap if n % 2 == 0 else 4.79 - gap

    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("n", [100, 101])
def test_phase_adjust_puts_both_phases_on_A0_track(n):
    gap = 2.8

    adjusted = phase_adjust(synthetic(n), n, gap)
    expected = 4.79 - 1.1 / n

    np.testing.assert_allclose(adjusted, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("n", [100, 101])
def test_phase_adjusted_adjacent_recovers_CBS(n):
    gap = 2.8

    C_n = phase_adjust(synthetic(n), n, gap)
    C_n1 = phase_adjust(synthetic(n + 1), n + 1, gap)

    result = richardson_adjacent(C_n, C_n1, n)

    np.testing.assert_allclose(result, 4.79, atol=1e-10, rtol=0)


# ============================================================
# Real-tree tests — GCRR-GC at K = 105
# ============================================================


def test_same_phase_richardson_real():
    n = 1000
    C_BS = bs_call(S, K, T, r, sigma)

    result = richardson_same_phase(gc_price(n), gc_price(n + 2), n)

    error = abs(result - C_BS)

    assert error < 1e-6


def test_same_phase_richardson_is_second_order():
    """Error ratio between n and 2n is ~4 (rate n^-2). Uses n = 200, 400:
    at larger n, double-precision cancellation in (n+2)C_{n+2} - nC_n
    contaminates errors below ~1e-8 (observed 19% at n = 5000)."""
    C_BS = bs_call(S, K, T, r, sigma)

    errors = []

    for n in (200, 400):
        result = richardson_same_phase(gc_price(n), gc_price(n + 2), n)
        errors.append(abs(result - C_BS))

    ratio = errors[0] / errors[1]

    assert 3.5 < ratio < 4.5, (
        f"Expected approximately 4x error reduction; observed ratio = {ratio:.6f}"
    )


def test_adjacent_richardson_error_is_phase_gap():
    n = 1000

    C_BS = bs_call(S, K, T, r, sigma)
    gap = phase_gap(S, K, T, r, sigma)

    result = richardson_adjacent(gc_price(n), gc_price(n + 1), n)

    assert abs((result - C_BS) - gap) < 1e-3


def test_phase_adjusted_adjacent_is_first_order():
    C_BS = bs_call(S, K, T, r, sigma)
    gap = phase_gap(S, K, T, r, sigma)

    errors = []

    for n in (1000, 2000):
        C_n = phase_adjust(gc_price(n), n, gap)
        C_n1 = phase_adjust(gc_price(n + 1), n + 1, gap)

        result = richardson_adjacent(C_n, C_n1, n)

        errors.append(abs(result - C_BS))

    ratio = errors[0] / errors[1]

    assert 1.8 < ratio < 2.2, (
        f"Expected approximately 2x error reduction; observed ratio = {ratio:.6f}"
    )
