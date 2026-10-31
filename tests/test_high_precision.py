"""Tests for high-precision GCRR-GC and Black-Scholes pricing.

All high-precision calculations use mp.workdps(40) to avoid
changing the global mpmath precision for other tests.
"""

import mpmath as mp
import pytest

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.extrapolation import richardson_same_phase
from gcrr.high_precision import _upper_tail_mp, bs_call_mp, gcrr_gc_call_mp
from gcrr.pricer import two_tail_call_price

S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20


# Test high-precision versus double-precision call pricing
@pytest.mark.parametrize("n", [51, 500])
def test_mp_call_matches_double(n):
    params = schemes.gcrr_gc(S, K, T, r, sigma, n)

    C_double = two_tail_call_price(S, K, T, r, params, m=n // 2 + 1)

    with mp.workdps(40):
        C_mp = gcrr_gc_call_mp(S, K, T, r, sigma, n)

        assert abs(float(C_mp) - C_double) < 1e-10


# Test high-precision versus double-precision Black-Scholes
def test_bs_call_mp_matches_double():
    C_double = bs_call(S, K, T, r, sigma)

    with mp.workdps(40):
        C_mp = bs_call_mp(S, K, T, r, sigma)

        assert abs(float(C_mp) - C_double) < 1e-12


# Test high-precision same-phase Richardson convergence
def test_same_phase_rate_is_second_order_high_precision():
    """Verify the observed O(n^-2) rate at n=1000 and n=2000.

    High precision avoids the rounding contamination that caused
    the corresponding double-precision test to fail.
    """
    with mp.workdps(40):
        C_BS = bs_call_mp(S, K, T, r, sigma)

        def extrapolation_error(n):
            C_n = gcrr_gc_call_mp(S, K, T, r, sigma, n)
            C_n2 = gcrr_gc_call_mp(S, K, T, r, sigma, n + 2)

            C_R = richardson_same_phase(C_n, C_n2, n)

            return C_R - C_BS

        error_1000 = extrapolation_error(1000)
        error_2000 = extrapolation_error(2000)

        ratio = abs(error_1000) / abs(error_2000)

        assert mp.mpf("3.99") < ratio < mp.mpf("4.03"), (
            f"Expected ratio near 4.007; observed {mp.nstr(ratio, 12)}"
        )


@pytest.mark.parametrize("n", [51, 1000, 5001])
@pytest.mark.parametrize("p", ["0.49", "0.5", "0.5123"])
def test_upper_tail_matches_betainc(n, p):
    """Direct tail sum agrees with mpmath's incomplete beta where the latter converges."""
    with mp.workdps(40):
        m = n // 2 + 1
        direct = _upper_tail_mp(n, m, mp.mpf(p))
        beta = mp.betainc(m, n - m + 1, 0, mp.mpf(p), regularized=True)
        assert abs(direct - beta) / beta < mp.mpf("1e-35")


def test_mp_price_at_large_n_matches_double():
    """At n = 100,000, beyond betainc's range, the 40-digit price agrees with double precision."""
    n = 100_000
    with mp.workdps(40):
        price_mp = gcrr_gc_call_mp(S, K, T, r, sigma, n)
    params = schemes.gcrr_gc(S, K, T, r, sigma, n)
    price_double = two_tail_call_price(S, K, T, r, params)
    assert abs(float(price_mp) - price_double) < 1e-9
