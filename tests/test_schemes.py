"""Correctness gates for the scheme parameterizations."""
import numpy as np
import pytest

from gcrr import schemes
from gcrr.pricer import binomial_price
from gcrr.black_scholes import bs_call

# Standard test case
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_crr_is_lambda_one():
    """CRR is the GCRR special case lambda = 1; the symmetry u*d = 1 holds."""
    p = schemes.crr(S, K, T, r, sigma, n=101)
    assert p.lam == 1.0
    assert p.u * p.d == pytest.approx(1.0, abs=1e-12)


def test_crr_converges_to_bs():
    """CRR price should approach Black-Scholes as n grows."""
    bs = bs_call(S, K, T, r, sigma)
    p = schemes.crr(S, K, T, r, sigma, n=2001)
    price = binomial_price(S, K, T, r, p, "call", "european")
    assert price == pytest.approx(bs, abs=1e-2)


def test_gcrr_xpc_requires_even_n():
    """GCRR-XPC must reject odd n explicitly."""
    with pytest.raises(ValueError):
        schemes.gcrr_xpc(S, K, T, r, sigma, n=101)


def test_gcrr_xpc_central_node_on_strike():
    """GCRR-XPC: the central terminal node S * u^(n/2) * d^(n/2) must equal K."""
    n = 200
    p = schemes.gcrr_xpc(S, K, T, r, sigma, n)
    half = n // 2
    central_node = S * (p.u ** half) * (p.d ** half)
    assert central_node == pytest.approx(K, rel=1e-10)


def test_gcrr_gc_strike_alignment_even():
    """GCRR-GC: for even n, the central node must equal K (matches GCRR-XPC)."""
    n = 200
    p = schemes.gcrr_gc(S, K, T, r, sigma, n)
    N = n // 2
    central_node = S * (p.u ** N) * (p.d ** (n - N))
    assert central_node == pytest.approx(K, rel=1e-10)


def test_gcrr_gc_geometric_center_odd():
    """GCRR-GC: for odd n, K^2 must equal S_{N} * S_{N+1} (geometric midpoint)."""
    n = 201
    p = schemes.gcrr_gc(S, K, T, r, sigma, n)
    N = n // 2  # = 100
    S_lower = S * (p.u ** N) * (p.d ** (n - N))
    S_upper = S * (p.u ** (N + 1)) * (p.d ** (n - N - 1))
    assert S_lower * S_upper == pytest.approx(K**2, rel=1e-10)


def test_gcrr_gc_and_xpc_coincide_at_even_n():
    """The two lambdas must be identical for even n (sanity check on the algebra)."""
    n = 200
    p_gc = schemes.gcrr_gc(S, K, T, r, sigma, n)
    p_xpc = schemes.gcrr_xpc(S, K, T, r, sigma, n)
    assert p_gc.lam == pytest.approx(p_xpc.lam, rel=1e-14)
    assert p_gc.u == pytest.approx(p_xpc.u, rel=1e-14)
    assert p_gc.d == pytest.approx(p_xpc.d, rel=1e-14)


def test_gcrr_gc_converges_to_bs():
    """GCRR-GC should converge to Black-Scholes as n grows."""
    bs = bs_call(S, K, T, r, sigma)
    p = schemes.gcrr_gc(S, K, T, r, sigma, n=2001)
    price = binomial_price(S, K, T, r, p, "call", "european")
    assert price == pytest.approx(bs, abs=1e-2)


def test_gcrr_fixed_lambda_one_equals_crr():
    """gcrr_fixed with lam=1 must produce identical TreeParameters as crr()."""
    n = 101
    p_crr = schemes.crr(S, K, T, r, sigma, n)
    p_fixed = schemes.gcrr_fixed(S, K, T, r, sigma, n, lam=1.0)
    assert p_crr.u == pytest.approx(p_fixed.u, rel=1e-14)
    assert p_crr.d == pytest.approx(p_fixed.d, rel=1e-14)
    assert p_crr.p == pytest.approx(p_fixed.p, rel=1e-14)
