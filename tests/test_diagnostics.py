"""Strike-location diagnostic tests."""
import math

import numpy as np
import pytest

from gcrr import schemes
from gcrr.diagnostics import strike_location, strike_location_sweep

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_strike_location_returns_valid_coordinates():
    """Basic sanity: gamma, J, theta, Delta have the right ranges."""
    p = schemes.crr(S, K, T, r, sigma, n=101)
    loc = strike_location(S, K, p)
    assert isinstance(loc.J, int)
    assert loc.J == math.ceil(loc.gamma)
    assert 0.0 <= loc.theta < 1.0
    assert -1.0 < loc.Delta <= 1.0
    # Delta = 1 - 2*theta
    assert loc.Delta == pytest.approx(1.0 - 2.0 * loc.theta, abs=1e-14)


def test_gcrr_gc_even_n_strike_on_node():
    """GCRR-GC at even n: strike sits exactly on the central node.
    => gamma = n/2 (integer), theta = 0, Delta = 1."""
    n = 200
    p = schemes.gcrr_gc(S, K, T, r, sigma, n)
    loc = strike_location(S, K, p)
    assert loc.gamma == pytest.approx(n / 2, abs=1e-10)
    assert loc.J == n // 2 or loc.J == n // 2  # = 100 (either way ceil hits 100)
    assert loc.theta == pytest.approx(0.0, abs=1e-10)
    assert loc.Delta == pytest.approx(1.0, abs=1e-10)


def test_gcrr_gc_odd_n_strike_at_geometric_midpoint():
    """GCRR-GC at odd n: strike sits at geometric midpoint of two central nodes.
    => gamma = n/2 (half-integer), theta = 1/2, Delta = 0."""
    n = 201
    p = schemes.gcrr_gc(S, K, T, r, sigma, n)
    loc = strike_location(S, K, p)
    assert loc.gamma == pytest.approx(n / 2.0, abs=1e-10)
    assert loc.J == (n + 1) // 2  # = 101
    assert loc.theta == pytest.approx(0.5, abs=1e-10)
    assert loc.Delta == pytest.approx(0.0, abs=1e-10)


def test_joshi_2009_strike_at_geometric_midpoint():
    """Joshi (2009) for odd n: same geometric midpoint property as GCRR-GC odd n.
    Different parameterization, same geometric placement."""
    n = 201
    p = schemes.joshi_2009(S, K, T, r, sigma, n)
    loc = strike_location(S, K, p)
    assert loc.gamma == pytest.approx(n / 2.0, abs=1e-10)
    assert loc.theta == pytest.approx(0.5, abs=1e-10)
    assert loc.Delta == pytest.approx(0.0, abs=1e-10)


def test_gcrr_xpc_even_n_strike_on_node():
    """GCRR-XPC: same property as GCRR-GC at even n."""
    n = 200
    p = schemes.gcrr_xpc(S, K, T, r, sigma, n)
    loc = strike_location(S, K, p)
    assert loc.Delta == pytest.approx(1.0, abs=1e-10)


def test_crr_delta_oscillates():
    """CRR's Delta coordinate should NOT be constant across n -- it oscillates.
    Verify by checking the standard deviation across a range of n."""
    n_values = np.arange(50, 250)
    out = strike_location_sweep(S, K, T, r, sigma, schemes.crr, n_values)
    # CRR's Delta should bounce around; expect std well above 0.
    assert np.std(out["Delta"]) > 0.3


def test_gcrr_gc_delta_is_parity_locked():
    """GCRR-GC's Delta should be exactly 1 for even n and exactly 0 for odd n,
    across the full sweep -- the 'parity-free' property in Chen's draft."""
    n_values = np.arange(50, 250)
    out = strike_location_sweep(S, K, T, r, sigma, schemes.gcrr_gc, n_values)
    even_mask = (n_values % 2 == 0)
    odd_mask = ~even_mask
    np.testing.assert_allclose(out["Delta"][even_mask], 1.0, atol=1e-10)
    np.testing.assert_allclose(out["Delta"][odd_mask], 0.0, atol=1e-10)


def test_sweep_returns_correct_shape():
    """Sanity: sweep output arrays have the same length as input n_values."""
    n_values = np.array([51, 101, 201, 501])
    out = strike_location_sweep(S, K, T, r, sigma, schemes.crr, n_values)
    for key in ("gamma", "J", "theta", "Delta"):
        assert len(out[key]) == len(n_values)
