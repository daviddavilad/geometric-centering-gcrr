import numpy as np
import pytest

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.errors import fit_leading_constant
from gcrr.pricer import binomial_price, two_tail_call_price
from gcrr.theory import A_eta


@pytest.mark.parametrize(
    "scheme",
    [
        schemes.crr,
        schemes.gcrr_gc,
        schemes.gcrr_fixed,
        schemes.tian_1999,
        schemes.joshi_2009,
    ],
)
@pytest.mark.parametrize("K", [95.0, 105.0])
@pytest.mark.parametrize("n", [50, 51, 200, 201])
def test_two_tail_matches_backward_induction(scheme, K, n):
    S = 100.0
    T = 0.5
    r = 0.06
    sigma = 0.20

    # Construct the binomial tree
    params = scheme(S, K, T, r, sigma, n)

    # Original backward-induction pricing
    backward = binomial_price(S, K, T, r, params, "call", "european")

    # Exact two-numeraire tail representation
    two_tail = two_tail_call_price(S, K, T, r, params)

    # Numerical equivalence
    assert abs(two_tail - backward) < 1e-10, (
        f"{scheme.__name__}, K={K}, n={n}: "
        f"backward={backward}, two_tail={two_tail}, "
        f"difference={abs(two_tail - backward)}"
    )


@pytest.mark.parametrize(
    "K", [95.0, 100.0, 105.0]
)  # K=100: ATM, where GCRR-GC = CRR (lambda = 1)
@pytest.mark.parametrize("eta", [0, 1])
def test_fitted_A_eta_matches_theorem(K, eta):
    S = 100.0
    T = 0.5
    r = 0.06
    sigma = 0.20

    # Step counts, separated by parity
    bases = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    n_values = np.array([base + eta for base in bases])

    # Black-Scholes reference
    C_BS = bs_call(S, K, T, r, sigma)

    # Empirical scaled errors
    scaled_errors = []

    for n in n_values:
        n = int(n)

        params = schemes.gcrr_gc(S, K, T, r, sigma, n)

        # Proposition 2.1: unified strict threshold
        m = n // 2 + 1

        C_n = two_tail_call_price(S, K, T, r, params, m=m)

        scaled_errors.append(n * (C_n - C_BS))

    # Extrapolate the leading asymptotic coefficient
    A_hat = fit_leading_constant(n_values, scaled_errors)

    # Independent theoretical prediction
    A_theory = A_eta(S, K, T, r, sigma, eta)

    print(
        f"\nK={K}, eta={eta}: "
        f"A_hat={A_hat:.10f}, "
        f"A_theory={A_theory:.10f}, "
        f"error={abs(A_hat - A_theory):.3e}"
    )

    assert abs(A_hat - A_theory) < 1e-5
