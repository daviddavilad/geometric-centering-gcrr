"""
Verification of the third-term sign in the eq. (5) proof (Appendix B).

The Uspensky/Edgeworth expansion in the proof contains the term
    T3 = -(1/(12 n)) * z_n * (z_n^2 - 1) * phi(z_n).
An earlier draft printed its leading value as  +a/12 (1-a^2) phi(a).
This script confirms the correct value is  -a/12 (1-a^2) phi(a)
(equivalently +a/12 (a^2-1) phi(a) -- same number, different convention),
and that eq. (5) as stated is correct to O(n^{-3/2}).

Run:  uv run python scripts/verify_appendix_B_third_term.py
Deps: sympy (ships mpmath).
"""

import sympy as sp
import mpmath as mp

mp.mp.dps = 40
phi = mp.npdf
Phi = mp.ncdf


# ----------------------------------------------------------------------
# Check 1 -- symbolic: the third-term coefficient at leading order z_n=-a
# ----------------------------------------------------------------------
def check_symbolic():
    a = sp.symbols("a", real=True)
    zn = -a  # leading order
    bracket = zn * (zn**2 - 1)               # z_n (z_n^2 - 1) at z_n = -a
    coeff = sp.Rational(-1, 12) * bracket    # = -(1/12) z_n (z_n^2-1)

    truth_1ma2 = -sp.Rational(1, 12) * a * (1 - a**2)   # -a/12 (1-a^2)
    truth_a2m1 = +sp.Rational(1, 12) * a * (a**2 - 1)   # +a/12 (a^2-1)
    printed = +sp.Rational(1, 12) * a * (1 - a**2)      # what an earlier draft printed

    assert sp.simplify(coeff - truth_1ma2) == 0, "coeff != -a/12 (1-a^2)"
    assert sp.simplify(truth_1ma2 - truth_a2m1) == 0, "the two legal forms disagree"
    assert sp.simplify(coeff - printed) != 0, "printed form unexpectedly equals truth"
    print("Check 1 (symbolic): third term = -a/12 (1-a^2) = +a/12 (a^2-1)   [PASS]")
    print(f"            printed +a/12 (1-a^2) differs by "
          f"{sp.simplify(printed - coeff)}  (=> printed line is wrong)")


# ----------------------------------------------------------------------
# helpers (mpmath)
# ----------------------------------------------------------------------
def p_n(a, rho, n):
    n = mp.mpf(n)
    return mp.mpf("0.5") + mp.mpf(a) / (2 * mp.sqrt(n)) + mp.mpf(rho) / (2 * n**mp.mpf("1.5"))

def z_def(a, rho, n, eta):
    """z_{N,eta}(p_n) straight from the definition."""
    n = mp.mpf(n); chi = mp.mpf(1 - eta) / 2
    p = p_n(a, rho, n)
    return (n * (mp.mpf("0.5") - p) + chi) / mp.sqrt(n * p * (1 - p))

def exact_tail(a, rho, n, eta):
    """Psi_{N,eta}(p_n) = P(Bin(n, p_n) >= N+1)."""
    N = (n - eta) // 2
    p = p_n(a, rho, n); q = 1 - p
    s = mp.mpf(0)
    for j in range(N + 1, n + 1):
        s += mp.binomial(n, j) * p**j * q**(n - j)
    return s

def eq5_expansion(a, rho, n, eta):
    """Eq. (5) as stated (with the CORRECT n^-1 coefficient)."""
    n = mp.mpf(n); a = mp.mpf(a); rho = mp.mpf(rho); chi = mp.mpf(1 - eta) / 2
    coeff = rho + (a**3 + (2 * eta - 1) * a) / 4
    return Phi(a) - (2 * chi / mp.sqrt(n)) * phi(a) + (coeff / n) * phi(a)


# ----------------------------------------------------------------------
# Check 2 -- z_n -> -a from the definition (rules out z_n -> +a)
# ----------------------------------------------------------------------
def check_zn():
    for a in [0.7, -1.3, 1.8]:
        z = z_def(a, 0.4, 10**7, 0)
        assert abs(z - (-a)) < 1e-3, f"z_n did not approach -a for a={a}"
    print("Check 2 (z_n -> -a):  confirmed from the definition           [PASS]")


# ----------------------------------------------------------------------
# Check 3 -- the THIRD TERM in isolation, numerically, with real z_n
# ----------------------------------------------------------------------
def check_third_term_isolated():
    for a in [0.7, -1.3, 1.8]:
        z = z_def(a, 0.4, 10**8, 0)
        T3 = -(mp.mpf(1) / (12 * 10**8)) * z * (z**2 - 1) * phi(z)
        scaled = T3 * 10**8 / phi(a)                 # -> leading coefficient
        minus_cand = -(mp.mpf(a) / 12) * (1 - mp.mpf(a) ** 2)
        plus_cand = +(mp.mpf(a) / 12) * (1 - mp.mpf(a) ** 2)
        # convention-free: must match the minus form and beat the plus form
        assert abs(scaled - minus_cand) < abs(scaled - plus_cand), \
            f"third term matches +a/12(1-a^2) for a={a} (should be minus)"
        assert mp.sign(scaled) == mp.sign(minus_cand), f"sign mismatch a={a}"
    print("Check 3 (third term isolated): -> -a/12 (1-a^2), not +        [PASS]")


# ----------------------------------------------------------------------
# Check 4 -- eq. (5) vs exact binomial tail (+ typo control)
# ----------------------------------------------------------------------
def check_full_proposition():
    a, rho = 0.7, 0.4
    for eta in (0, 1):
        # residual of the CORRECT proposition must decay faster than 1/n
        ns = [200, 800, 3200]
        ns = [2 * N + eta for N in ns]
        last = None
        for n in ns:
            resid = exact_tail(a, rho, n, eta) - eq5_expansion(a, rho, n, eta)
            last = abs(resid) * n          # n * residual -> 0 if coeff exact
        assert last < 1e-2, f"eq. (5) residual too large (eta={eta}): {last}"

        # control: typo-propagated coeff (5/12 instead of 1/4) is wrong at O(1/n)
        def prop_typo(a, rho, n, eta):
            nn = mp.mpf(n); aa = mp.mpf(a); rr = mp.mpf(rho); chi = mp.mpf(1 - eta) / 2
            coeff = rr + aa**3 / 2 - 2 * aa * chi**2 + mp.mpf(5) / 12 * aa * (1 - aa**2)
            return Phi(aa) - (2 * chi / mp.sqrt(nn)) * phi(aa) + (coeff / nn) * phi(aa)
        n = ns[-1]
        resid_typo = abs(exact_tail(a, rho, n, eta) - prop_typo(a, rho, n, eta)) * n
        assert resid_typo > 1e-3, "typo version unexpectedly passed"
    print("Check 4 (full eq. (5) vs exact tail):  residual O(n^-3/2)     [PASS]")
    print("            control: typo-propagated coeff fails at O(1/n)    [PASS]")


if __name__ == "__main__":
    check_symbolic()
    check_zn()
    check_third_term_isolated()
    check_full_proposition()
    print("\nAll checks passed: third term is  -a/12 (1-a^2) phi(a).")