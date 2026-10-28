"""Phase-aware extrapolation and phase alignment (Section 3.3).

Theorem 3.2 gives C_{N,eta} = C_BS + A_eta / n + O(n^{-3/2}) with A_0 != A_1
in general. Consequences tested in Table 3:

    richardson_same_phase   n and n+2 share a phase, so A_eta/n cancels.
    richardson_adjacent     n and n+1 differ in phase; the leading terms
                            leave an O(1) error equal to +/-(A_1 - A_0).
    phase_adjust            C^PA_n = C_{N,eta} - eta * (A_1 - A_0) / n puts
                            both phases on the even-phase track A_0 / n.
"""

from __future__ import annotations


def richardson_same_phase(C_n: float, C_n2: float, n: int) -> float:
    """Same-phase Richardson extrapolation from n and n+2 (Section 3.3):
        ((n + 2) C_{n+2} - n C_n) / 2.
    If C_m = C_BS + A/m exactly, this returns C_BS.
    """
    return ((n + 2) * C_n2 - n * C_n) / 2


def richardson_adjacent(C_n: float, C_n1: float, n: int) -> float:
    """Adjacent-count Richardson extrapolation from n and n+1:
        (n + 1) C_{n+1} - n C_n.
    Under Theorem 3.2 this returns C_BS + A_{eta(n+1)} - A_{eta(n)}, i.e.
    C_BS + (A_1 - A_0) for even n and C_BS - (A_1 - A_0) for odd n.
    """
    return (n + 1) * C_n1 - n * C_n


def phase_adjust(C: float, n: int, gap: float) -> float:
    """Phase-adjusted price (Section 3.3):  C^PA_n = C_n - eta * gap / n,
    eta = n mod 2, gap = A_1 - A_0 from Corollary 3.3 (gcrr.theory.phase_gap).
    """
    return C - (n % 2) * gap / n
