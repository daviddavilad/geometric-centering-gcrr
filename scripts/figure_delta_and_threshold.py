r"""Strike-location diagnostics: Delta_n and J_n^+ - N across four schemes.

Two stacked panels showing the geometric structure of the strike's
position in the terminal log-lattice:

Top panel: Delta_n = 1 - 2*theta_n, the continuous centered location
coordinate. Reference lines at Delta = 0 (geometric midpoint) and
Delta = +/-1 (strike on a node).

Bottom panel: J_n^+ - N, the integer-valued strict upper threshold
relative to the central index N. Chen (Section 9.2) claims that
GCRR-GC produces the "unified strict threshold j >= N+1" -- i.e.,
J_n^+ - N = 1 for all n, regardless of parity. The figure verifies
this claim alongside the contrast with CRR (whose J_n^+ - N varies
with n) and the centered designs (Tian, Joshi) which also satisfy
the unified threshold within their respective parity restrictions.

This is the geometric mechanism behind the convergence behavior:
CRR's threshold chaos drives its scaled-error oscillation; GCRR-GC's
uniform threshold produces phase-wise smooth convergence.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.diagnostics import strike_location_sweep
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Parameters ---
S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20  # OTM
N_VALUES = np.arange(50, 501)

# Same scheme/marker setup as figure_delta_diagnostic.py so the two
# figures share visual vocabulary.
SCHEMES = [
    ("crr",          schemes.crr,         "o", 8,  True,  0.5),   # filled circle
    ("tian_1999",    schemes.tian_1999,   "s", 60, False, 0.9),   # open square
    ("joshi_2009",   schemes.joshi_2009,  "^", 60, False, 0.9),   # open triangle
    ("gcrr_gc",      schemes.gcrr_gc,     "D", 18, True,  0.9),   # filled diamond
]


def main():
    set_style()

    results = {
        key: strike_location_sweep(S, K, T, r, sigma, fn, N_VALUES)
        for key, fn, *_ in SCHEMES
    }

    # Compute N for each n (parity-aware)
    N_arr = (N_VALUES - N_VALUES % 2) // 2

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True
    )

    # Top panel: Delta_n
    for key, _, marker, size, filled, alpha in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        if filled:
            ax_top.scatter(
                N_VALUES, out["Delta"],
                color=color, marker=marker, s=size,
                alpha=alpha, label=label, edgecolors="none",
            )
        else:
            ax_top.scatter(
                N_VALUES, out["Delta"],
                facecolors="none", edgecolors=color, marker=marker, s=size,
                linewidths=1.0, alpha=alpha, label=label,
            )

    # Reference lines on top panel
    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax_top.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.axhline(-1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.set_ylabel(r"$\Delta_n = 1 - 2\theta_n$")
    ax_top.set_ylim(-1.15, 1.15)

    # Bottom panel: J_n^+ - N
    for key, _, marker, size, filled, alpha in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        threshold = out["J_strict"] - N_arr
        if filled:
            ax_bot.scatter(
                N_VALUES, threshold,
                color=color, marker=marker, s=size,
                alpha=alpha, edgecolors="none",
            )
        else:
            ax_bot.scatter(
                N_VALUES, threshold,
                facecolors="none", edgecolors=color, marker=marker, s=size,
                linewidths=1.0, alpha=alpha,
            )

    # Reference line at the "unified threshold" value of 1
    ax_bot.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_bot.set_xlabel(r"$n$ (periods)")
    ax_bot.set_ylabel(r"$J_n^+ - N$")

    # Legend on top panel only (cleaner)
    ax_top.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        f"Strike-location diagnostics  "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 0.97))

    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / "delta_and_threshold.pdf"
    out_png = out_dir / "delta_and_threshold.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


if __name__ == "__main__":
    main()
