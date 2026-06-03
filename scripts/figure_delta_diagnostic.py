r"""Strike-location coordinate Delta_n across four schemes.

Delta_n in (-1, 1] is the centered log-location coordinate of the strike
within the terminal binomial lattice.
    Delta = 0  <=> strike at geometric midpoint of two adjacent terminal nodes
    Delta = 1  <=> strike exactly on a terminal node
    Delta in (-1, 1)\{0,1}: strike between nodes, off-center

Observations from this figure:
    - CRR: Delta drifts slowly in saw-tooth bands. Within each band, the
      fractional part of log(K/S0) / (2 sigma sqrt(T/n)) drifts roughly
      linearly with n; at the same n values where Tian's j_0 transitions,
      CRR's drift wraps around. This is the geometric mechanism behind
      both CRR's price oscillation and Tian's step-discontinuities.
    - Tian (1999): Delta locks to 1 most of the time, with brief excursions
      to -1 at j_0 transitions. Same n-pattern as CRR's wraparounds.
    - Joshi (2009): Delta = 0 for every odd n (its only domain).
    - GCRR-GC: Delta = 1 for even n, Delta = 0 for odd n. Parity-locked.

The figure uses markers only (no connecting lines) because connecting
alternating-parity data points produces spurious vertical sweeps.
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

# Plot in this order so overlapping schemes are visible: bottom to top.
# CRR plotted first (background) since it spans the widest range;
# the centered schemes drawn over it with distinct markers.
SCHEMES = [
    ("crr",          schemes.crr,         "o", 8,  True,  0.5),   # filled circle
    ("tian_1999",    schemes.tian_1999,   "s", 60, False, 0.9),   # open square (no fill)
    ("joshi_2009",   schemes.joshi_2009,  "^", 60, False, 0.9),   # open triangle
    ("gcrr_gc",      schemes.gcrr_gc,     "D", 18, True,  0.9),   # filled diamond
]
# Format: (key, fn, marker, size, filled, alpha)


def main():
    set_style()

    results = {
    key: strike_location_sweep(S, K, T, r, sigma, fn, N_VALUES)
    for key, fn, *_ in SCHEMES  # <-- *_ catches everything after fn
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for key, _, marker, size, filled, alpha in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        if filled:
            ax.scatter(
                N_VALUES, out["Delta"],
                color=color, marker=marker, s=size,
                alpha=alpha, label=label, edgecolors="none",
            )
        else:
            # Open markers: visible outline, no fill, so stacked points
            # show through.
            ax.scatter(
                N_VALUES, out["Delta"],
                facecolors="none", edgecolors=color, marker=marker, s=size,
                linewidths=1.0, alpha=alpha, label=label,
            )

    # Reference lines at the geometrically meaningful values of Delta
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax.axhline(-1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")

    ax.set_xlabel(r"$n$ (periods)")
    ax.set_ylabel(r"$\Delta_n = 1 - 2\theta_n$")
    ax.set_ylim(-1.15, 1.15)

    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        f"Strike-location coordinate $\\Delta_n$  "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.85, 0.97))

    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / "delta_diagnostic.pdf"
    out_png = out_dir / "delta_diagnostic.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


if __name__ == "__main__":
    main()
