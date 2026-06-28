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


# --- Parameters (Chen §9.4 primary non-ATM case) ---
S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20  # OTM
N_VALUES = np.arange(25, 501)  # Chen §9.4: n=25..500 for figures

# Plot in this order so overlapping schemes are visible: bottom to top.
# CRR plotted first (background) since it spans the widest range;
# the centered schemes drawn over it with distinct markers.
SCHEMES = [
    ("crr",          schemes.crr,         20),  # background
    ("tian_1999",    schemes.tian_1999,   10),
    ("joshi_2009",   schemes.joshi_2009,  10),
    ("gcrr_gc",      schemes.gcrr_gc,      1),  # foreground
]
# Format: (key, fn, marker_size)


def main():
    set_style()

    results = {
        key: strike_location_sweep(S, K, T, r, sigma, fn, N_VALUES)
        for key, fn, _ in SCHEMES
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        ax.scatter(
            N_VALUES, out["Delta"],
            color=color, marker="o", s=size,
            alpha=0.85, label=label, edgecolors="none",
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
