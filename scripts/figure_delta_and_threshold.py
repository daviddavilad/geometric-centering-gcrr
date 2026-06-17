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


# --- Parameters (Chen §9.4 base case) ---
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N_VALUES = np.arange(25, 1001)

# Same scheme/marker setup as figure_delta_diagnostic.py so the two
# figures share visual vocabulary.
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

    # Compute N for each n (parity-aware)
    N_arr = (N_VALUES - N_VALUES % 2) // 2

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True
    )

    # Top panel: Delta_n
    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        ax_top.scatter(
            N_VALUES, out["Delta"],
            color=color, marker="o", s=size,
            alpha=0.85, label=label, edgecolors="none",
        )

    # Reference lines on top panel
    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax_top.axhline(1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.axhline(-1, color="black", linewidth=0.4, alpha=0.4, zorder=0, linestyle="--")
    ax_top.set_ylabel(r"$\Delta_n = 1 - 2\theta_n$")
    ax_top.set_ylim(-1.15, 1.15)

    # Bottom panel: J_n^+ - N
    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        threshold = out["J_strict"] - N_arr
        ax_bot.scatter(
            N_VALUES, threshold,
            color=color, marker="o", s=size,
            alpha=0.85, edgecolors="none",
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
