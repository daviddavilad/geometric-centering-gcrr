r"""Strike-location diagnostics: Delta_n and m_n - N across four schemes.

Two stacked panels showing the geometric structure of the strike's
position in the terminal log-lattice:

One row of panels per scheme (CRR, Tian, GCRR-GC): Delta_n on the left,
m_n - N on the right.

Generates:
    figures_paper/delta_and_threshold.{pdf,png}
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from gcrr import schemes
from gcrr.diagnostics import strike_location_sweep
from gcrr.plotting import (
    SCHEME_LABELS,
    plot_paper_series,
    set_style_paper,
)

matplotlib.use("Agg")  # same layout on every OS

# --- Parameters (Section 5.1 primary non-ATM case) ---
S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.06, 0.20  # OTM
N_VALUES = np.arange(25, 501)  # Section 5.1: n=25..500 for figures

# One row of panels per scheme. Joshi is omitted: here it repeats
# GCRR-GC's odd phase exactly.
SCHEMES = [
    ("crr", schemes.crr),
    ("tian_1999", schemes.tian_1999),
    ("gcrr_gc", schemes.gcrr_gc),
]


def _compute_results():
    """Compute strike-location diagnostics for each scheme."""
    return {
        key: strike_location_sweep(S, K, T, r, sigma, fn, N_VALUES)
        for key, fn in SCHEMES
    }


def make_figure_paper(results: dict, out_dir: Path):
    """One row per scheme: Delta_n on the left, m_n - N on the right."""
    set_style_paper()

    N_arr = (N_VALUES - N_VALUES % 2) // 2
    fig, axes = plt.subplots(len(SCHEMES), 2, figsize=(8, 6), sharex=True, sharey="col")

    for row, (key, _) in zip(axes, SCHEMES):
        ax_delta, ax_m = row
        out = results[key]
        plot_paper_series(ax_delta, N_VALUES, out["Delta"], key, lines=False)
        plot_paper_series(ax_m, N_VALUES, out["m"] - N_arr, key, lines=False)

        # Solid light gray instead of transparency (journal rule)
        for level, ls in [(0, "-"), (1, "--"), (-1, "--")]:
            ax_delta.axhline(level, color="0.6", linewidth=0.4, linestyle=ls, zorder=0)
        ax_m.axhline(1, color="0.6", linewidth=0.4, linestyle="--", zorder=0)

        ax_delta.set_ylabel(SCHEME_LABELS[key])
        ax_delta.set_ylim(-1.15, 1.15)
        ax_delta.set_yticks([-1, 0, 1])
        ax_m.yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0, 0].set_title(r"$\Delta_n = 1 - 2\theta_n$")
    axes[0, 1].set_title(r"$m_n - N$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$n$ (periods)")

    fig.tight_layout()

    stem = "delta_and_threshold"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def main():
    out_dir = Path(__file__).parent.parent / "figures_paper"
    out_dir.mkdir(exist_ok=True)
    make_figure_paper(_compute_results(), out_dir)


if __name__ == "__main__":
    main()
