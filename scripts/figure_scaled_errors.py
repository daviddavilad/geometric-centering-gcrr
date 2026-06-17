"""Phase-separated scaled errors for European call schemes.

For each scheme, plots n * (C_n - C_BS) against n, separated by parity
(even n vs odd n). Under phase-wise smooth convergence (Theorem 7.1),
each phase should approach a horizontal line; CRR (un-centered) shows
oscillation that doesn't flatten.

Generates two figures:
    figures/scaled_errors_itm.{pdf,png}  -- K = 95  (ITM)
    figures/scaled_errors_otm.{pdf,png}  -- K = 105 (OTM)

Test case (Chen §9.4 base case): European call.
    S = 100, K = 100, T = 1, r = 0.05, sigma = 0.20.

Schemes shown: CRR (gray baseline), Tian (1999), Joshi (2009), GCRR-GC.
GCRR-fixed and GCRR-XPC omitted for legibility of the headline figure;
they appear in a separate appendix figure (TODO).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.errors import phase_separated_errors
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Fixed parameters ---
S, T, r, sigma = 100.0, 1.0, 0.05, 0.20  # Chen §9.4 base case
N_VALUES = np.arange(25, 1001)

SCHEMES = [
    ("crr",          schemes.crr,         20),  # background
    ("tian_1999",    schemes.tian_1999,   10),
    ("joshi_2009",   schemes.joshi_2009,  10),
    ("gcrr_gc",      schemes.gcrr_gc,      1),  # foreground
]
# Format: (key, fn, marker_size)


def make_figure(K: float, moneyness_tag: str, out_dir: Path):
    results = {
        key: phase_separated_errors(S, K, T, r, sigma, fn, N_VALUES, "call")
        for key, fn, _ in SCHEMES
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax_even, ax_odd = axes

    for key, _, size in SCHEMES:
        out = results[key]
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        ax_even.scatter(
            out.n_even, out.scaled_err_even,
            color=color, marker="o", s=size,
            alpha=0.85, label=label, edgecolors="none",
        )
        ax_odd.scatter(
            out.n_odd, out.scaled_err_odd,
            color=color, marker="o", s=size,
            alpha=0.85, edgecolors="none",
        )

    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title("Even $n$")
    ax_odd.set_title("Odd $n$")
    ax_even.set_ylabel(r"$n \cdot (C_n - C^{\mathrm{BS}})$")

    # Legend below both panels: one row, four entries, centered
    handles, labels = ax_even.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        f"Phase-separated scaled error: {moneyness_tag} "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    stem = f"scaled_errors_{moneyness_tag.lower()}"
    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_pdf} (+ .png)")
    plt.close(fig)


def main():
    set_style()
    out_dir = Path(__file__).parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    make_figure(K=100.0, moneyness_tag="ATM", out_dir=out_dir)


if __name__ == "__main__":
    main()
