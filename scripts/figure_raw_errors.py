"""Raw call-price errors for three GCRR-family schemes.

For each scheme, plots the raw error C_n - C_BS against n (no scaling).
Companion to scripts/figure_scaled_errors.py: the scaled-error figure
shows convergence rates by dividing by 1/n; this figure shows the
raw error magnitudes.

The contrast Chen (Section 9.1) is after: a stretch parameter alone
does not produce smooth convergence; the stretch must be chosen by
the geometric-center condition. The fixed-stretch GCRR scheme (with
lambda = 1.1) is included as the control. CRR (lambda = 1) is the
baseline. GCRR-GC (lambda chosen by the geometric-center equation)
is the proposed construction.

Generates one figure (Chen §9.4 base case):
    figures/raw_errors_atm.{pdf,png}  -- K = 100 (ATM)

Test case (Chen §9.4 base case): European call.
    S = 100, K = 100, T = 1, r = 0.05, sigma = 0.20.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.black_scholes import bs_call
from gcrr.pricer import binomial_price
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Fixed parameters ---
S, T, r, sigma = 100.0, 1.0, 0.05, 0.20  # Chen §9.4 base case
N_VALUES = np.arange(25, 1001)

SCHEMES = [
    ("crr",          schemes.crr,         20),  # background
    ("gcrr_fixed",   schemes.gcrr_fixed,  10),
    ("gcrr_gc",      schemes.gcrr_gc,      3),  # foreground
]
# Format: (key, fn, marker_size)


def make_figure(K: float, moneyness_tag: str, out_dir: Path):
    """Generate the single-panel raw-error figure at a given strike."""
    C_BS = bs_call(S, K, T, r, sigma)

    results = {}
    for key, scheme_fn, _ in SCHEMES:
        errs = np.empty(len(N_VALUES))
        for i, n in enumerate(N_VALUES):
            params = scheme_fn(S, K, T, r, sigma, int(n))
            price = binomial_price(S, K, T, r, params, "call", "european")
            errs[i] = price - C_BS
        results[key] = errs

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for key, _, size in SCHEMES:
        color = SCHEME_COLORS[key]
        label = SCHEME_LABELS[key]
        ax.scatter(
            N_VALUES, results[key],
            color=color, marker="o", s=size,
            alpha=0.85, label=label, edgecolors="none",
        )

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel(r"$n$ (periods)")
    ax.set_ylabel(r"$C_n - C^{\mathrm{BS}}$")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Raw call-price errors: {moneyness_tag} "
        f"($S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={r}$, $\\sigma={sigma}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"raw_errors_{moneyness_tag.lower()}"
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
