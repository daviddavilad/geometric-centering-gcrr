"""Phase-separated GCRR-GC scaled errors across the WAND robustness panel.

Per Chen Section 9.4 (revised draft), the WAND-style panel serves as
the robustness benchmark for the paper. This script produces one figure
per WAND case (8 cases total) showing GCRR-GC's phase-separated scaled
errors n * (C_n - C_BS) at n in [25, 500].

The figures demonstrate that the phase-wise smoothness property of
Theorem 7.1 holds uniformly across the parameter grid: for each case,
the even and odd subsequences converge to distinct stable constants.
These constants are reported quantitatively in Table 2.

Placement: online appendix (per Chen Section 9.4 revised draft, the
WAND panel is appendix material; the main paper's headline figures
are at K in {95, 105}).

Generates 8 figures:
    figures/wand_scaled_errors_case_1.{pdf,png}  -- K=35, sigma=0.20, T=0.25
    figures/wand_scaled_errors_case_2.{pdf,png}  -- K=35, sigma=0.20, T=0.50
    figures/wand_scaled_errors_case_3.{pdf,png}  -- K=35, sigma=0.40, T=0.25
    figures/wand_scaled_errors_case_4.{pdf,png}  -- K=35, sigma=0.40, T=0.50
    figures/wand_scaled_errors_case_5.{pdf,png}  -- K=45, sigma=0.20, T=0.25
    figures/wand_scaled_errors_case_6.{pdf,png}  -- K=45, sigma=0.20, T=0.50
    figures/wand_scaled_errors_case_7.{pdf,png}  -- K=45, sigma=0.40, T=0.25
    figures/wand_scaled_errors_case_8.{pdf,png}  -- K=45, sigma=0.40, T=0.50

Common parameters: S = 40, r = 0.06.
"""
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcrr import schemes
from gcrr.errors import phase_separated_errors
from gcrr.plotting import SCHEME_COLORS, SCHEME_LABELS, set_style


# --- Fixed WAND parameters ---
S = 40.0
R = 0.06
K_VALUES = [35.0, 45.0]
SIGMA_VALUES = [0.20, 0.40]
T_VALUES = [0.25, 0.50]

# --- n range: matches main paper figures (Chen §9.4: "n=25..500 for figures") ---
N_VALUES = np.arange(25, 501)


def build_cases() -> list[dict]:
    """Enumerate the 8 WAND cases; same ordering as wand_robustness_data.py."""
    cases = []
    for case_id, (K, sigma, T) in enumerate(
        product(K_VALUES, SIGMA_VALUES, T_VALUES), start=1
    ):
        cases.append({
            "case_id": case_id,
            "K": K,
            "sigma": sigma,
            "T": T,
        })
    return cases


def make_figure(case: dict, out_dir: Path):
    """One two-panel figure for a single WAND case."""
    K, sigma, T = case["K"], case["sigma"], case["T"]

    # Compute GCRR-GC phase-separated errors
    out = phase_separated_errors(S, K, T, R, sigma, schemes.gcrr_gc, N_VALUES, "call")

    fig, (ax_even, ax_odd) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    color = SCHEME_COLORS["gcrr_gc"]
    label = SCHEME_LABELS["gcrr_gc"]

    ax_even.scatter(
        out.n_even, out.scaled_err_even,
        color=color, marker="o", s=8,
        alpha=0.85, label=label, edgecolors="none",
    )
    ax_odd.scatter(
        out.n_odd, out.scaled_err_odd,
        color=color, marker="o", s=8,
        alpha=0.85, edgecolors="none",
    )

    for ax in (ax_even, ax_odd):
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlabel(r"$n$ (periods)")

    ax_even.set_title(r"Even $n$")
    ax_odd.set_title(r"Odd $n$")
    ax_even.set_ylabel(r"$n(C_n - C^{\mathrm{BS}})$")
    ax_even.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"WAND case {case['case_id']}: "
        f"$S={S:.0f}$, $K={K:.0f}$, $T={T}$, $r={R}$, $\\sigma={sigma}$",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"wand_scaled_errors_case_{case['case_id']}"
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

    for case in build_cases():
        make_figure(case, out_dir)


if __name__ == "__main__":
    main()
