# GCRR-GC Research

Private research repository. Joint work with Dr. Hsuan-Chi Chen (UNM).
Contains preliminary, unpublished results.

**Do not make public or circulate** until the paper is submitted for
journal publication.

## Paper

*Lattice Geometry, Strike-Location Error, and Parity-Free Smooth
Convergence in Binomial Option Pricing* by Hsuan-Chi Chen (UNM) and
David Davila (UNM).

The paper introduces GCRR-GC, a parity-free centered binomial scheme
for European option pricing, and proves phase-wise smooth O(1/n)
convergence (Theorem 7.1) with a closed-form phase gap (Corollary 7.2).

## Structure

- `src/gcrr/` — core library (7 modules)
- `scripts/` — figure, table, and verification scripts
- `tests/` — correctness tests (39 tests, all passing)
- `figures/` — generated PDF and PNG figures
- `tables/` — generated CSV and Markdown tables
- `data/` — numerical results consumed by table scripts
- `verification/` — handwritten math verification notes (iPad scans)

## Library modules (`src/gcrr/`)

- `schemes.py` — binomial scheme parameterizations (CRR, GCRR-GC,
  fixed-stretch GCRR, GCRR-XPC (Chung-Shih 2007), Tian (1999),
  Joshi (2009)). Each returns a `TreeParameters` object.
- `pricer.py` — generic backward-induction pricer. Scheme-agnostic:
  consumes any `TreeParameters` and returns European or American prices.
- `black_scholes.py` — Black-Scholes reference prices for calls, puts,
  and strict digital options.
- `digital.py` — strict and half-node corrected digital pricing on
  binomial trees (Section 9.3).
- `diagnostics.py` — strike-location diagnostics: continuous strike
  index $\gamma_n$, weak/strict upper thresholds $J_n$ and $J_n^+$,
  centered log-location coordinate $\Delta_n$.
- `errors.py` — phase-separated error computation for scaled-error
  figures.
- `plotting.py` — shared plot style, scheme color and label conventions.

## Reproducing the numerical section

**Requirements:** Python 3.11+, `uv` for dependency management.

```bash
uv sync            # install dependencies
uv run pytest -v   # run tests (should show 39 passing)
```

Run figures (each script produces PDF and PNG in `figures/`):

```bash
uv run python scripts/figure_raw_errors.py
uv run python scripts/figure_scaled_errors.py
uv run python scripts/figure_delta_diagnostic.py
uv run python scripts/figure_delta_and_threshold.py
uv run python scripts/figure_digital_errors.py
uv run python scripts/figure_digital_corrected_errors.py
```

Run tables (each script produces CSV and Markdown in `tables/`):

```bash
uv run python scripts/wand_robustness_data.py       # ~10s; produces data/wand_results.csv
uv run python scripts/table_1_representative_errors.py
uv run python scripts/table_2_phase_gap.py
```

Run verification scripts (empirical checks of specific claims):

```bash
uv run python scripts/verify_appendix_B_third_term.py
uv run python scripts/verify_phase_gap_wand.py
uv run python scripts/verify_proposition_B1.py
uv run python scripts/verify_phase_gap.py
```

## Figure and table mapping

Per Chen §9.5 (revised draft):

| Paper item | Script | Output |
|:-----------|:-------|:-------|
| Figure 1 (raw errors) | `figure_raw_errors.py` | `raw_errors_itm.pdf`, `raw_errors_otm.pdf` |
| Figure 2 (scaled errors) | `figure_scaled_errors.py` | `scaled_errors_itm.pdf`, `scaled_errors_otm.pdf` |
| Figure 3 (Δ + threshold) | `figure_delta_and_threshold.py` | `delta_and_threshold.pdf` |
| Figure 4 (strict digital) | `figure_digital_errors.py` | `digital_errors_itm.pdf`, `digital_errors_otm.pdf` |
| Figure 5 (corrected digital) | `figure_digital_corrected_errors.py` | `digital_corrected_errors_itm.pdf`, `digital_corrected_errors_otm.pdf` |
| Table 1a (absolute errors) | `table_1_representative_errors.py` | `table_1a_absolute_errors.csv/md` |
| Table 1b (scaled errors) | `table_1_representative_errors.py` | `table_1b_scaled_errors.csv/md` |
| Table 2 (phase gap) | `table_2_phase_gap.py` | `table_2_phase_gap.csv/md` |

Primary non-ATM case per Chen §9.4: $S_0 = 100$, $X \in \{95, 105\}$,
$T = 0.5$, $\sigma = 0.20$, $r = 0.06$.

WAND-style robustness panel per Chen §9.4: $S_0 = 40$, $X \in \{35, 45\}$,
$\sigma \in \{0.20, 0.40\}$, $T \in \{0.25, 0.50\}$, $r = 0.06$.

ATM diagnostic figures (K=100, T=1, r=0.05) are archived under
`figures/*_atm.{pdf,png}`. Per Chen §9.4 revised draft, ATM is a
degenerate case moved to the online appendix.

## Test suite (`tests/`)

- `test_schemes.py` — scheme parameterization correctness, convergence
  to Black-Scholes, parity restrictions.
- `test_diagnostics.py` — strike-location coordinates, floating-point
  robustness at exact-alignment cases, unified strict threshold claim
  of GCRR-GC.
- `test_errors.py` — phase-separated error infrastructure.
- `test_digital.py` — strict and half-node corrected digital pricing,
  including the exact symmetry $\Psi^*_{N,0}(1/2) = \Psi^*_{N,1}(1/2) = 1/2$
  and the O(1/√n) central-atom discrepancy characterization.

Run all tests: `uv run pytest -v`.
