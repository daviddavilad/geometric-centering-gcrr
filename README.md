# Geometric Centering in Generalized Cox–Ross–Rubinstein Trees

[![Tests](https://github.com/daviddavilad/geometric-centering-gcrr/actions/workflows/tests.yml/badge.svg)](https://github.com/daviddavilad/geometric-centering-gcrr/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23866510.svg)](https://doi.org/10.5281/zenodo.23866510)

Code for

> Chen, H.-C. and Dávila, D., Geometric centering in generalized Cox–Ross–Rubinstein trees. Working Paper, University of New Mexico, 2026.

The paper proposes GCRR-GC, a binomial tree whose stretch parameter places the strike at the geometric center of the terminal lattice for both even and odd numbers of steps (Proposition 2.1). This repository implements GCRR-GC and five benchmark trees, checks the paper's closed-form results numerically, and regenerates every table and figure in the paper and its Online Appendix.

## Quick start

Requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/daviddavilad/geometric-centering-gcrr.git
cd geometric-centering-gcrr
./reproduce.sh
```

`reproduce.sh` installs the locked dependencies, runs the test suite, and regenerates all tables and figures. It takes about 30 seconds on a desktop.

## Paper outputs

| Paper | Script | Output |
|---|---|---|
| Table 1 | `scripts/table_1_main.py` | `tables/table_1_main.{csv,md}` |
| Table 2 | `scripts/table_2_phase_gap.py` | `tables/table_2_phase_gap.{csv,md}` |
| Table 3 | `scripts/table_3_extrapolation.py` | `tables/table_3_extrapolation.{csv,md}` |
| Table OA.1 | `scripts/wand_robustness_data.py`, then `scripts/table_oa1_rms.py` | `tables/table_oa1_rms.{csv,md}` |
| Figure 1 | `scripts/figure_raw_errors.py` | `figures_paper/raw_errors_{itm,otm}.{pdf,png}` |
| Figure 2 | `scripts/figure_scaled_errors.py` | `figures_paper/scaled_errors_{itm,otm}.{pdf,png}` |
| Figure 3 | `scripts/figure_delta_and_threshold.py` | `figures_paper/delta_and_threshold.{pdf,png}` |
| Figure OA.1 | `scripts/figure_digital_errors.py` | `figures_paper/digital_errors_{itm,otm}.{pdf,png}` |
| Figure OA.2 | `scripts/figure_digital_corrected_errors.py` | `figures_paper/digital_corrected_errors_{itm,otm}.{pdf,png}` |
| Eq. (5) and Appendix B | `scripts/verify_tail_expansion.py`, `scripts/verify_appendix_B_third_term.py` | `scripts/*_output.txt` |

Files ending in `_itm` use K = 95 and files ending in `_otm` use K = 105. Each script also runs on its own, for example `uv run python scripts/figure_scaled_errors.py`. Table OA.1 reads `data/wand_results.csv`, so run `wand_robustness_data.py` first. Spreadsheet copies of the tables (`tables/*.xlsx`) are written alongside the CSV files but not tracked.

## Numerical precision

- **Tables 2 and 3 use 40-digit arithmetic** (mpmath). Table 2 fits n up to 100,000, and Richardson extrapolation in Table 3 amplifies rounding error by a factor of about n, so double precision would contaminate both. These two tables come out identical on any machine.
- **Everything else uses double precision.** On another operating system, the figures, Table 1, Table OA.1 and the eq. (5) check can differ in trailing digits, with no visible change. The committed outputs were generated on Linux.

## Package layout

The `gcrr` package lives in `src/gcrr`:

| Module | Contents |
|---|---|
| `schemes.py` | Tree parameterizations: CRR, fixed-stretch GCRR, GCRR-XPC (Chung and Shih 2007), GCRR-GC, Tian (1999), Joshi (2009) |
| `pricer.py` | Backward-induction pricer and the two-tail call price of Proposition 3.1 |
| `diagnostics.py` | Strike location in the terminal lattice: θ_n, Δ_n and the exercise threshold m_n (Section 2.1) |
| `theory.py` | Closed forms: ρ_B and ρ_S (Appendix A.3), A_η (eq. 8), the phase gap (Corollary 3.3), digital coefficients (Appendix C.3) |
| `digital.py` | Strict and half-node digital prices (Section 4) |
| `extrapolation.py` | Same-phase and adjacent Richardson extrapolation, phase adjustment (Section 3.3) |
| `errors.py` | Phase-separated errors and least-squares fits of the leading constant |
| `high_precision.py` | 40-digit GCRR-GC and Black–Scholes call prices |
| `black_scholes.py` | Black–Scholes call and digital prices |
| `plotting.py` | Shared figure style |

## Tests

```bash
uv run pytest
```

The tests check the closed forms in `theory.py` against the exact tree, the two-tail pricer against backward induction, fitted A_η against Theorem 3.2, the convergence rates of the extrapolation schemes, the 40-digit prices against double precision, and the strike-location properties of Proposition 2.1.

## Citation

Please cite the paper. If you use the code, please also cite the software release.

```bibtex
@techreport{chen2026geometric,
  author      = {Chen, Hsuan-Chi and D{\'a}vila, David},
  title       = {Geometric Centering in Generalized {Cox--Ross--Rubinstein} Trees},
  institution = {University of New Mexico},
  type        = {Working Paper},
  year        = {2026}
}

@software{chen2026geometric_code,
  author    = {Chen, Hsuan-Chi and D{\'a}vila, David},
  title     = {Geometric Centering in Generalized {Cox--Ross--Rubinstein} Trees: code},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23866511},
  url       = {https://doi.org/10.5281/zenodo.23866511}
}
```

## License

MIT. See [LICENSE](LICENSE).