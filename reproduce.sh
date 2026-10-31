#!/usr/bin/env bash
# Regenerate every table and figure in the paper, plus the outputs of the
# two verification scripts, from a fresh clone.
#
# Usage: ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

# Fixed PDF timestamps, so reruns produce identical files
export SOURCE_DATE_EPOCH=0

uv sync --frozen
uv run pytest -q

# Tables (Table OA.1 reads the WAND panel data, so build that first)
uv run python scripts/wand_robustness_data.py
uv run python scripts/table_oa1_rms.py
uv run python scripts/table_1_main.py
uv run python scripts/table_2_phase_gap.py
uv run python scripts/table_3_extrapolation.py

# Figures
for fig in raw_errors scaled_errors delta_and_threshold digital_errors digital_corrected_errors; do
    uv run python "scripts/figure_${fig}.py"
done

# Numerical checks of eq. (5) and Appendix B
uv run python scripts/verify_tail_expansion.py > scripts/verify_tail_expansion_output.txt
uv run python scripts/verify_appendix_B_third_term.py > scripts/verify_appendix_B_third_term_output.txt

echo "Done: tables/, figures_paper/ and scripts/*_output.txt regenerated."