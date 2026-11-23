#!/usr/bin/env bash
# Regenerate every result and number of the paper. Run from this directory with numpy/scipy from requirements.txt.
# Total time on a laptop: about 1.3 h with 6 workers (the QAOA simulations dominate).
set -euo pipefail
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1
$PY scripts/verify_circuits.py               # Sec. 3: gate-level check, 108 configurations (~1 s)
$PY scripts/run_resources.py                 # Table 1, Sec. 5.1: resources for n = 16..256 (~2 s)
$PY scripts/run_preservation.py enumerate    # Table 4: optimum preservation, n = 12..24 (~1 min)
$PY scripts/run_tradeoff.py --workers "${WORKERS:-6}"   # Tables 2-3, Fig. 2: QAOA on 48 SK instances (~1.3 h)
$PY scripts/run_preservation.py tradeoff     # Sec. 5.3: preservation vs P_ret on the same instances (~15 s)
$PY scripts/make_numbers.py                  # results/numbers.tex
(cd figures && xelatex -interaction=nonstopmode fig_tradeoff.tex >/dev/null && echo "built figures/fig_tradeoff.pdf")
