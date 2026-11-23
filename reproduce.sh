#!/usr/bin/env bash
# Regenerate every result and number of the paper. Run from this directory with numpy/scipy from requirements.txt.
# Total time on a laptop: about 4 h with 8 workers (the QAOA simulations dominate).
set -euo pipefail
PY=${PYTHON:-python}
export OMP_NUM_THREADS=1
$PY scripts/verify_circuits.py               # Sec. 3: gate-level check, 108 configurations (~1 s)
$PY scripts/run_resources.py                 # Table 1, Sec. 5.1: resources for n = 16..256 (~2 s)
$PY scripts/run_preservation.py enumerate    # Table 4: optimum preservation, n = 12..24 (~1 min)
for C in tradeoff.json tradeoff_regular3.json; do    # QAOA on 48 SK + 48 weighted 3-regular MaxCut instances (~2 h each)
  $PY scripts/run_tradeoff.py --config $C --workers "${WORKERS:-8}"
  $PY scripts/run_preservation.py tradeoff $C       # preservation vs P_ret on the same instances (~15 s)
done
$PY scripts/make_numbers.py                  # results/numbers.tex
(cd figures && xelatex -interaction=nonstopmode fig_tradeoff.tex >/dev/null && echo "built figures/fig_tradeoff.pdf")
