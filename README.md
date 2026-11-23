# Weight quantization for Hamming-weight phasing in weighted QAOA

Code and data for the paper by Do Duc Long and Nguyen Van Duy (2026; full citation to be added on publication).

On a fault-tolerant computer each arbitrary-angle rotation costs tens of $T$ gates, and the phase separator of a
weighted QAOA needs one rotation per term because every weight gives a different angle. Rounding each weight to a sign
times a $b$-bit magnitude leaves at most $2^b-1$ distinct angles; all terms with the same magnitude are then applied
together with Hamming-weight phasing (HWP), and the sign costs one Clifford gate. Samples are always scored with the
original cost, so the price of rounding is a loss of QAOA solution quality, which this code measures.

## Layout

```
wqhwp/            library
  instances.py    SK and weighted 3-regular MaxCut instances, exact energies, optimum-preservation test
  quantize.py     BP_b (sign + b-bit magnitude), MAGKM_b, PRUNE_b, CLUSTER_k (optimal 1D k-means), GRID_k
  qaoa.py         statevector QAOA, L-BFGS-B with INTERP initialization
  resources.py    rotations / Toffolis / ancillas / T-count of every construction, cost models M1 and M2
  circuits.py     reversible gate-level simulation used to verify the constructions
scripts/          one script per result (see table below)
configs/          parameters of each script (instance sizes, seeds, arms, batch sizes)
results/          raw outputs (JSON) and numbers.tex, the LaTeX macros quoted in the paper
figures/          fig_tradeoff.tex (Fig. 2), built from results/numbers.tex
```

## Reproducing the paper

Python 3.14 with the versions in `requirements.txt`:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
./reproduce.sh                      # everything, about 4 h with 8 workers on a laptop
```

| script | paper | output | time |
|---|---|---|---|
| `scripts/verify_circuits.py` | Sec. 3, gate-level check of 108 configurations | `results/verify.json` | ~1 s |
| `scripts/run_resources.py` | Table 1, Sec. 5.1 (n = 16..256) | `results/resources.json` | ~2 s |
| `scripts/run_preservation.py enumerate` | Table 4 (n = 12..24, exhaustive) | `results/preservation.json` | ~1 min |
| `scripts/run_tradeoff.py --config tradeoff.json` | Tables 2-3, Fig. 2 (48 SK instances, p = 1..3) | `results/tradeoff.json` | ~2 h |
| `scripts/run_tradeoff.py --config tradeoff_regular3.json` | weighted MaxCut on 3-regular graphs (48 instances) | `results/tradeoff_regular3.json` | ~2 h |
| `scripts/run_preservation.py tradeoff <config>` | Sec. 5.3 (preservation vs. P_ret) | `results/preservation_<output>` | ~15 s |
| `scripts/make_numbers.py` | all numbers in the text and tables | `results/numbers.tex` | ~10 s |

Run the scripts from this directory. Each JSON output records the SHA-256 of the script and config that produced it
and the Python / NumPy / SciPy versions. With `OMP_NUM_THREADS=1` and the library versions above, the outputs are
reproduced bit for bit on the machine they were produced on; on other platforms (different BLAS) the optimizer may
reach slightly different local optima, which changes the numbers in the last digits only.

## Definitions used in the code

- Instances: SK, complete graph, $w_{ij}\sim\mathcal N(0,1)$; weighted MaxCut on random 3-regular graphs,
  $w_e\sim U(0,1)$. Instance seeds are given in `configs/`.
- BP$_b$: $\hat w_e = s_e\,d\,m_e$, $m_e\in\{0,\dots,2^b-1\}$, step $d$ minimizing $\lVert w-\hat w\rVert_2$ on a grid.
- MAGKM$_b$: $\hat w_e = s_e\,q(|w_e|)$ with $q$ the optimal 1D quantizer with levels $\{0,c_1,\dots,c_{2^b-1}\}$: same
  number of equal-angle groups as BP$_b$, unequal levels (enough for VALUE, not for the bit-plane constructions).
- PRUNE$_b$: original weights on the edges BP$_b$ keeps, 0 on the edges it drops (separates dropping from rounding).
- QAOA: phase Hamiltonian of each arm divided by its standard deviation; angles maximize the expected *original* cost.
- Quality: $r=(E_{\max}-\overline E)/(E_{\max}-E_{\min})$, $\Delta r = r_w - r_{\hat w}$;
  $P_{\mathrm{ret}} = P_{\mathrm{opt}}(\hat w)/P_{\mathrm{opt}}(w)$, all at $p=3$.
- Cost: VALUE construction (one HWP group per magnitude, batches of at most $B$ terms); model M1 uses 40 $T$ per
  rotation ($\delta=10^{-4}$) and 4 $T$ per Toffoli.

## License

MIT, see `LICENSE`.
