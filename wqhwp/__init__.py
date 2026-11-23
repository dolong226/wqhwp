"""Weight quantization + Hamming-weight phasing for weighted QAOA phase separators.

Modules
-------
instances     SK instances, spin enumeration, energies
quantize      BP_b (sign + b-bit magnitude), CLUSTER_k (optimal 1D k-means), GRID_k
qaoa          statevector QAOA, L-BFGS-B optimizer with INTERP initialization
resources     rotation / Toffoli / ancilla / T-count formulas for every construction
circuits      reversible gate-level simulation used to verify the constructions
"""
