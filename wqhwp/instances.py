"""Sherrington-Kirkpatrick instances and exact energies.

Convention: E_w(z) = sum_{i<j} w_ij z_i z_j is minimized. Computational-basis bit 0 <-> z = +1, bit 1 <-> z = -1,
qubit 0 is the most significant bit.
"""
import numpy as np

CHUNK = 1 << 18


def sk_instance(seed, n):
    """Complete graph on n vertices, w_ij ~ N(0, 1) drawn from numpy default_rng(seed) in lexicographic edge order."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return edges, np.random.default_rng(seed).normal(0, 1, len(edges))


def spins(n):
    """All 2^n spin configurations as a (2^n, n) array of +-1."""
    b = (np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (1 - 2 * b).astype(np.float64)


def energy(Z, edges, w):
    """E_w(z) for every row z of Z."""
    E = np.zeros(Z.shape[0])
    for (i, j), v in zip(edges, w):
        E += v * Z[:, i] * Z[:, j]
    return E


def energies_half(n, edges, ws):
    """Energies of the 2^(n-1) states with z_0 = +1 (one per global spin-flip pair), for each weight vector in ws.
    Memory-light version of energy() for exhaustive enumeration up to n = 24."""
    Ws = []
    for w in ws:
        W = np.zeros((n, n))
        for (i, j), v in zip(edges, w):
            W[i, j] = W[j, i] = v
        Ws.append(W)
    N = 1 << (n - 1)
    out = [np.empty(N) for _ in ws]
    sh = np.arange(n - 2, -1, -1)
    for s in range(0, N, CHUNK):
        k = np.arange(s, min(N, s + CHUNK))
        Z = np.ones((len(k), n))
        Z[:, 1:] = 1 - 2 * ((k[:, None] >> sh) & 1)
        for o, W in zip(out, Ws):
            o[s:s + len(k)] = 0.5 * np.einsum("ij,ij->i", Z @ W, Z)
    return out


def preservation(E, Eq, iopt):
    """Is the true optimum (index iopt of E) still the unique ground state of the rounded energies Eq?
    Returns (number of states strictly below it under Eq, number of ties, preserved flag). Ties count as not preserved."""
    tol = 1e-9 * max(1.0, np.abs(Eq).max())
    e0 = Eq[iopt]
    rank = int((Eq < e0 - tol).sum())
    ties = int((np.abs(Eq - e0) <= tol).sum()) - 1
    return rank, ties, bool(rank == 0 and ties == 0)
