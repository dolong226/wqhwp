"""Weight quantizers.

BP_b       w_hat_e = s_e * d * m_e with s_e = sign(w_e) and m_e in {0, ..., 2^b - 1}; the step d minimizes ||w - w_hat||_2
           over a 1D grid. Edges with m_e = 0 drop out of the phase separator.
CLUSTER_k  optimal 1D k-means on the signed weights (dynamic programming).
GRID_k     k equally spaced levels between min(w) and max(w).
MAGKM_b    sign(w_e) times an optimal 1D quantizer of |w_e| with levels {0, c_1, ..., c_K}, K = 2^b - 1 (dynamic
           programming; the smallest magnitudes map to 0). Same number of equal-angle groups as BP_b, unequal levels.
PRUNE_b    w_e where BP_b keeps the edge (m_e > 0), 0 where BP_b drops it: isolates the effect of dropping edges.
"""
import numpy as np


def bitplane(w, b):
    """BP_b quantizer. Returns (w_hat, nonzero bit-plane sizes, step d)."""
    L = 2 ** b - 1
    a = np.abs(w)
    best = None
    for d in (a.max() / L) * np.linspace(0.1, 1.2, 1101):
        mag = np.clip(np.round(a / d), 0, L).astype(int)
        e = np.sum((w - np.sign(w) * mag * d) ** 2)
        if best is None or e < best[0]:
            best = (e, d, mag)
    _, d, mag = best
    q = np.sign(w) * mag * d
    planes = [mag & (1 << j) > 0 for j in range(b)]
    assert np.allclose(sum((2 ** j) * d * np.sign(w) * planes[j] for j in range(b)), q)
    return q, [int(p.sum()) for p in planes if p.sum() > 0], float(d)


def magnitudes(w, b):
    """Integer magnitudes m_e of BP_b and the step d."""
    q, _, d = bitplane(w, b)
    return np.rint(np.abs(q) / d).astype(int), d


def kmeans1d(x, k):
    """Optimal 1D k-means by dynamic programming on sorted values; returns the quantized values in original order."""
    order = np.argsort(x, kind="stable")
    s = x[order]
    m = len(s)
    k = min(k, m)
    c1 = np.concatenate([[0], np.cumsum(s)])
    c2 = np.concatenate([[0], np.cumsum(s * s)])

    def sse(a, b):
        cnt = b - a
        tot = c1[b] - c1[a]
        return (c2[b] - c2[a]) - tot * tot / cnt

    D = np.full((k + 1, m + 1), np.inf)
    A = np.zeros((k + 1, m + 1), dtype=int)
    D[0, 0] = 0.0
    for c in range(1, k + 1):
        for b in range(c, m + 1):
            best, arg = np.inf, -1
            for a in range(c - 1, b):
                v = D[c - 1, a] + sse(a, b)
                if v < best:
                    best, arg = v, a
            D[c, b], A[c, b] = best, arg
    q = np.empty(m)
    b = m
    for c in range(k, 0, -1):
        a = A[c, b]
        q[a:b] = s[a:b].mean()
        b = a
    out = np.empty(m)
    out[order] = q
    return out


def magkm(w, b):
    """MAGKM_b: minimize ||w - sign(w) * q(|w|)||_2 over q with at most 2^b - 1 nonzero levels plus the level 0."""
    K = 2 ** b - 1
    a = np.abs(w)
    order = np.argsort(a, kind="stable")
    s = a[order]
    m = len(s)
    c1 = np.concatenate([[0], np.cumsum(s)])
    c2 = np.concatenate([[0], np.cumsum(s * s)])
    D = np.full((K + 1, m + 1), np.inf)
    A = np.full((K + 1, m + 1), -1, dtype=int)
    D[0] = c2                                   # s[:j] all mapped to 0
    for k in range(1, K + 1):
        for j in range(1, m + 1):
            i = np.arange(j)
            v = D[k - 1, :j] + (c2[j] - c2[i]) - (c1[j] - c1[i]) ** 2 / (j - i)
            t = int(np.argmin(v))
            if v[t] < D[k - 1, j]:
                D[k, j], A[k, j] = v[t], t
            else:                                # fewer than k nonzero levels is at least as good
                D[k, j] = D[k - 1, j]
    q = np.zeros(m)
    j, k = m, K
    while k > 0 and j > 0:
        i = A[k, j]
        if i >= 0:
            q[i:j] = s[i:j].mean()
            j = i
        k -= 1
    out = np.empty(m)
    out[order] = q
    return np.sign(w) * out


def grid(x, k):
    lo, hi = x.min(), x.max()
    if k == 1:
        return np.full_like(x, 0.5 * (lo + hi))
    lev = np.linspace(lo, hi, k)
    return lev[np.abs(x[:, None] - lev[None, :]).argmin(1)]


def quantize(arm, w):
    """arm in {EXACT, BP<b>, CLUSTER_<k>, GRID_<k>, MAGKM_<b>, PRUNE_<b>}."""
    if arm == "EXACT":
        return w.copy()
    if arm.startswith("BP"):
        return bitplane(w, int(arm[2:]))[0]
    kind, k = arm.split("_")
    if kind == "MAGKM":
        return magkm(w, int(k))
    if kind == "PRUNE":
        return np.where(magnitudes(w, int(k))[0] > 0, w, 0.0)
    return {"CLUSTER": kmeans1d, "GRID": grid}[kind](w, int(k))


def level_sizes(q):
    """Number of terms per distinct weight value (one equal-angle group per value)."""
    _, counts = np.unique(np.round(q, 12), return_counts=True)
    return counts.tolist()
