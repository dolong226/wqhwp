"""Weight quantizers.

BP_b       w_hat_e = s_e * d * m_e with s_e = sign(w_e) and m_e in {0, ..., 2^b - 1}; the step d minimizes ||w - w_hat||_2
           over a 1D grid. Edges with m_e = 0 drop out of the phase separator.
CLUSTER_k  optimal 1D k-means on the signed weights (dynamic programming).
GRID_k     k equally spaced levels between min(w) and max(w).
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


def grid(x, k):
    lo, hi = x.min(), x.max()
    if k == 1:
        return np.full_like(x, 0.5 * (lo + hi))
    lev = np.linspace(lo, hi, k)
    return lev[np.abs(x[:, None] - lev[None, :]).argmin(1)]


def quantize(arm, w):
    """arm in {EXACT, BP<b>, CLUSTER_<k>, GRID_<k>}."""
    if arm == "EXACT":
        return w.copy()
    if arm.startswith("BP"):
        return bitplane(w, int(arm[2:]))[0]
    kind, k = arm.split("_")
    return kmeans1d(w, int(k)) if kind == "CLUSTER" else grid(w, int(k))


def level_sizes(q):
    """Number of terms per distinct weight value (one equal-angle group per value)."""
    _, counts = np.unique(np.round(q, 12), return_counts=True)
    return counts.tolist()
