"""Resource counts of phase-separator constructions (rotations, Toffolis, peak extra ancillas, T-count).

Hamming-weight phasing (HWP) of s equal-angle terms: floor(log2 s) + 1 rotations, s - w(s) Toffolis and at most
s + (s - w(s)) extra ancillas (s parity copies + adder carries), w(s) = number of 1 bits of s (Campbell 2021).
Groups larger than B are split into ceil(s / B) near-equal batches (Kivlichan et al. 2020).

Cost models
  M1: T_rot = ceil(3 log2(1/delta)) = 40 at delta = 1e-4 (Ross-Selinger); Toffoli = 4 T (temporary logical-AND).
  M2: RUS rotation 1.15 (b_pha + log2 R) + 10.925 T with b_pha = 14; Toffoli = 2 T (Sanders et al. 2020).
"""
import math

DELTA, BPHA = 1e-4, 14
TROT = math.ceil(3 * math.log2(1 / DELTA))          # 40
lg = lambda x: math.ceil(math.log2(x))


def w2(x):
    return bin(x).count("1")


def batch_sizes(s, B):
    """Split a group of s equal-angle terms into near-equal batches of size <= B (B = None: no limit)."""
    if B is None or s <= B:
        return [s]
    nb = math.ceil(s / B)
    base, extra = divmod(s, nb)
    return [base + 1] * extra + [base] * (nb - extra)


def batches(items, B):
    out, i = [], 0
    for b in batch_sizes(len(items), B):
        out.append(items[i:i + b]); i += b
    return out


def tcount(group_sizes, B):
    """T-count (model M1) and peak extra ancillas of HWP applied to equal-angle groups of the given sizes."""
    T = peak = 0
    for s in group_sizes:
        for x in (batch_sizes(s, B) if s else []):
            t = x - w2(x)
            T += (math.floor(math.log2(x)) + 1) * TROT + 4 * t
            peak = max(peak, x + t)
    return T, peak


def count_tree(col_sizes):
    """Weighted column-compression adder tree. col_sizes = {weight exponent: number of input bits}.
    Returns (Toffolis, number of output digits = rotations)."""
    cols = {k: v for k, v in col_sizes.items() if v}
    tof, k = 0, min(cols) if cols else 0
    while k <= max(cols, default=-1):
        v = cols.get(k, 0)
        while v >= 3:
            v -= 2; tof += 1; cols[k + 1] = cols.get(k + 1, 0) + 1
        if v == 2:
            v = 1; tof += 1; cols[k + 1] = cols.get(k + 1, 0) + 1
        cols[k] = v
        k += 1
    return tof, sum(1 for v in cols.values() if v)


def value_groups(mag, b, B):
    """VALUE construction: one equal-angle group per magnitude m = 1..2^b-1, split into batches."""
    return [(k, Bt) for k in range(1, 2 ** b) for E in [[e for e in range(len(mag)) if mag[e] == k]] if E
            for Bt in batches(E, B)]


def value_count(mag, b, B):
    rot = tof = peak = 0
    for _, Bt in value_groups(mag, b, B):
        t, dg = count_tree({0: len(Bt)})
        rot += dg; tof += t; peak = max(peak, len(Bt) + t)
    return rot, tof, peak


def constructions(mag, b, B):
    """{construction: (rotations, Toffolis, peak extra ancillas)} for one instance with BP_b magnitudes mag.

    PER_TERM_L     one rotation per term of the original Hamiltonian
    PER_TERM_LP    one rotation per nonzero term of the rounded Hamiltonian
    S_DIRECT_PG_LP direct-energy oracle of Sanders et al. (Alg. 1) on the nonzero terms + phase-gradient multiplication
    VALUE          HWP per magnitude value (main construction)
    PLANE          HWP per bit-plane (an edge with several set bits appears in several groups)
    JOINT_PG       weighted adder tree per batch, accumulated and added into a phase-gradient register (no rotation)
    """
    L = len(mag)
    act = [e for e in range(L) if mag[e] > 0]
    Lp = len(act)
    bdir = lg(Lp * (2 ** b - 1) + 1) + 1
    out = {"PER_TERM_L": (L, 0, 0), "PER_TERM_LP": (Lp, 0, 0),
           "S_DIRECT_PG_LP": (0, 2 * Lp * (bdir - 2) + BPHA * BPHA / 2, bdir + bdir - 1 + BPHA),
           "VALUE": value_count(mag, b, B)}
    rot = tof = peak = 0
    for j in range(b):
        E = [e for e in act if (mag[e] >> j) & 1]
        for Bt in (batches(E, B) if E else []):
            t, dg = count_tree({0: len(Bt)}); rot += dg; tof += t; peak = max(peak, len(Bt) + t)
    out["PLANE"] = (rot, tof, peak)
    trees = []
    for Bt in batches(act, B):
        cs = {}
        for e in Bt:
            for k in range(b):
                if (mag[e] >> k) & 1:
                    cs[k] = cs.get(k, 0) + 1
        t, dg = count_tree(cs)
        trees.append((t, dg, sum(cs.values())))
    if len(trees) == 1:
        t, dg, nb = trees[0]
        out["JOINT_PG"] = (0, t + BPHA * BPHA / 2, nb + t + BPHA)
    else:
        acc = lg(sum(int(m) for m in mag) + 1)
        tt = sum(t for t, _, _ in trees) + len(trees) * (acc - 1)
        out["JOINT_PG"] = (0, 2 * tt + BPHA * BPHA / 2, max(nb + t for t, _, nb in trees) + 2 * acc + BPHA)
    return out, bdir


def tcost(rot, tof, model):
    """T-count of `rot` arbitrary rotations and `tof` Toffolis under model M1 or M2."""
    if model == "M1":
        return rot * TROT + 4 * tof
    trot = 1.15 * (BPHA + math.log2(rot)) + 10.925 if rot else 0
    return rot * trot + 2 * tof
