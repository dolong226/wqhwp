"""Gate-level verification of the phase-separator constructions.

A circuit built from CNOT, X, Toffoli and diagonal phases maps every basis state |z, 0> to e^{i phi(z)} |z', a'>.
It equals the target exp(-i gamma sum_e w_hat_e Z_i Z_j) up to a global phase iff z' = z, a' = 0 and phi(z) - target(z)
is the same for all z, so simulating it on every basis state is a complete check. Rotations and phase-gradient additions
are applied exactly (ideal-phase model); synthesis error only enters the cost model.

Parity of edge e = (i, j): p_e = x_i xor x_j xor [s_e < 0], so that s_e z_i z_j = 1 - 2 p_e (the sign costs one X gate).
"""
from .resources import batches, count_tree


class Circ:
    """Reversible circuit acting on one basis state, with phase and Toffoli bookkeeping."""

    def __init__(self, bits):
        self.b = dict(bits)
        self.phase = 0.0
        self.toffoli = 0
        self.log = []

    def new(self, name):
        assert name not in self.b
        self.b[name] = 0
        return name

    def cx(self, c, t, rec=True):
        self.b[t] ^= self.b[c]
        if rec:
            self.log.append(("cx", c, t))

    def ccx(self, c1, c2, t, rec=True):
        self.b[t] ^= self.b[c1] & self.b[c2]
        if rec:
            self.toffoli += 1
            self.log.append(("ccx", c1, c2, t))

    def uncompute(self):
        for g in reversed(self.log):
            if g[0] == "cx":
                self.cx(g[1], g[2], rec=False)
            else:
                self.ccx(g[1], g[2], g[3], rec=False)
        self.log = []


def compress(c, cols, tag):
    """Weighted column compression with full/half adders (one Toffoli each).
    cols = {weight exponent: [bit names]}; returns {exponent: digit bit} whose value is sum_k 2^k bits."""
    val0 = sum(c.b[x] << k for k, L in cols.items() for x in L)
    cols = {k: list(L) for k, L in cols.items() if L}
    cnt, k = 0, min(cols) if cols else 0
    while k <= max(cols, default=-1):
        L = cols.get(k, [])
        while len(L) >= 3:
            a, b_, d = L.pop(), L.pop(), L.pop()
            cnt += 1
            t = c.new(f"{tag}_fa{cnt}")
            c.cx(a, b_); c.cx(a, d); c.ccx(b_, d, t); c.cx(a, t); c.cx(b_, d); c.cx(a, d)
            L.append(d); cols.setdefault(k + 1, []).append(t)
        if len(L) == 2:
            a, b_ = L.pop(), L.pop()
            cnt += 1
            t = c.new(f"{tag}_ha{cnt}")
            c.ccx(a, b_, t); c.cx(a, b_)
            L.append(b_); cols.setdefault(k + 1, []).append(t)
        k += 1
    digits = {k: L[0] for k, L in cols.items() if L}
    assert sum(c.b[x] << k for k, x in digits.items()) == val0, "column compression wrong"
    return digits


def _finish_group(c, cols, cbits, tag):
    c.uncompute()
    for jj, L in cols.items():
        for p, v in zip(L, cbits[jj]):
            assert c.b[p] == v
            c.b[p] ^= v
            del c.b[p]
    for name in [k for k in c.b if k.startswith(tag + "_")]:
        assert c.b[name] == 0, "ancilla not returned to 0"
        del c.b[name]


def _parities(c, gi, e, edges, sign, jlist, cols):
    i1, i2 = edges[e]
    for jj in jlist:
        p = c.new(f"p{gi}_{e}_{jj}")
        c.cx(f"x{i1}", p, rec=False); c.cx(f"x{i2}", p, rec=False)
        if sign[e] < 0:
            c.b[p] ^= 1
        cols.setdefault(jj, []).append(p)


def phase_separator(x, edges, mag, sign, d, b, gamma, mode, B):
    """Apply the phase separator in one of four modes to basis state x (list of bits).
    value: HWP per magnitude value; plane: HWP per bit-plane; joint: weighted adder tree, one rotation per digit;
    pg: weighted adder tree, phase applied by an ideal phase-gradient addition.
    Returns (phase, rotations, Toffolis in the compute half, peak extra ancillas)."""
    n = len(x)
    c = Circ({f"x{i}": int(x[i]) for i in range(n)})
    theta = gamma * d
    rot, tof_total, peak = 0, 0, 0
    if mode == "value":
        from .resources import value_groups
        groups = [("v", k, Bt) for k, Bt in value_groups(mag, b, B)]
    elif mode == "plane":
        groups = [("p", j, Bt) for j in range(b) for E in [[e for e in range(len(edges)) if (mag[e] >> j) & 1]] if E
                  for Bt in batches(E, B)]
    else:
        act = [e for e in range(len(edges)) if mag[e] > 0]
        groups = [("j", None, Bt) for Bt in (batches(act, B) if act else [])]
    for gi, (kind, j, E) in enumerate(groups):
        n0, t0 = len(c.b), c.toffoli
        cols = {}
        for e in E:
            jl = [0] if kind == "v" else ([j] if kind == "p" else [k for k in range(b) if (mag[e] >> k) & 1])
            _parities(c, gi, e, edges, sign, jl, cols)
        cbits = {jj: [c.b[p] for p in L] for jj, L in cols.items()}
        tag = f"g{gi}"
        if kind == "v":
            w, digits, tot = j, compress(c, {0: cols[0]}, tag), len(cols[0])
        elif kind == "p":
            w, digits, tot = 2 ** j, compress(c, {0: cols[j]}, tag), len(cols[j])
        else:
            w, digits = 1, compress(c, cols, tag)
            tot = sum((2 ** jj) * len(L) for jj, L in cbits.items())
        if mode == "pg":
            c.phase += 2 * theta * w * sum(c.b[q] << k for k, q in digits.items())
        else:
            for k, q in digits.items():
                c.phase += 2 * theta * w * (2 ** k) * c.b[q]
                rot += 1
        c.phase += -theta * w * tot
        tof_total += c.toffoli - t0
        peak = max(peak, len(c.b) - n0)
        _finish_group(c, cols, cbits, tag)
    assert all(c.b[f"x{i}"] == int(x[i]) for i in range(n)) and len(c.b) == n
    return c.phase, rot, tof_total, peak


def predict(mag, b, mode, B):
    """Counting formula used in the resource tables for the same construction and batching."""
    from .resources import value_count
    if mode == "value":
        return value_count(mag, b, B)
    rot = tof = peak = 0
    if mode == "plane":
        for j in range(b):
            E = [e for e in range(len(mag)) if (mag[e] >> j) & 1]
            for Bt in (batches(E, B) if E else []):
                t, dg = count_tree({0: len(Bt)})
                rot += dg; tof += t; peak = max(peak, len(Bt) + t)
    else:
        act = [e for e in range(len(mag)) if mag[e] > 0]
        for Bt in (batches(act, B) if act else []):
            cs = {}
            for e in Bt:
                for k in range(b):
                    if (mag[e] >> k) & 1:
                        cs[k] = cs.get(k, 0) + 1
            t, dg = count_tree(cs)
            rot += dg if mode == "joint" else 0; tof += t; peak = max(peak, sum(cs.values()) + t)
    return rot, tof, peak
