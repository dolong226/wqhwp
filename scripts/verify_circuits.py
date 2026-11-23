"""Gate-level check of the four constructions (value, plane, joint, pg) on every computational basis state. ~1 min.

A configuration passes when the circuit phase equals exp(-i gamma H_hat) up to a global phase on all 2^n basis states,
every ancilla returns to 0, and the simulated rotation / Toffoli counts equal the counting formula of the resource tables.
Output: results/verify.json
Usage (from release/): python scripts/verify_circuits.py
"""
import cmath

import numpy as np

from _common import load_config, write_result
from wqhwp.instances import sk_instance, spins, energy
from wqhwp.quantize import bitplane
from wqhwp.circuits import phase_separator, predict

CFG = load_config("verify.json")


def main():
    rows = []
    for n, idx in CFG["instances"]:
        edges, w = sk_instance(777 + 10 * n + idx, n)
        for b in CFG["bits_b"]:
            q, _, d = bitplane(w, b)
            mag = np.rint(np.abs(q) / d).astype(int).tolist()
            sign = np.sign(w).astype(int).tolist()
            gamma = CFG["gamma"]
            target = -gamma * energy(spins(n), edges, q)
            for mode, Bs in CFG["modes"].items():
                for B in Bs:
                    diffs, rots, tofs, peaks = [], set(), set(), set()
                    for zi in range(2 ** n):
                        bits = [(zi >> (n - 1 - i)) & 1 for i in range(n)]
                        ph, r, t, pk = phase_separator(bits, edges, mag, sign, d, b, gamma, mode, B)
                        diffs.append(cmath.exp(1j * (ph - target[zi]))); rots.add(r); tofs.add(t); peaks.add(pk)
                    dd = np.array(diffs)
                    err = float(np.max(np.abs(dd - dd[0])))
                    pr, pt, pp = predict(mag, b, mode, B)
                    ok = err < CFG["tolerance"] and rots == {pr} and tofs == {pt} and max(peaks) <= pp
                    rows.append({"n": n, "idx": idx, "b": b, "mode": mode, "B": B, "max_phase_error": err,
                                 "rotations": pr, "toffolis": pt, "peak_ancilla": max(peaks), "ok": bool(ok)})
                    print("OK  " if ok else "FAIL", n, idx, b, mode, B, f"err={err:.1e}")
    write_result("verify.json", __file__, "verify.json",
                 {"all_ok": all(r["ok"] for r in rows), "n_configs": len(rows),
                  "max_phase_error": max(r["max_phase_error"] for r in rows), "rows": rows})


if __name__ == "__main__":
    main()
