"""Analytic resources of the phase-separator constructions for n = 16..256 (Table 1, Sec. 5.1). A few seconds.

For every instance, BP_b magnitudes are compiled by each construction; reductions are per-term rotation cost divided by
the construction's cost, against the original (red_L) and the rounded (red_Lp) Hamiltonian, under models M1 and M2.
Output: results/resources.json (per-instance rows and medians per size / b / B)
Usage (from release/): python scripts/run_resources.py
"""
import numpy as np

from _common import load_config, write_result
from wqhwp.quantize import magnitudes
from wqhwp.resources import constructions, tcost

CFG = load_config("resources.json")


def main():
    rows, table = [], {}
    for n in CFG["sizes"]:
        for idx in range(CFG["instances_per_size"]):
            w = np.random.default_rng(20261115 + 1000 * n + idx).normal(0, 1, n * (n - 1) // 2)
            for b in CFG["bits_b"]:
                mag, _ = magnitudes(w, b)
                mag = mag.tolist()
                L, Lp = len(mag), sum(m > 0 for m in mag)
                for B in CFG["batch_B"]:
                    res, bdir = constructions(mag, b, B)
                    rec = {"n": n, "idx": idx, "b": b, "B": B, "L": L, "L_plus": Lp, "b_dir": bdir, "res": res}
                    for model in ("M1", "M2"):
                        bL, bLp = tcost(*res["PER_TERM_L"][:2], model), tcost(*res["PER_TERM_LP"][:2], model)
                        rec[model] = {k: {"red_L": bL / tcost(r, t, model), "red_Lp": bLp / tcost(r, t, model)}
                                      for k, (r, t, _) in res.items()}
                    rows.append(rec)
    for n in CFG["sizes"]:
        for b in CFG["bits_b"]:
            for B in CFG["batch_B"]:
                R = [r for r in rows if r["n"] == n and r["b"] == b and r["B"] == B]
                t = {m: {k: {x: round(float(np.median([r[m][k][x] for r in R])), 2) for x in ("red_L", "red_Lp")}
                         for k in R[0][m]} for m in ("M1", "M2")}
                t["resources"] = {k: [float(np.median([r["res"][k][i] for r in R])) for i in range(3)] for k in R[0]["res"]}
                t["zero_frac"] = float(np.median([1 - r["L_plus"] / r["L"] for r in R]))
                t["b_dir"] = int(np.median([r["b_dir"] for r in R]))
                table[f"n{n}_b{b}_B{B}"] = t
    write_result("resources.json", __file__, "resources.json", {"table": table, "rows": rows})


if __name__ == "__main__":
    main()
