"""Optimum preservation under weight rounding.

Part 'enumerate' (Table 4): exhaustive enumeration on fresh SK instances, n = 12..24, no QAOA (about 1 min).
Part 'tradeoff' (Sec. 5.3): the same test on the instances of a trade-off run, related to P_ret = P_opt(arm) /
P_opt(EXACT) at p = 3 (seconds; requires run_tradeoff.py with the same config first).
Output: results/preservation.json, results/preservation_<trade-off output name>
Usage (from release/): python scripts/run_preservation.py enumerate
                       python scripts/run_preservation.py tradeoff [tradeoff.json | tradeoff_regular3.json]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from _common import RESULTS, load_config, write_result
from wqhwp.instances import INSTANCES, sk_instance, spins, energy, energies_half, preservation
from wqhwp.quantize import quantize
import run_tradeoff as TR

CFG = load_config("preservation.json")


def enum_task(task):
    n, idx = task
    seed = 20261116 + 100 * n + idx
    edges, w = sk_instance(seed, n)
    qs = [quantize(a, w) for a in CFG["arms"]]
    E, *Eqs = energies_half(n, edges, [w] + qs)
    iopt = int(np.argmin(E))
    rec = {"n": n, "idx": idx, "seed": seed, "arms": {}}
    for a, q, Eq in zip(CFG["arms"], qs, Eqs):
        rank, ties, ok = preservation(E, Eq, iopt)
        rec["arms"][a] = {"eps": float(np.linalg.norm(w - q) / np.linalg.norm(w)), "rank": rank, "ties": ties,
                          "preserved": ok}
    return rec


def enumerate_part():
    tasks = sorted([(int(n), i) for n, c in CFG["sizes"].items() for i in range(c)], key=lambda t: -t[0])
    t0, res = time.time(), []
    with Pool(8) as pool:
        for r in pool.imap_unordered(enum_task, tasks):
            res.append(r)
    res.sort(key=lambda r: (r["n"], r["idx"]))
    write_result("preservation.json", __file__, "preservation.json", {"runtime_s": time.time() - t0, "instances": res})


def tradeoff_part(cfg_name="tradeoff.json"):
    c = TR.cfg(cfg_name)
    tr = json.loads((RESULTS / c["output"]).read_text())
    arms = [a for a in c["arms"] if a != "EXACT"]
    rows = []
    for r in tr["instances"]:
        n, idx = r["n"], r["idx"]
        edges, w = INSTANCES[c["family_key"]](r["seed"], n)
        Z = spins(n)
        half = 1 << (n - 1)
        E = energy(Z, edges, w)[:half]
        iopt = int(np.argmin(E))
        for a in arms:
            q = quantize(a, w)
            assert abs(float(np.linalg.norm(w - q) / np.linalg.norm(w)) - r["arms"][a]["eps"]) < 1e-12
            rank, ties, ok = preservation(E, energy(Z, edges, q)[:half], iopt)
            rows.append({"n": n, "idx": idx, "arm": a, "rank": rank, "ties": ties, "preserved": ok,
                         "P_ret": r["arms"][a]["p"]["3"]["P_opt"] / r["arms"]["EXACT"]["p"]["3"]["P_opt"]})

    def summ(R):
        P = [x["P_ret"] for x in R if x["preserved"]]
        N = [x["P_ret"] for x in R if not x["preserved"]]
        return {"n_rows": len(R), "n_preserved": len(P), "n_not": len(N),
                "median_P_ret_preserved": float(np.median(P)) if P else None,
                "median_P_ret_not": float(np.median(N)) if N else None}

    write_result("preservation_" + c["output"], __file__, cfg_name,
                 {"pooled": summ(rows), "by_arm": {a: summ([x for x in rows if x["arm"] == a]) for a in arms},
                  "rows": rows})


if __name__ == "__main__":
    {"enumerate": enumerate_part, "tradeoff": tradeoff_part}[sys.argv[1]](*sys.argv[2:])
