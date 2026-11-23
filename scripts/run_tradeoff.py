"""Quality-resource trade-off on SK instances: QAOA with original and rounded weights, p = 1..3.

Output: results/tradeoff.json (per instance and arm: weight error eps, VALUE T-count, nonzero terms L_nz, and r, P_opt,
angles at each depth). About 1.6 h with 3 workers on a laptop.
Usage (from release/): OMP_NUM_THREADS=1 python scripts/run_tradeoff.py [--workers 3] [--only n:idx]
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from _common import load_config, write_result
from wqhwp.instances import sk_instance, spins, energy
from wqhwp.quantize import quantize, magnitudes
from wqhwp.qaoa import optimize, interp
from wqhwp.resources import tcount

CFG = load_config("tradeoff.json")
ARMS = CFG["arms"]


def seed_of(n, idx):
    assert 0 <= idx < 1000
    return 600000000 + 1000 * n + idx


def run_task(task):
    n, idx = task
    seed = seed_of(n, idx)
    edges, w = sk_instance(seed, n)
    Z = spins(n)
    f = -energy(Z, edges, w)
    fmin, fmax = f.min(), f.max()
    opt = f >= fmax - 1e-9 * max(1.0, abs(fmax))
    fnorm = (f - f.mean()) / f.std()
    out = {"n": n, "idx": idx, "seed": seed, "L": len(w), "arms": {}}
    for ai, arm in enumerate(ARMS):
        q = quantize(arm, w)
        H = energy(Z, edges, q); H = H / H.std()
        rec = {"eps": float(np.linalg.norm(w - q) / np.linalg.norm(w)), "p": {}}
        if arm.startswith("BP"):
            b = int(arm[2:])
            mag, _ = magnitudes(w, b)
            rec.update({"T_value_B16": int(tcount([int((mag == v).sum()) for v in range(1, 2 ** b)], 16)[0]),
                        "L_nz": int((mag > 0).sum())})
        prev = None
        for p in CFG["depths"]:
            rs = np.random.default_rng(seed * 100 + ai * 10 + p)
            x, prob = optimize(H, fnorm, n, p, rs, interp(prev, p - 1) if prev is not None else None)
            prev = x
            rec["p"][str(p)] = {"r": float((prob @ f - fmin) / (fmax - fmin)), "P_opt": float(prob[opt].sum()),
                                "angles": x.tolist()}
        out["arms"][arm] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", help="run a single instance n:idx and print it (no output file)")
    a = ap.parse_args()
    if a.only:
        n, idx = map(int, a.only.split(":"))
        r = run_task((n, idx))
        print({arm: {p: (v["r"], v["P_opt"]) for p, v in rec["p"].items()} for arm, rec in r["arms"].items()})
        return r
    tasks = sorted([(int(n), i) for n, c in CFG["sizes"].items() for i in range(c)], key=lambda t: -t[0])
    t0, res = time.time(), []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_task, tasks):
            res.append(r)
            print(f"[{time.time() - t0:7.1f}s] n={r['n']} idx={r['idx']}", flush=True)
    res.sort(key=lambda r: (r["n"], r["idx"]))
    write_result("tradeoff.json", __file__, "tradeoff.json",
                 {"runtime_s": time.time() - t0, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                  "workers": a.workers, "instances": res})


if __name__ == "__main__":
    main()
