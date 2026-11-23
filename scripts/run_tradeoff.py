"""Quality-resource trade-off: QAOA with original and rounded weights, p = 1..3.

The config fixes the instance family, sizes, seeds and arms. Output (named in the config): per instance and arm the
weight error eps, nonzero terms L_nz, VALUE T-count for arms with sign absorption, and r, P_opt, angles at each depth.
About 2 h per config with 8 workers on a laptop.
Usage (from release/): OMP_NUM_THREADS=1 python scripts/run_tradeoff.py [--config tradeoff.json] [--workers 8]
                       [--only n:idx]
"""
import argparse
import os
import time
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from _common import load_config, write_result
from wqhwp.instances import INSTANCES, spins, energy
from wqhwp.quantize import quantize, magnitudes, level_sizes
from wqhwp.qaoa import optimize, interp
from wqhwp.resources import tcount

cfg = lru_cache(None)(load_config)


def seed_of(c, n, idx):
    assert 0 <= idx < 1000
    return c["seed_base"] + 1000 * n + idx


def value_groups(arm, w, q):
    """Sizes of the equal-angle groups of the VALUE construction (sign absorbed by X), or None for arms without it."""
    if arm.startswith("BP"):
        mag, _ = magnitudes(w, int(arm[2:]))
        return [int((mag == v).sum()) for v in range(1, 2 ** int(arm[2:]))]
    if arm.startswith("MAGKM"):
        a = np.abs(q)
        return level_sizes(a[a > 0])
    return None


def run_task(task):
    cfg_name, n, idx = task
    c = cfg(cfg_name)
    seed = seed_of(c, n, idx)
    edges, w = INSTANCES[c["family_key"]](seed, n)
    Z = spins(n)
    f = -energy(Z, edges, w)
    fmin, fmax = f.min(), f.max()
    opt = f >= fmax - 1e-9 * max(1.0, abs(fmax))
    fnorm = (f - f.mean()) / f.std()
    out = {"n": n, "idx": idx, "seed": seed, "L": len(w), "arms": {}}
    for ai, arm in enumerate(c["arms"]):
        q = quantize(arm, w)
        H = energy(Z, edges, q); H = H / H.std()
        rec = {"eps": float(np.linalg.norm(w - q) / np.linalg.norm(w)), "p": {}}
        g = value_groups(arm, w, q)
        if g is not None:
            rec["T_value_B16"] = int(tcount(g, 16)[0])
        if arm != "EXACT":
            rec["L_nz"] = int((q != 0).sum())
        prev = None
        for p in c["depths"]:
            rs = np.random.default_rng(seed * 100 + ai * 10 + p)
            x, prob = optimize(H, fnorm, n, p, rs, interp(prev, p - 1) if prev is not None else None)
            prev = x
            rec["p"][str(p)] = {"r": float((prob @ f - fmin) / (fmax - fmin)), "P_opt": float(prob[opt].sum()),
                                "angles": x.tolist()}
        out["arms"][arm] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tradeoff.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", help="run a single instance n:idx and print it (no output file)")
    a = ap.parse_args()
    c = cfg(a.config)
    if a.only:
        n, idx = map(int, a.only.split(":"))
        r = run_task((a.config, n, idx))
        print({arm: {p: (v["r"], v["P_opt"]) for p, v in rec["p"].items()} for arm, rec in r["arms"].items()})
        return r
    tasks = sorted([(a.config, int(n), i) for n, k in c["sizes"].items() for i in range(k)], key=lambda t: -t[1])
    t0, res = time.time(), []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_task, tasks):
            res.append(r)
            print(f"[{time.time() - t0:7.1f}s] n={r['n']} idx={r['idx']}", flush=True)
    res.sort(key=lambda r: (r["n"], r["idx"]))
    write_result(c["output"], __file__, a.config,
                 {"runtime_s": time.time() - t0, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                  "workers": a.workers, "instances": res})


if __name__ == "__main__":
    main()
