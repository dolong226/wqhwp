"""Turn results/*.json into results/numbers.tex: every number quoted in the paper, as LaTeX macros.

Nothing in the paper is typed by hand; tables and the figure \\input this file.
Usage (from release/): python scripts/make_numbers.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _common import RESULTS
from wqhwp.instances import INSTANCES, sk_instance, spins, energy
from wqhwp.quantize import quantize, magnitudes, level_sizes
from wqhwp.resources import tcount, TROT

med = lambda v: float(np.median(v))
NAMES = {12: "Twelve", 14: "Fourteen", 16: "Sixteen", 18: "Eighteen", 20: "Twenty", 22: "Twentytwo", 24: "Twentyfour"}


def resources_macros(rt):
    out = []
    NS, NT = (16, 32, 64, 128, 256), 64
    k16, k64, kinf = f"n{NT}_b2_B16", f"n{NT}_b2_B64", f"n{NT}_b2_BNone"
    out.append("\\providecommand{\\labPerTerm}{per-term (rounded $H$)}\\providecommand{\\labSandersPG}{Sanders direct + PG}")
    cons = [("PER_TERM_LP", k16, "\\labPerTerm"), ("S_DIRECT_PG_LP", k16, "\\labSandersPG"),
            ("VALUE", k16, "\\textbf{value}, $B{=}16$"), ("PLANE", k16, "plane, $B{=}16$"),
            ("JOINT_PG", k16, "pg, $B{=}16$"), ("VALUE", k64, "\\textbf{value}, $B{=}64$"),
            ("PLANE", kinf, "plane, $B{=}\\infty$")]
    rows = []
    for c, key, label in cons:
        r, t, a = rt[key]["resources"][c]
        rows.append(f"{label} & {r:.0f} & {t:.0f} & {a:.0f} & {rt[key]['M1'][c]['red_Lp']:.1f} & "
                    f"{rt[key]['M2'][c]['red_Lp']:.1f}\\\\")
    out.append("\\newcommand{\\tabsanders}{" + "\n".join(rows) + "}")
    out += [f"\\newcommand{{\\sanN}}{{{NT}}}", f"\\newcommand{{\\sanL}}{{{NT * (NT - 1) // 2}}}",
            f"\\newcommand{{\\sanZero}}{{{100 * rt[k16]['zero_frac']:.0f}}}",
            f"\\newcommand{{\\sanBdir}}{{{rt[k16]['b_dir']}}}"]
    for B, tag in ((16, "s"), (64, "l")):
        for base, bt in (("red_Lp", "Lp"), ("red_L", "L")):
            v = [rt[f"n{n}_b2_B{B}"]["M1"]["VALUE"][base] for n in NS]
            out += [f"\\newcommand{{\\val{tag}{bt}min}}{{{min(v):.1f}}}", f"\\newcommand{{\\val{tag}{bt}max}}{{{max(v):.1f}}}"]
        anc = [rt[f"n{n}_b2_B{B}"]["resources"]["VALUE"][2] for n in NS]
        out += [f"\\newcommand{{\\val{tag}Amin}}{{{min(anc):.0f}}}", f"\\newcommand{{\\val{tag}Amax}}{{{max(anc):.0f}}}"]
    sd = [rt[f"n{n}_b2_B16"][m]["S_DIRECT_PG_LP"]["red_Lp"] for n in NS for m in ("M1", "M2")]
    out.append(f"\\newcommand{{\\sdMax}}{{{max(sd):.1f}}}")
    vp = [rt[f"n{n}_b2_B16"]["M1"]["VALUE"]["red_Lp"] / rt[f"n{n}_b2_B16"]["M1"]["PLANE"]["red_Lp"] for n in NS]
    out += [f"\\newcommand{{\\vpMin}}{{{min(vp):.2f}}}", f"\\newcommand{{\\vpMax}}{{{max(vp):.2f}}}"]
    for b, bw in ((2, "BPtwo"), (3, "BPthree")):
        vs = [rt[f"n{n}_b{b}_B16"]["M1"]["VALUE"] for n in (32, 64, 128, 256)]
        for key, tag in (("red_L", "TredLarge"), ("red_Lp", "TredRoundedLarge")):
            out.append(f"\\newcommand{{\\{tag}{bw}}}{{{min(v[key] for v in vs):.1f}\\text{{--}}{max(v[key] for v in vs):.1f}}}")
    return out


def preservation_macros(pr):
    out, ns = [], sorted({r["n"] for r in pr})
    rows = []
    for a, lab in (("BP1", "BP1"), ("BP2", "\\textbf{BP2}"), ("BP3", "BP3"), ("BP4", "BP4"),
                   ("CLUSTER_4", "CLUSTER$_4$"), ("GRID_4", "GRID$_4$")):
        fr = [np.mean([r["arms"][a]["preserved"] for r in pr if r["n"] == n]) for n in ns]
        rows.append(lab + " & " + f"{med([r['arms'][a]['eps'] for r in pr]):.2f}" + " & "
                    + " & ".join(f"{x:.2f}" for x in fr) + "\\\\")
        t = {"BP1": "BPone", "BP2": "BPtwo", "BP3": "BPthree", "BP4": "BPfour", "CLUSTER_4": "Cluster", "GRID_4": "Grid"}[a]
        out += [f"\\newcommand{{\\presMin{t}}}{{{min(fr):.2f}}}", f"\\newcommand{{\\presMax{t}}}{{{max(fr):.2f}}}",
                f"\\newcommand{{\\presLast{t}}}{{{fr[-1]:.2f}}}"]
    out.append("\\newcommand{\\tabpresrows}{" + "\n".join(rows) + "}")
    out.append("\\newcommand{\\tabpreshead}{" + " & ".join(f"${n}$" for n in ns) + "}")
    out.append("\\newcommand{\\tabprescnt}{" + ", ".join(str(sum(r["n"] == n for r in pr)) for n in ns) + "}")
    out += [f"\\newcommand{{\\presN}}{{{len(pr)}}}", f"\\newcommand{{\\presNmin}}{{{ns[0]}}}", f"\\newcommand{{\\presNmax}}{{{ns[-1]}}}"]
    return out


def tradeoff_macros(tr):
    S = {"BP1": "BPone", "BP2": "BPtwo", "BP3": "BPthree", "CLUSTER_4": "Cluster", "GRID_4": "Grid"}
    acc = {a: {"ell": [], "dr": [], "P": [], "eps": [], "lay": 0, "T": 0, "L": 0, "Lnz": 0, "anc": 0} for a in S}
    r0s, ns, gains = [], [], []
    for r in tr:
        edges, w = sk_instance(r["seed"], r["n"])
        f = -energy(spins(r["n"]), edges, w)
        r0 = (f.mean() - f.min()) / (f.max() - f.min())          # r of uniform sampling
        g = lambda a, p="3": r["arms"][a]["p"][p]["r"] - r0
        lay = 1 - g("EXACT", "2") / g("EXACT")
        r0s.append(r0); ns.append(r["n"]); gains.append(100 * (g("EXACT") - g("EXACT", "2")))
        for a in S:
            x = acc[a]
            ell = 1 - g(a) / g("EXACT")
            x["ell"].append(ell); x["lay"] += ell < lay
            x["dr"].append(100 * (g("EXACT") - g(a)))              # Delta r in % of the energy range
            x["eps"].append(r["arms"][a]["eps"])
            x["P"].append(r["arms"][a]["p"]["3"]["P_opt"] / r["arms"]["EXACT"]["p"]["3"]["P_opt"])
            if a.startswith("BP"):
                b = int(a[2:])
                mag, _ = magnitudes(w, b)
                T, pk = tcount([int((mag == v).sum()) for v in range(1, 2 ** b)], 16)
                assert T == r["arms"][a]["T_value_B16"]
                x["Lnz"] += int((mag > 0).sum())
            else:
                T, pk = tcount(level_sizes(quantize(a, w)), 16)
                x["Lnz"] += len(w)
            x["T"] += T; x["L"] += len(w); x["anc"] = max(x["anc"], pk)
    nsu = sorted(set(ns))
    out = []
    for a, tag in S.items():
        x = acc[a]
        P, dr = np.array(x["P"]), np.array(x["dr"])
        out += [f"\\newcommand{{\\epsMed{tag}}}{{{med(x['eps']):.2f}}}",
                f"\\newcommand{{\\nNeg{tag}}}{{{int((100 * np.asarray(x['ell']) < 0).sum())}}}",
                f"\\newcommand{{\\nBelowGain{tag}}}{{{x['lay']}}}",
                f"\\newcommand{{\\PretMed{tag}}}{{{np.median(P):.2f}}}",
                f"\\newcommand{{\\PretQone{tag}}}{{{np.quantile(P, 0.25):.2f}}}",
                f"\\newcommand{{\\PretQthree{tag}}}{{{np.quantile(P, 0.75):.2f}}}",
                f"\\newcommand{{\\Tred{tag}}}{{{x['L'] * TROT / x['T']:.2f}}}",
                f"\\newcommand{{\\TredRounded{tag}}}{{{x['Lnz'] * TROT / x['T']:.2f}}}",
                f"\\newcommand{{\\ancMax{tag}}}{{{x['anc']}}}",
                f"\\newcommand{{\\drMed{tag}}}{{{np.median(dr):.2f}}}",
                f"\\newcommand{{\\drNinety{tag}}}{{{np.quantile(dr, 0.9):.2f}}}",
                f"\\newcommand{{\\drMax{tag}}}{{{dr.max():.2f}}}"]
        for n, nt in ((n, NAMES[n]) for n in nsu):
            out.append(f"\\newcommand{{\\drMed{tag}{nt}}}{{{med([d for d, nn in zip(x['dr'], ns) if nn == n]):.2f}}}")
    out += [f"\\newcommand{{\\nInst}}{{{len(tr)}}}",
            "\\newcommand{\\instCounts}{" + ", ".join(str(ns.count(n)) for n in nsu) + "}",
            "\\newcommand{\\instSizes}{" + ", ".join(str(n) for n in nsu) + "}",
            f"\\newcommand{{\\layerGain}}{{{med(gains):.1f}}}", f"\\newcommand{{\\rUniform}}{{{med(r0s):.2f}}}"]
    ret = lambda r, a, k, p="3": r["arms"][a]["p"][p][k] / r["arms"]["EXACT"]["p"][p][k]
    D = [ret(r, "BP2", "P_opt") - ret(r, "CLUSTER_4", "P_opt") for r in tr]
    out += [f"\\newcommand{{\\pairWins}}{{{int(sum(d > 0 for d in D))}}}",
            f"\\newcommand{{\\pairN}}{{{len(D)}}}", f"\\newcommand{{\\pairMedDiff}}{{{float(np.median(D)):+.2f}}}"]
    # fit 1 - r_hat/r = c eps^2 through the origin, pooled over the rounded arms
    for p, name in (("1", "Pone"), ("2", "Ptwo"), ("3", "Pthree")):
        xs = np.array([r["arms"][a]["eps"] ** 2 for r in tr for a in S])
        ys = np.array([1 - ret(r, a, "r", p) for r in tr for a in S])
        c = (xs @ ys) / (xs @ xs)
        r2 = 1 - ((ys - c * xs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
        out += [f"\\newcommand{{\\cFit{name}}}{{{c:.2f}}}", f"\\newcommand{{\\rsqFit{name}}}{{{r2:.2f}}}"]
        if p == "3":                     # sign changes and how the scatter around c eps^2 grows with eps
            res_ = ys - c * xs
            k = np.polyfit(np.log(np.sqrt(xs)), np.log(np.abs(res_) + 1e-12), 1)[0]
            out += [f"\\newcommand{{\\fitNeg}}{{{int((ys < 0).sum())}}}", f"\\newcommand{{\\fitPairs}}{{{len(ys)}}}",
                    f"\\newcommand{{\\residExp}}{{{k:.1f}}}"]
    rows = []
    for n in nsu:
        rn = [r for r in tr if r["n"] == n]
        cells = [f"{med([r['arms'][a]['p']['3']['r'] for r in rn]):.3f} & {med([r['arms'][a]['p']['3']['P_opt'] for r in rn]):.3f}"
                 for a in ("EXACT", "BP1", "BP2", "BP3")]
        rows.append(f"{n} & " + " & ".join(cells) + "\\\\")
    out.append("\\newcommand{\\tabAbsRows}{" + "\n".join(rows) + "}")
    return out

FAMILY = {"sk": "sk", "reg": "regular3"}
ARMTAG = {"BP1": "BPone", "BP2": "BPtwo", "BP3": "BPthree", "CLUSTER_4": "Cluster", "GRID_4": "Grid",
          "MAGKM_2": "MagTwo", "MAGKM_3": "MagThree", "PRUNE_2": "PruneTwo"}


def arm_macros(tr, prows, fam):
    """Per-arm summary at p = 3 for one instance family, macros \\<fam><Arm><Stat>, plus a table body \\<fam>ArmRows."""
    out, body = [], []
    for a in [a for a in tr[0]["arms"] if a != "EXACT"]:
        t = fam + ARMTAG[a]
        eps = [r["arms"][a]["eps"] for r in tr]
        dr = np.array([100 * (r["arms"]["EXACT"]["p"]["3"]["r"] - r["arms"][a]["p"]["3"]["r"]) for r in tr])
        P = [r["arms"][a]["p"]["3"]["P_opt"] / r["arms"]["EXACT"]["p"]["3"]["P_opt"] for r in tr]
        pres = sum(x["preserved"] for x in prows if x["arm"] == a)
        if a.startswith(("BP", "MAGKM")):
            T = sum(r["arms"][a]["T_value_B16"] for r in tr)
        elif a.startswith("PRUNE"):                                  # per-term rotations on the kept edges
            T = sum(r["arms"][a]["L_nz"] * TROT for r in tr)
        else:                                                        # one group per signed level, no sign absorption
            T = sum(tcount(level_sizes(quantize(a, INSTANCES[FAMILY[fam]](r["seed"], r["n"])[1])), 16)[0] for r in tr)
        tred = f"{sum(r['L'] for r in tr) * TROT / T:.2f}"
        zero = 1 - sum(r["arms"][a].get("L_nz", r["L"]) for r in tr) / sum(r["L"] for r in tr)
        vals = {"Eps": f"{med(eps):.2f}", "Dr": f"{np.median(dr):.2f}", "DrNinety": f"{np.quantile(dr, 0.9):.2f}",
                "Pret": f"{med(P):.2f}", "Pres": str(pres), "Tred": tred, "Zero": f"{100 * zero:.0f}"}
        out += [f"\\newcommand{{\\{t}{k}}}{{{v}}}" for k, v in vals.items()]
        body.append(f"{a.replace('_', '$_{') + '}$' if '_' in a else a} & {vals['Eps']} & {vals['Zero']} & {tred} & "
                    f"{vals['Dr']} & {vals['DrNinety']} & {vals['Pret']} & {pres}/{len(tr)}\\\\")
    xs = np.array([r["arms"][a]["eps"] ** 2 for r in tr for a in r["arms"] if a != "EXACT"])
    ys = np.array([1 - r["arms"][a]["p"]["3"]["r"] / r["arms"]["EXACT"]["p"]["3"]["r"] for r in tr for a in r["arms"] if a != "EXACT"])
    c = (xs @ ys) / (xs @ xs)
    out += ["\\newcommand{\\" + fam + "ArmRows}{" + "\n".join(body) + "}",
            f"\\newcommand{{\\{fam}cFit}}{{{c:.2f}}}",
            f"\\newcommand{{\\{fam}rsqFit}}{{{1 - ((ys - c * xs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum():.2f}}}",
            f"\\newcommand{{\\{fam}N}}{{{len(tr)}}}"]
    loss = lambda r, a: 100 * (r["arms"]["EXACT"]["p"]["3"]["r"] - r["arms"][a]["p"]["3"]["r"])
    for a, ref, tag in (("MAGKM_2", "BP2", "Two"), ("MAGKM_3", "BP3", "Three")):   # paired: MAGKM_b against BP_b
        d = [loss(r, a) - loss(r, ref) for r in tr]
        out += [f"\\newcommand{{\\{fam}MagWins{tag}}}{{{sum(x < 0 for x in d)}}}",
                f"\\newcommand{{\\{fam}MagDiff{tag}}}{{{med(d):+.2f}}}".replace("-0.00", "0.00").replace("+0.00", "0.00")]
    share = med([loss(r, "PRUNE_2") for r in tr]) / med([loss(r, "BP2") for r in tr])
    out.append(f"\\newcommand{{\\{fam}PruneShare}}{{{100 * share:.0f}}}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    res = Path(ap.parse_args().results)
    J = lambda f: json.loads((res / f).read_text())
    out = ["% GENERATED by scripts/make_numbers.py from results/*.json -- do not edit."]
    out += resources_macros(J("resources.json")["table"])
    out += preservation_macros(J("preservation.json")["instances"])
    out += tradeoff_macros(J("tradeoff.json")["instances"])
    v = J("verify.json")
    out += [f"\\newcommand{{\\verN}}{{{v['n_configs']}}}", f"\\newcommand{{\\verOK}}{{{sum(r['ok'] for r in v['rows'])}}}",
            "\\newcommand{\\verErr}{" + f"{v['max_phase_error']:.0e}".replace("e-", "\\times10^{-") + "}}"]
    rows = J("preservation_tradeoff.json")["rows"]
    orig = ("BP1", "BP2", "BP3", "CLUSTER_4", "GRID_4")          # arms of the main comparison
    for sel, tag in ((orig, "All"), (("BP2",), "BPtwo")):
        R = [x for x in rows if x["arm"] in sel]
        P = [x["P_ret"] for x in R if x["preserved"]]
        N = [x["P_ret"] for x in R if not x["preserved"]]
        out += [f"\\newcommand{{\\PretPreserved{tag}}}{{{med(P):.2f}}}",
                f"\\newcommand{{\\PretNotPreserved{tag}}}{{{med(N):.2f}}}",
                f"\\newcommand{{\\nPreserved{tag}}}{{{len(P)}}}",
                f"\\newcommand{{\\nNotPreserved{tag}}}{{{len(N)}}}",
                f"\\newcommand{{\\nPairs{tag}}}{{{len(R)}}}"]
    out += arm_macros(J("tradeoff.json")["instances"], rows, "sk")
    if (res / "tradeoff_regular3.json").exists():
        out += arm_macros(J("tradeoff_regular3.json")["instances"], J("preservation_tradeoff_regular3.json")["rows"], "reg")
    (res / "numbers.tex").write_text("\n".join(out) + "\n")
    print("wrote", res / "numbers.tex", len(out), "lines")


if __name__ == "__main__":
    main()
