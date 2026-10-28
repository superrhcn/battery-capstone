"""
simulation/run_pilot9.py

Pilot 9 combinations (configs/cells_pilot9.json). Runs only if
outputs/verify_criteria/report.json says all_pass, and uses its selected tolerance.

Per cell (separate process, so peak RAM and failures are per cell):
  1. Q0 from a cloned reference test of the initial state (SOC 20 %).
  2. Numerical-error proxies: Q0 and R10 at the initial state with tolerance x0.01 and mesh x2.
  3. Control check: 3.3 A charge, 5 A discharge, one cycle, SOC boundary update 60 s vs 10 s.
  4. Grading run from a separate clone of the initial state: 1.5 A / 5 A, SOC 20-80 %,
     5 min rests, until 5 EFC (60 min limit not applied).
  5. At 5 EFC, on clones: Q5 (reference test) and R10; repeated at tolerance x0.01 from
     the same state. Cell 4 also repeats the whole 5 EFC run at tolerance x0.01.

Run:     python simulation/run_pilot9.py            (all cells, then summary)
         python simulation/run_pilot9.py --cell 4   (one cell)
Output:  outputs/pilot9/cell_<id>.json, summary.json, summary.md
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pybamm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import battery as B  # noqa: E402

CFG = json.loads((B.ROOT / "configs" / "cells_pilot9.json").read_text(encoding="utf-8"))
OUT = Path(os.environ.get("BATTERY_OUTPUT_DIR", B.ROOT / "outputs")) / "pilot9"
OUT.mkdir(parents=True, exist_ok=True)
VERIFY = B.ROOT / "outputs" / "verify_criteria" / "report.json"


def save(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=B.jsonable), encoding="utf-8")
    tmp.replace(path)


def end_state(c, sol):
    st = B.last_step(sol)
    out = c.degradation_Ah(st)
    out.update({"soc": c.soc(st), "discharge_capacity_Ah": B.scalar(st, "Discharge capacity [A.h]"),
                "t_s": float(st.t[-1])})
    try:
        out["li_balance_residual_mol"] = (B.scalar(st, "Total lithium lost [mol]")
                                          - B.scalar(st, "Total lithium lost to side reactions [mol]"))
    except Exception as e:
        out["li_balance_residual_mol"] = f"unavailable: {e}"
    return out


def operation(c, S0, cell_id, traj, q0, I_ch, I_dis, dt, **kw):
    rs = B.RunState(cell_id=cell_id, trajectory_id=traj, q0_Ah=q0, t_abs_s=float(S0.t[-1]))
    t0 = time.perf_counter()
    sol, cycles, status, chk = B.run_operation(c, S0, rs, I_ch, I_dis, dt, **kw)
    wall = time.perf_counter() - t0
    # SOC-terminal errors only; a discharge stopped at the EFC target is not a SOC termination
    soc_err = [abs(cyc["soc_charge_error"]) for cyc in cycles if cyc.get("charge_end") in ("soc", "soc_already")]
    soc_err += [abs(cyc["soc_discharge_error"]) for cyc in cycles
                if cyc.get("discharge_end") in ("soc", "soc_already")]
    return sol, rs, {"status": status, "cycles": cycles, "checks": chk, "ledger_end": B.ledger(rs),
                     "wall_s": wall, "max_abs_soc_error": max(soc_err) if soc_err else None,
                     "end_state": end_state(c, sol)}


def run_cell(cell_id: int, tolerance: dict):
    t_cell = time.perf_counter()
    spec = next(x for x in CFG["cells"] if x["id"] == cell_id)
    ov = {"Negative particle radius [m]": spec["negative_particle_radius_m"],
          "SEI solvent diffusivity [m2.s-1]": spec["sei_solvent_diffusivity_m2s"]}
    strict = {"rtol": tolerance["rtol"] * 0.01, "atol": tolerance["atol"] * 0.01}
    res = {"cell": spec, "overrides": ov, "tolerance": tolerance, "strict_tolerance": strict,
           "pybamm_version": pybamm.__version__, "failures": []}
    path = OUT / f"cell_{cell_id}.json"
    try:
        c = B.Cell(ov, **tolerance)
        res["parameters_used"] = {k: c.param[k] for k in (
            "Negative particle radius [m]", "SEI solvent diffusivity [m2.s-1]",
            "Lithium plating kinetic rate constant [m.s-1]", "Nominal cell capacity [A.h]")}
        S0 = c.initial_state()
        ref0 = c.reference_test(S0.last_state.copy())
        q0 = ref0["Q_trapz_Ah"]
        r10_0 = c.r10_test(S0)
        res["Q0"] = {"Q0_Ah": q0, "reference_test": ref0,
                     "integral_minus_cum_rel_q0": abs(ref0["Q_trapz_Ah"] - ref0["Q_cum_Ah"]) / q0}
        res["R10_initial"] = r10_0
        if not ref0["ok"]:
            res["failures"].append("reference_test_Q0_termination")
        save(path, res)

        num = {}
        for label, kw in (("tol_x0.01", strict), ("mesh_x2", {**tolerance, "mesh_scale": 2.0})):
            cv = B.Cell(ov, **kw)
            Sv = cv.initial_state()
            rq, rr = cv.reference_test(Sv.last_state.copy()), cv.r10_test(Sv)
            num[label] = {"Q0_Ah": rq["Q_trapz_Ah"], "dQ0_Ah": rq["Q_trapz_Ah"] - q0,
                          "R10_ohm": rr["R10_ohm"], "dR10_ohm": rr["R10_ohm"] - r10_0["R10_ohm"],
                          "ok": rq["ok"] and rr["ok"]}
            del cv, Sv
        res["numerical_initial"] = num
        save(path, res)

        cc = CFG["control_check"]
        ctrl = {}
        for dt in cc["update_intervals_s"]:
            _, _, info = operation(c, S0, cell_id, f"ctrl_{cc['charge_current_A']}A_dt{dt}", q0,
                                   cc["charge_current_A"], cc["discharge_current_A"], float(dt),
                                   max_cycles=1, charge_limit_s=cc["charge_time_limit_s"])
            ctrl[f"dt{dt}"] = info
            if info["status"] != "max_cycles":
                res["failures"].append(f"control_dt{dt}:{info['status']}")
        a, b = ctrl["dt60"], ctrl["dt10"]
        ca, cb = a["cycles"][0], b["cycles"][0]
        ctrl["dt60_vs_dt10"] = {
            "charge_time_diff_s": ca["charge_time_s"] - cb["charge_time_s"],
            "soc_charge_end_diff": ca["soc_charge_end"] - cb["soc_charge_end"],
            "soc_discharge_end_diff": ca.get("soc_discharge_end", np.nan) - cb.get("soc_discharge_end", np.nan),
            "end_state_diff": {k: a["end_state"][k] - b["end_state"][k]
                               for k in ("sei_Ah", "plated_Ah", "dead_Ah", "discharge_capacity_Ah", "soc")},
            "wall_ratio_dt10_over_dt60": b["wall_s"] / a["wall_s"],
        }
        res["control_check"] = ctrl
        save(path, res)

        g = CFG["grading_run"]
        sol5, rs5, info5 = operation(c, S0, cell_id, "grade_1.5A", q0, g["charge_current_A"],
                                     g["discharge_current_A"], float(g["update_interval_s"]),
                                     efc_stop=g["efc_stop"])
        res["grading_run"] = info5
        if info5["status"] != "efc_target":
            res["failures"].append(f"grading:{info5['status']}")
        else:
            led_b, y_b = copy.deepcopy(B.ledger(rs5)), B.y_end(sol5)
            ref5 = c.reference_test(sol5.last_state.copy())
            r10 = c.r10_test(sol5)
            res["at_5EFC"] = {"Q5_Ah": ref5["Q_trapz_Ah"], "SOH_percent": 100 * ref5["Q_trapz_Ah"] / q0,
                              "reference_test": ref5, "R10": r10,
                              "main_state_unchanged": float(np.max(np.abs(B.y_end(sol5) - y_b))) == 0.0,
                              "ledger_unchanged": B.ledger(rs5) == led_b}
            if not (ref5["ok"] and r10["ok"]):
                res["failures"].append("diagnostic_at_5EFC_termination")
            cs = B.Cell(ov, **strict)
            ref5s, r10s = cs.reference_test(sol5.last_state.copy()), cs.r10_test(sol5)
            res["at_5EFC"]["strict_from_same_state"] = {
                "Q5_Ah": ref5s["Q_trapz_Ah"], "dQ5_Ah": ref5s["Q_trapz_Ah"] - ref5["Q_trapz_Ah"],
                "R10_ohm": r10s["R10_ohm"], "dR10_ohm": r10s["R10_ohm"] - r10["R10_ohm"]}
            del cs
        save(path, res)

        if cell_id == 4 and "at_5EFC" in res:
            cs = B.Cell(ov, **strict)
            Ss = cs.initial_state()
            q0s = cs.reference_test(Ss.last_state.copy())["Q_trapz_Ah"]
            sols, _, infos = operation(cs, Ss, cell_id, "grade_1.5A_strict", q0s, g["charge_current_A"],
                                       g["discharge_current_A"], float(g["update_interval_s"]),
                                       efc_stop=g["efc_stop"])
            q5s = cs.reference_test(sols.last_state.copy())["Q_trapz_Ah"]
            r10t = cs.r10_test(sols)["R10_ohm"]
            res["trajectory_strict"] = {"status": infos["status"], "Q0_Ah": q0s, "Q5_Ah": q5s, "R10_ohm": r10t,
                                        "dQ5_Ah": q5s - res["at_5EFC"]["Q5_Ah"],
                                        "dR10_ohm": r10t - res["at_5EFC"]["R10"]["R10_ohm"],
                                        "dSOH_pp": 100 * q5s / q0s - res["at_5EFC"]["SOH_percent"],
                                        "wall_s": infos["wall_s"]}
    except Exception:
        res["failures"].append("exception")
        res["traceback"] = traceback.format_exc()
    res["wall_s"] = time.perf_counter() - t_cell
    res["memory"] = B.memory_mb()
    save(path, res)


def summarize():
    rows, cells = [], {}
    for spec in CFG["cells"]:
        p = OUT / f"cell_{spec['id']}.json"
        cells[spec["id"]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"failures": ["missing"]}
    traj = cells.get(4, {}).get("trajectory_strict", {})
    for k, r in cells.items():
        a5 = r.get("at_5EFC", {})
        ctrl = r.get("control_check", {})
        num = r.get("numerical_initial", {})
        soc_errs = []
        for run in (ctrl.get("dt60", {}), ctrl.get("dt10", {}), r.get("grading_run", {})):
            for cy in run.get("cycles", []):
                if cy.get("charge_end") in ("soc", "soc_already"):
                    soc_errs.append(abs(cy["soc_charge_error"]))
                if cy.get("discharge_end") in ("soc", "soc_already"):
                    soc_errs.append(abs(cy["soc_discharge_error"]))
        eps_q = max([abs(a5.get("strict_from_same_state", {}).get("dQ5_Ah", 0.0))]
                    + [abs(v.get("dQ0_Ah", 0.0)) for v in num.values()] + [abs(traj.get("dQ5_Ah", 0.0))])
        eps_r = max([abs(a5.get("strict_from_same_state", {}).get("dR10_ohm", 0.0))]
                    + [abs(v.get("dR10_ohm", 0.0)) for v in num.values()] + [abs(traj.get("dR10_ohm", 0.0))])
        g_cycles = r.get("grading_run", {}).get("cycles", [])
        rows.append({
            "id": k, "radius_um": r.get("cell", {}).get("negative_particle_radius_m", np.nan) * 1e6,
            "D_sol": r.get("cell", {}).get("sei_solvent_diffusivity_m2s"),
            "Q0_Ah": r.get("Q0", {}).get("Q0_Ah"), "Q5_Ah": a5.get("Q5_Ah"), "SOH5_pct": a5.get("SOH_percent"),
            "R10_mohm": a5.get("R10", {}).get("R10_ohm", np.nan) * 1e3 if a5 else None,
            "max_abs_soc_error_pp": 100 * max(soc_errs) if soc_errs else None,
            "dt60_minus_dt10_soc_charge_end_pp": 100 * ctrl.get("dt60_vs_dt10", {}).get("soc_charge_end_diff", np.nan)
            if ctrl else None,
            "charge_min_3p3A": ctrl.get("dt60", {}).get("cycles", [{}])[0].get("charge_time_s", np.nan) / 60
            if ctrl else None,
            "cv_entered_3p3A": ctrl.get("dt60", {}).get("cycles", [{}])[0].get("cv_entered") if ctrl else None,
            "charge_min_1p5A_first": g_cycles[0]["charge_time_s"] / 60 if g_cycles else None,
            "cv_entered_1p5A_any": any(cy.get("cv_entered") for cy in g_cycles) if g_cycles else None,
            "grading_cycles": len(g_cycles),
            "wall_s": r.get("wall_s"), "peak_rss_mb": r.get("memory", {}).get("peak_rss_mb"),
            "failures": r.get("failures"),
            "eps_Q_Ah": eps_q, "eps_R_mohm": eps_r * 1e3,
        })
    ok = [x for x in rows if x["Q5_Ah"] is not None and x["R10_mohm"] is not None]
    dist = {"Q5": [0, 0], "R10": [0, 0]}
    for a, b in itertools.combinations(ok, 2):
        dist["Q5"][1] += 1
        dist["R10"][1] += 1
        if abs(a["Q5_Ah"] - b["Q5_Ah"]) > a["eps_Q_Ah"] + b["eps_Q_Ah"]:
            dist["Q5"][0] += 1
        if abs(a["R10_mohm"] - b["R10_mohm"]) > a["eps_R_mohm"] + b["eps_R_mohm"]:
            dist["R10"][0] += 1
    summary = {"rows": rows, "distinguishable_pairs": dist,
               "spread": {"Q5_Ah": [min(x["Q5_Ah"] for x in ok), max(x["Q5_Ah"] for x in ok)] if ok else None,
                          "R10_mohm": [min(x["R10_mohm"] for x in ok), max(x["R10_mohm"] for x in ok)] if ok else None,
                          "max_eps_Q_Ah": max((x["eps_Q_Ah"] for x in ok), default=None),
                          "max_eps_R_mohm": max((x["eps_R_mohm"] for x in ok), default=None)},
               "trajectory_strict_cell4": traj}
    save(OUT / "summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int)
    args = ap.parse_args()
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    if not v.get("all_pass"):
        raise SystemExit("verify_criteria did not pass; pilot run not started")
    tolerance = v["selected_tolerance"]
    if args.cell:
        run_cell(args.cell, tolerance)
        return
    for spec in CFG["cells"]:
        p = OUT / f"cell_{spec['id']}.json"
        if p.exists():
            done = json.loads(p.read_text(encoding="utf-8"))
            if "memory" in done and not done.get("failures"):
                continue   # completed without failure in an earlier run of this same code
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cell", str(spec["id"])],
                       check=False)
    summarize()
    (OUT / "DONE").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
