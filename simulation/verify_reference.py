"""
simulation/verify_reference.py

Revised plan sections 7-8, for virtual cell #4 (both parameters at baseline).

  A. Reference test on a cloned branch of the initial state -> Q0.
     Direct integral of the 0.5 A discharge vs Discharge capacity end-start.
     Main state and main resume unchanged by the diagnostic.
     Q0 repeated at three solver tolerances.
  B. 600 s continuous vs 60 s x 10 chained (CC 3.0 A, last_state chaining)
     at three solver tolerances; piece-boundary continuity; plating convergence.
  C. CC->CV state connection: one Experiment (internal CC->CV) vs our chunked path
     (CC 60 s pieces until 4.2 V, then a separate CV model), at three tolerances.

Run:     python simulation/verify_reference.py
Output:  outputs/verify_reference/{report.json, log.txt}
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pybamm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import battery as B  # noqa: E402

OUT = Path(os.environ.get("BATTERY_OUTPUT_DIR", B.ROOT / "outputs")) / "verify_reference"
OUT.mkdir(parents=True, exist_ok=True)

TOLS = [(1e-4, 1e-6), (1e-6, 1e-8), (1e-8, 1e-10)]   # default, stricter, strictest
I_CHECK = 3.0
DT = float(B.MODEL_CFG["boundary_update_s"])

# Differential (must be continuous) vs algebraic (may jump at a control change)
STATE_VARS = {
    "neg_particle_conc": "Negative particle concentration [mol.m-3]",
    "pos_particle_conc": "Positive particle concentration [mol.m-3]",
    "electrolyte_conc": "Electrolyte concentration [mol.m-3]",
    "sei_thickness": "X-averaged negative SEI thickness [m]",
    "plated_li": "X-averaged negative lithium plating concentration [mol.m-3]",
    "dead_li": "X-averaged negative dead lithium concentration [mol.m-3]",
    "neg_porosity": "Negative electrode porosity",
    "discharge_capacity": "Discharge capacity [A.h]",
}
ALGEBRAIC_VARS = {"current": "Current [A]", "voltage": "Voltage [V]"}
LI_LOST = "Total lithium lost [mol]"
LI_LOST_SIDE = "Total lithium lost to side reactions [mol]"

_lines: list[str] = []
REPORT: dict = {
    "pybamm_version": pybamm.__version__, "python": sys.version,
    "model_config": B.MODEL_CFG, "diagnostic_config": B.DIAG_CFG,
    "tolerances": [{"rtol": r, "atol": a} for r, a in TOLS],
    "cell": {"id": 4, "negative_particle_radius_m": None, "sei_solvent_diffusivity_m2s": None},
}


def log(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def flush():
    (OUT / "log.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps(REPORT, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def snapshot(step, index=-1, names=None) -> dict:
    names = names or {**STATE_VARS, **ALGEBRAIC_VARS}
    out = {}
    for k, n in names.items():
        try:
            out[k] = B.values(step, n, index)
        except Exception as e:
            out[k] = f"unavailable: {type(e).__name__}"
    return out


def diff(sa: dict, sb: dict) -> dict:
    res = {}
    for k in sa:
        a, b = sa[k], sb.get(k)
        if isinstance(a, str) or isinstance(b, str) or b is None:
            res[k] = {"status": "unavailable"}
        elif a.shape != b.shape:
            res[k] = {"status": f"shape {a.shape} vs {b.shape}"}
        else:
            res[k] = B.compare_values(a, b)
    return res


def li_balance(step) -> dict:
    try:
        lost = float(B.values(step, LI_LOST)[0])
        side = float(B.values(step, LI_LOST_SIDE)[0])
        return {"total_lost_mol": lost, "lost_to_side_reactions_mol": side,
                "residual_mol": lost - side}
    except Exception as e:
        return {"status": f"unavailable: {type(e).__name__}: {e}"}


def range_check(step) -> dict:
    out = {}
    for k in ("plated_li", "dead_li", "sei_thickness", "electrolyte_conc",
              "neg_particle_conc", "pos_particle_conc"):
        try:
            a = np.asarray(step[STATE_VARS[k]].entries, dtype=float)
            out[k] = {"min": float(a.min()), "negative": bool(a.min() < 0)}
        except Exception:
            out[k] = "unavailable"
    try:
        a = np.asarray(step[STATE_VARS["neg_porosity"]].entries, dtype=float)
        out["neg_porosity"] = {"min": float(a.min()), "max": float(a.max()),
                               "out_of_0_1": bool(a.min() <= 0 or a.max() >= 1)}
    except Exception:
        out["neg_porosity"] = "unavailable"
    return out


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


# ── A. reference test ───────────────────────────────────────────────
def reference_test(start, solver):
    sim = B.make_sim(B.reference_test_steps(), solver=solver,
                     period_s=B.DIAG_CFG["reference_test"]["output_period_s"])
    sol, wall = timed(B.solve_from, sim, start)
    steps = B.steps_of_last_cycle(sol)
    labels = ["cc_charge", "cv_hold", "rest", "discharge"]
    info = {}
    for lab, st in zip(labels, steps):
        info[lab] = {"termination": str(st.termination),
                     "t_start_s": float(st.t[0]), "t_end_s": float(st.t[-1]),
                     "duration_s": float(st.t[-1] - st.t[0]),
                     "I_end_A": float(B.values(st, "Current [A]")[0]),
                     "V_end_V": float(B.values(st, "Voltage [V]")[0]),
                     "n_points": len(st.t)}
    dis = steps[3]
    t = np.asarray(dis["Time [s]"].entries, dtype=float)
    i = np.asarray(dis["Current [A]"].entries, dtype=float)
    q = np.asarray(dis["Discharge capacity [A.h]"].entries, dtype=float)
    q_trapz = float(np.trapezoid(i, t) / 3600.0)
    q_cum = float(q[-1] - q[0])
    q_ixt = float(np.mean(i) * (t[-1] - t[0]) / 3600.0)
    info["Q"] = {
        "trapz_Ah": q_trapz, "discharge_capacity_end_minus_start_Ah": q_cum,
        "mean_current_x_duration_Ah": q_ixt,
        "trapz_minus_cum_Ah": q_trapz - q_cum,
        "trapz_minus_cum_rel": (q_trapz - q_cum) / q_cum,
        "current_min_A": float(i.min()), "current_max_A": float(i.max()),
        "cum_final_value_Ah": float(q[-1]),
    }
    info["wall_s"] = wall
    info["total_sim_hours"] = float(steps[-1].t[-1] - steps[0].t[0]) / 3600.0
    info["memory"] = B.memory_mb()
    return sol, info


def part_a():
    log("=== A. reference test")
    solver = B.make_solver()
    S0 = B.initial_state(solver=solver)
    y0 = B.y_end(S0)
    ls = S0.last_state
    ls_y0 = np.asarray(ls.all_ys[-1], dtype=float).copy()
    t0_main = float(S0.t[-1])

    # control: main run with no diagnostic
    main_sim = B.make_sim([B.cc_charge(I_CHECK, 600)], solver=solver)
    ctrl, w_ctrl = timed(B.solve_from, main_sim, S0.last_state)
    ctrl_step = B.steps_of_last_cycle(ctrl)[-1]
    y_ctrl, t_ctrl = B.y_end(ctrl), float(ctrl_step.t[-1])

    # diagnostic on a copy of the main state
    D, info = reference_test(S0.last_state.copy(), solver)
    d_start = float(B.steps_of_last_cycle(D)[0].t[0])
    del D
    gc.collect()

    y0_after = B.y_end(S0)
    ls_after = np.asarray(S0.last_state.all_ys[-1], dtype=float)

    # resume main after the diagnostic, same built model
    res, w_res = timed(B.solve_from, main_sim, S0.last_state)
    res_step = B.steps_of_last_cycle(res)[-1]
    y_res, t_res = B.y_end(res), float(res_step.t[-1])

    # diagnostic WITHOUT copy: does it mutate the object it was given, or the main solution?
    same_object = S0.last_state is S0.last_state
    ls_obj = S0.last_state
    ls_obj_y_before = np.asarray(ls_obj.all_ys[-1], dtype=float).copy()
    D2, info2 = reference_test(ls_obj, solver)
    del D2
    gc.collect()
    y0_after_nocopy = B.y_end(S0)
    ls_obj_change = float(np.max(np.abs(np.asarray(ls_obj.all_ys[-1], dtype=float) - ls_obj_y_before)))

    q = info["Q"]
    REPORT["A_reference_test"] = {
        "main_start_t_s": t0_main,
        "diagnostic_branch_start_t_s": d_start,
        "reference_test": info,
        "Q0_Ah": q["trapz_Ah"],
        "state_preservation": {
            "main_y_max_abs_change_after_diag_copy": float(np.max(np.abs(y0_after - y0))),
            "last_state_y_max_abs_change_after_diag_copy": float(np.max(np.abs(ls_after - ls_y0))),
            "main_y_max_abs_change_after_diag_no_copy": float(np.max(np.abs(y0_after_nocopy - y0))),
            "passed_object_y_change_after_diag_no_copy": ls_obj_change,
            "last_state_property_returns_same_object": same_object,
            "resume_vs_control_y_max_abs_diff": float(np.max(np.abs(y_res - y_ctrl))),
            "resume_t_end_s": t_res, "control_t_end_s": t_ctrl,
            "resume_time_not_shifted": abs(t_res - t_ctrl) < 1e-9 and abs(t_res - (t0_main + 600)) < 1e-6,
        },
        "no_copy_run_Q_Ah": info2["Q"]["trapz_Ah"],
        "wall_s": {"control_main": w_ctrl, "resume_main": w_res,
                   "reference_test_copy": info["wall_s"], "reference_test_no_copy": info2["wall_s"]},
    }
    sp = REPORT["A_reference_test"]["state_preservation"]
    log(f"  Q0 trapz={q['trapz_Ah']:.6f} Ah, cum diff={q['discharge_capacity_end_minus_start_Ah']:.6f} Ah, "
        f"rel diff={q['trapz_minus_cum_rel']:.2e}")
    log(f"  preservation: main dy={sp['main_y_max_abs_change_after_diag_copy']:.2e}, "
        f"resume vs control dy={sp['resume_vs_control_y_max_abs_diff']:.2e}, "
        f"time ok={sp['resume_time_not_shifted']}")
    flush()

    # Q0 at stricter tolerances
    q_tol = []
    for rtol, atol in TOLS:
        s = B.make_solver(rtol, atol)
        S = B.initial_state(solver=s)
        D, inf = reference_test(S.last_state.copy(), s)
        q_tol.append({"rtol": rtol, "atol": atol, "Q_trapz_Ah": inf["Q"]["trapz_Ah"],
                      "Q_cum_Ah": inf["Q"]["discharge_capacity_end_minus_start_Ah"],
                      "trapz_minus_cum_rel": inf["Q"]["trapz_minus_cum_rel"],
                      "discharge_duration_s": inf["discharge"]["duration_s"],
                      "cv_duration_s": inf["cv_hold"]["duration_s"],
                      "wall_s": inf["wall_s"], "memory": inf["memory"]})
        log(f"  Q0 rtol={rtol:g}: {inf['Q']['trapz_Ah']:.7f} Ah, wall {inf['wall_s']:.1f} s")
        del D
        gc.collect()
    qs = [x["Q_trapz_Ah"] for x in q_tol]
    REPORT["A_reference_test"]["Q0_vs_tolerance"] = q_tol
    REPORT["A_reference_test"]["Q0_change_default_to_strictest_Ah"] = qs[0] - qs[-1]
    REPORT["A_reference_test"]["Q0_change_stricter_to_strictest_Ah"] = qs[1] - qs[-1]
    flush()


# ── B. continuity, tolerance sweep ──────────────────────────────────
def chained_cc(S0, solver, n, dt, stop_on_voltage=False):
    sim = B.make_sim([B.cc_charge(I_CHECK, dt)], solver=solver)
    prev, boundaries, steps_end, walls = S0, [], None, []
    full_y_jumps = []
    for k in range(n):
        sol, w = timed(B.solve_from, sim, prev.last_state)
        walls.append(w)
        st = B.steps_of_last_cycle(sol)[-1]
        if k > 0:
            boundaries.append(diff(snapshot(prev_step, -1), snapshot(st, 0)))
            y_prev = np.asarray(prev_step.all_ys[-1], dtype=float)[:, -1]
            y_next = np.asarray(st.all_ys[0], dtype=float)[:, 0]
            full_y_jumps.append(float(np.max(np.abs(y_next - y_prev))))
        prev, prev_step = sol, st
        if stop_on_voltage and "voltage" in str(st.termination).lower():
            break
    return prev, prev_step, boundaries, full_y_jumps, walls


def part_b():
    log("=== B. 600 s continuous vs 60 s x 10")
    levels, ends = [], []
    for rtol, atol in TOLS:
        s = B.make_solver(rtol, atol)
        S0 = B.initial_state(solver=s)
        long_sim = B.make_sim([B.cc_charge(I_CHECK, 600)], solver=s)
        L, wl = timed(B.solve_from, long_sim, S0.last_state)
        Ls = B.steps_of_last_cycle(L)[-1]
        C, Cs, bnd, yj, wc = chained_cc(S0, s, 10, DT)
        end_diff = diff(snapshot(Ls), snapshot(Cs))
        worst_boundary = {}
        for k in STATE_VARS:
            vals = [b[k]["max_abs_diff"] for b in bnd if "max_abs_diff" in b.get(k, {})]
            worst_boundary[k] = max(vals) if vals else None
        levels.append({
            "rtol": rtol, "atol": atol,
            "t_end_continuous_s": float(Ls.t[-1]), "t_end_chained_s": float(Cs.t[-1]),
            "termination_continuous": str(Ls.termination),
            "end_diff_continuous_vs_chained": end_diff,
            "piece_boundary_max_abs_jump_state_vars": worst_boundary,
            "piece_boundary_max_abs_jump_full_y": max(yj) if yj else None,
            "li_balance_continuous": li_balance(Ls), "li_balance_chained": li_balance(Cs),
            "ranges_chained": range_check(Cs),
            "wall_s": {"continuous": wl, "chained_total": sum(wc)},
        })
        ends.append({"L": snapshot(Ls), "C": snapshot(Cs)})
        log(f"  rtol={rtol:g}: plated rel={end_diff['plated_li'].get('max_rel_diff'):.2e}, "
            f"dead rel={end_diff['dead_li'].get('max_rel_diff'):.2e}, "
            f"boundary jump full y={levels[-1]['piece_boundary_max_abs_jump_full_y']:.2e}")
        del L, C
        gc.collect()
    ref = ends[-1]["L"]
    for lv, e in zip(levels, ends):
        lv["continuous_vs_strictest_continuous"] = diff(ref, e["L"])
        lv["chained_vs_strictest_continuous"] = diff(ref, e["C"])
    REPORT["B_continuity"] = levels
    flush()


# ── C. CC->CV connection ────────────────────────────────────────────
def part_c():
    log("=== C. CC->CV connection")
    cv_total = 2 * DT
    levels = []
    for rtol, atol in TOLS:
        s = B.make_solver(rtol, atol)
        S0 = B.initial_state(solver=s)
        # A: PyBaMM internal transition in one Experiment
        simA = B.make_sim([B.cc_charge(I_CHECK, 3 * 3600), B.cv_hold(cv_total)], solver=s)
        A, wa = timed(B.solve_from, simA, S0.last_state)
        A_cc, A_cv = B.steps_of_last_cycle(A)[-2:]
        # B: our chunked path
        Cc, Cc_step, bnd, yj, wc = chained_cc(S0, s, 200, DT, stop_on_voltage=True)
        cv_sim = B.make_sim([B.cv_hold(DT)], solver=s)
        prev, prev_step, cv_bnd, wv = Cc, Cc_step, [], []
        first_cv_step = None
        for k in range(int(cv_total / DT)):
            sol, w = timed(B.solve_from, cv_sim, prev.last_state)
            wv.append(w)
            st = B.steps_of_last_cycle(sol)[-1]
            cv_bnd.append(diff(snapshot(prev_step, -1), snapshot(st, 0)))
            if first_cv_step is None:
                first_cv_step = st
            prev, prev_step = sol, st
        cc_to_cv_ours = cv_bnd[0]
        cc_to_cv_pybamm = diff(snapshot(A_cc, -1), snapshot(A_cv, 0))
        levels.append({
            "rtol": rtol, "atol": atol,
            "cc_end_t_s": {"one_experiment": float(A_cc.t[-1]), "chunked": float(Cc_step.t[-1])},
            "cc_termination": {"one_experiment": str(A_cc.termination), "chunked": str(Cc_step.termination)},
            "cc_pieces": len(wc),
            "transition_jump_ours_cc_end_to_cv_start": cc_to_cv_ours,
            "transition_jump_pybamm_internal": cc_to_cv_pybamm,
            "cv_to_cv_jump": cv_bnd[1:],
            "cv_start_current_A": {"one_experiment": float(B.values(A_cv, "Current [A]", 0)[0]),
                                   "chunked": float(B.values(first_cv_step, "Current [A]", 0)[0])},
            "end_diff_one_experiment_vs_chunked": diff(snapshot(A_cv), snapshot(prev_step)),
            "t_end_s": {"one_experiment": float(A_cv.t[-1]), "chunked": float(prev_step.t[-1])},
            "li_balance": {"one_experiment": li_balance(A_cv), "chunked": li_balance(prev_step)},
            "wall_s": {"one_experiment": wa, "chunked_cc": sum(wc), "chunked_cv": sum(wv)},
        })
        e = levels[-1]["end_diff_one_experiment_vs_chunked"]
        log(f"  rtol={rtol:g}: cc end t {levels[-1]['cc_end_t_s']}, "
            f"end plated rel={e['plated_li'].get('max_rel_diff'):.2e}, "
            f"dead rel={e['dead_li'].get('max_rel_diff'):.2e}")
        del A, Cc, prev
        gc.collect()
    REPORT["C_cc_to_cv"] = levels
    flush()


def main():
    p = B.make_param()
    REPORT["cell"]["negative_particle_radius_m"] = p["Negative particle radius [m]"]
    REPORT["cell"]["sei_solvent_diffusivity_m2s"] = p["SEI solvent diffusivity [m2.s-1]"]
    REPORT["cell"]["lithium_plating_kinetic_rate_constant_m_s"] = p["Lithium plating kinetic rate constant [m.s-1]"]
    t0 = time.perf_counter()
    for name, fn in (("A", part_a), ("B", part_b), ("C", part_c)):
        t = time.perf_counter()
        try:
            fn()
        except Exception:
            REPORT[f"{name}_error"] = traceback.format_exc()
            log(REPORT[f"{name}_error"])
        REPORT[f"{name}_wall_total_s"] = time.perf_counter() - t
        REPORT["memory_after_" + name] = B.memory_mb()
        flush()
    REPORT["wall_total_s"] = time.perf_counter() - t0
    flush()
    log(f"report: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
