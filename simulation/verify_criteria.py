"""
simulation/verify_criteria.py

Project first-stage criteria (cell #4, baseline parameters, 25 degC).

  A. Q0 from the cloned reference test
     |direct integral - (Discharge capacity end - start)| <= 0.01 % Q0
     Q0 change with solver tolerance (x0.1, x0.01) and with mesh x2: each <= 0.1 %
     main state, main resume and operating ledger unchanged by the diagnostic
  B. 600 s continuous vs 60 s x 10 (CC 3.0 A 240 s, rest 120 s, 5 A discharge 240 s),
     same initial state, same fixed SOC-boundary inputs, outputs every 1 s.
     max SOC diff <= 0.01 %p, voltage diff in the same control step <= 1 mV,
     charge / discharge Ah diff <= 0.1 % Q0, SEI / plated / dead Li (Ah) diff <= 0.01 % Q0,
     at solver tolerance x1, x0.1, x0.01.
  C. CC->CV: one Experiment vs chunked path (CC pieces, then CV pieces), same thresholds,
     plus the jump between the last CC state and the first CV state.

Run:     python simulation/verify_criteria.py
Output:  outputs/verify_criteria/{report.json, log.txt}
"""

from __future__ import annotations

import copy
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

OUT = Path(os.environ.get("BATTERY_OUTPUT_DIR", B.ROOT / "outputs")) / "verify_criteria"
OUT.mkdir(parents=True, exist_ok=True)
FACTORS = [1.0, 0.1, 0.01]
I_CHECK = 3.0
LIM = {"q_integral_rel": 1e-4, "q0_tol_rel": 1e-3, "q0_mesh_rel": 1e-3,
       "soc_abs": 1e-4, "voltage_V": 1e-3, "throughput_rel_q0": 1e-3, "degradation_rel_q0": 1e-4,
       "transition_jump_rel": 1e-10}
REPORT: dict = {"pybamm_version": pybamm.__version__, "python": sys.version,
                "model_config": B.MODEL_CFG, "diagnostic_config": B.DIAG_CFG,
                "limits": LIM, "default_tolerance": {"rtol": B.DEFAULT_RTOL, "atol": B.DEFAULT_ATOL},
                "default_var_pts": None}
_lines: list[str] = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def flush():
    (OUT / "log.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False,
                                                default=B.jsonable), encoding="utf-8")


def tol(f):
    return {"rtol": B.DEFAULT_RTOL * f, "atol": B.DEFAULT_ATOL * f}


# ── A ───────────────────────────────────────────────────────────────
def part_a():
    t0 = time.perf_counter()
    c = B.Cell()
    S0 = c.initial_state()
    rs = B.RunState(cell_id=4, trajectory_id="verify_A", t_abs_s=float(S0.t[-1]))
    led_before, y_before = copy.deepcopy(B.ledger(rs)), B.y_end(S0)
    ctrl_steps = [pybamm.step.current(-I_CHECK, duration=600, termination=f"{B.V_MAX} V")]
    _, ctrl_st, _ = c.solve("CTRL", ctrl_steps, S0.last_state)
    ref = c.reference_test(S0.last_state.copy())
    led_after, y_after = B.ledger(rs), B.y_end(S0)
    _, res_st, _ = c.solve("CTRL", ctrl_steps, S0.last_state)
    q0 = ref["Q_trapz_Ah"]
    res = {"reference_test": ref, "Q0_Ah": q0,
           "integral_minus_cum_Ah": ref["Q_trapz_Ah"] - ref["Q_cum_Ah"],
           "integral_minus_cum_rel_q0": abs(ref["Q_trapz_Ah"] - ref["Q_cum_Ah"]) / q0,
           "main_y_change": float(np.max(np.abs(y_after - y_before))),
           "ledger_unchanged": led_before == led_after,
           "resume_vs_control_y": float(np.max(np.abs(
               np.asarray(res_st.all_ys[-1])[:, -1] - np.asarray(ctrl_st.all_ys[-1])[:, -1]))),
           "resume_t_end_s": float(res_st.t[-1]), "control_t_end_s": float(ctrl_st.t[-1]),
           "variants": []}
    REPORT["default_var_pts"] = B.make_model().default_var_pts
    for label, kw in (("tol x0.1", tol(0.1)), ("tol x0.01", tol(0.01)), ("mesh x2", {"mesh_scale": 2.0})):
        cv = B.Cell(**kw)
        Sv = cv.initial_state()
        r = cv.reference_test(Sv.last_state.copy())
        res["variants"].append({"label": label, "settings": kw, "var_pts": cv.var_pts,
                                "Q_Ah": r["Q_trapz_Ah"], "rel_change_vs_default": abs(r["Q_trapz_Ah"] - q0) / q0,
                                "ok": r["ok"], "wall_s": r["wall_s"]})
        log(f"  A {label}: Q={r['Q_trapz_Ah']:.7f} Ah, rel={abs(r['Q_trapz_Ah'] - q0) / q0:.2e}, "
            f"wall {r['wall_s']:.1f} s")
        del cv, Sv
        gc.collect()
    v = {x["label"]: x for x in res["variants"]}
    res["criteria"] = {
        "reference_test_terminations_ok": ref["ok"],
        "integral_vs_cum": res["integral_minus_cum_rel_q0"] <= LIM["q_integral_rel"],
        "tolerance_x0.1": v["tol x0.1"]["rel_change_vs_default"] <= LIM["q0_tol_rel"],
        "tolerance_x0.01": v["tol x0.01"]["rel_change_vs_default"] <= LIM["q0_tol_rel"],
        "mesh_x2": v["mesh x2"]["rel_change_vs_default"] <= LIM["q0_mesh_rel"],
        "main_state_unchanged": res["main_y_change"] == 0.0,
        "ledger_unchanged": res["ledger_unchanged"],
        "resume_equals_control": res["resume_vs_control_y"] == 0.0
        and res["resume_t_end_s"] == res["control_t_end_s"],
    }
    res["pass"] = all(res["criteria"].values())
    res["wall_s"] = time.perf_counter() - t0
    REPORT["A_Q0"] = res
    log(f"A pass={res['pass']} Q0={q0:.7f} Ah  {res['criteria']}")
    flush()
    return q0


# ── B ───────────────────────────────────────────────────────────────
SERIES = ["Voltage [V]", "Discharge capacity [A.h]", "Loss of capacity to negative SEI [A.h]",
          "Volume-averaged negative lithium plating concentration [mol.m-3]",
          "Volume-averaged negative dead lithium concentration [mol.m-3]",
          B.VARS["Q_n"], B.VARS["Q_p"], B.VARS["n_Li_particles"], B.X]


def series(steps):
    t = np.concatenate([np.asarray(s["Time [s]"].entries, dtype=float) for s in steps])
    cols = {n: np.concatenate([np.ravel(np.asarray(s[n].entries, dtype=float)) for s in steps])
            for n in SERIES}
    key = np.round(t, 6)
    _, idx = np.unique(key, return_index=True)
    return key[idx], {n: v[idx] for n, v in cols.items()}


def compare_step(c, cont_step, pieces, q0):
    ta, a = series([cont_step])
    tb, b = series(pieces)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    k = c.neg_volume_m3 * B.F / 3600.0
    out = {"n_common_times": int(len(common)), "t_first": float(common[0]), "t_last": float(common[-1])}
    out["max_voltage_diff_V"] = float(np.max(np.abs(a["Voltage [V]"][ia] - b["Voltage [V]"][ib])))
    for name, key, scale in (("sei", "Loss of capacity to negative SEI [A.h]", 1.0),
                             ("plated", "Volume-averaged negative lithium plating concentration [mol.m-3]", k),
                             ("dead", "Volume-averaged negative dead lithium concentration [mol.m-3]", k)):
        va, vb = a[key][ia] * scale, b[key][ib] * scale
        out[f"{name}_Ah_end"] = float(va[-1])
        out[f"{name}_Ah_max_abs_diff"] = float(np.max(np.abs(va - vb)))
        out[f"{name}_rel_q0"] = out[f"{name}_Ah_max_abs_diff"] / q0
    qa, qb = a["Discharge capacity [A.h]"], b["Discharge capacity [A.h]"]
    out["throughput_Ah"] = {"continuous": float(qa[-1] - qa[0]), "chained": float(qb[-1] - qb[0])}
    out["throughput_abs_diff_Ah"] = abs(out["throughput_Ah"]["continuous"] - out["throughput_Ah"]["chained"])
    sel = sorted(set(list(range(0, len(common), 10)) + [len(common) - 1]))
    soc_d = []
    for j in sel:
        sa = B.soc_from_x(a[B.X][ia[j]], c.boundaries_from(a[B.VARS["Q_n"]][ia[j]], a[B.VARS["Q_p"]][ia[j]],
                                                            a[B.VARS["n_Li_particles"]][ia[j]]))
        sb = B.soc_from_x(b[B.X][ib[j]], c.boundaries_from(b[B.VARS["Q_n"]][ib[j]], b[B.VARS["Q_p"]][ib[j]],
                                                            b[B.VARS["n_Li_particles"]][ib[j]]))
        soc_d.append(abs(sa - sb))
    out["soc_points"] = len(sel)
    out["max_soc_diff"] = float(max(soc_d))
    return out


def part_b(q0):
    levels = []
    for f in FACTORS:
        t0 = time.perf_counter()
        c = B.Cell(**tol(f))
        S0 = c.initial_state()
        b0 = c.boundaries(B.last_step(S0))
        far = {B.X_HI: B.x_from_soc(0.99, b0), B.X_LO: B.x_from_soc(0.01, b0)}
        cont_steps = [B.step_cc(I_CHECK, 240, period=1, piece=False),
                      B.step_rest(120, period=1, piece=False),
                      B.step_dis(5.0, 240, period=1, piece=False)]
        cont, _, wc = c.solve("CONT", cont_steps, S0.last_state, far)
        cs = cont.cycles[-1].steps
        cont_classes = [B.classify(s) for s in cs]
        builders = {"CC": lambda: [B.step_cc(I_CHECK, 60, period=1)],
                    "REST": lambda: [B.step_rest(60, period=1)],
                    "DIS": lambda: [B.step_dis(5.0, 60, period=1)]}
        prev, t, pieces, classes, joins, wsum = S0, float(S0.t[-1]), {"CC": [], "REST": [], "DIS": []}, [], [], 0.0
        prev_st = B.last_step(S0)
        for kind, n in (("CC", 4), ("REST", 2), ("DIS", 4)):
            for _ in range(n):
                inp = {B.T_STOP: t + 60.0}
                if kind == "CC":
                    inp[B.X_HI] = far[B.X_HI]
                if kind == "DIS":
                    inp[B.X_LO] = far[B.X_LO]
                new, st, w = c.solve("P" + kind, builders[kind](), prev.last_state, inp)
                wsum += w
                if pieces["CC"] or pieces["REST"] or pieces["DIS"]:
                    d = B.snap_diff(B.snapshot(prev_st, -1), B.snapshot(st, 0))
                    joins.append(max(v.get("max_abs_diff", 0.0) / max(v.get("ref_max_abs", 1.0), 1e-300)
                                     for kk, v in d.items()))
                pieces[kind].append(st)
                classes.append(B.classify(st))
                prev, prev_st, t = new, st, float(st.t[-1])
        steps_cmp = {name: compare_step(c, cs[i], pieces[name], q0)
                     for i, name in enumerate(("CC", "REST", "DIS"))}
        m = {
            "max_soc_diff": max(s["max_soc_diff"] for s in steps_cmp.values()),
            "max_voltage_diff_V": max(s["max_voltage_diff_V"] for s in steps_cmp.values()),
            "charge_Ah_diff_rel_q0": steps_cmp["CC"]["throughput_abs_diff_Ah"] / q0,
            "discharge_Ah_diff_rel_q0": steps_cmp["DIS"]["throughput_abs_diff_Ah"] / q0,
            "sei_rel_q0": max(s["sei_rel_q0"] for s in steps_cmp.values()),
            "plated_rel_q0": max(s["plated_rel_q0"] for s in steps_cmp.values()),
            "dead_rel_q0": max(s["dead_rel_q0"] for s in steps_cmp.values()),
            "sei_abs_Ah": max(s["sei_Ah_max_abs_diff"] for s in steps_cmp.values()),
            "plated_abs_Ah": max(s["plated_Ah_max_abs_diff"] for s in steps_cmp.values()),
            "dead_abs_Ah": max(s["dead_Ah_max_abs_diff"] for s in steps_cmp.values()),
            "max_piece_join_rel_jump": max(joins),
        }
        crit = {"soc": m["max_soc_diff"] <= LIM["soc_abs"],
                "voltage": m["max_voltage_diff_V"] <= LIM["voltage_V"],
                "charge_Ah": m["charge_Ah_diff_rel_q0"] <= LIM["throughput_rel_q0"],
                "discharge_Ah": m["discharge_Ah_diff_rel_q0"] <= LIM["throughput_rel_q0"],
                "sei": m["sei_rel_q0"] <= LIM["degradation_rel_q0"],
                "plated": m["plated_rel_q0"] <= LIM["degradation_rel_q0"],
                "dead": m["dead_rel_q0"] <= LIM["degradation_rel_q0"],
                "all_pieces_time_class": all(x == "time" for x in classes),
                "continuous_steps_full": all(x == "time" for x in cont_classes)}
        levels.append({"tolerance": tol(f), "factor": f, "metrics": m, "criteria": crit,
                       "pass": all(crit.values()), "per_step": steps_cmp,
                       "end_values_Ah": {k: steps_cmp["DIS"][f"{k}_Ah_end"] for k in ("sei", "plated", "dead")},
                       "wall_s": {"continuous": wc, "chained": wsum, "level_total": time.perf_counter() - t0}})
        log(f"  B x{f:g}: pass={levels[-1]['pass']} soc={m['max_soc_diff']:.2e} V={m['max_voltage_diff_V']:.2e} "
            f"plated={m['plated_rel_q0']:.2e} dead={m['dead_rel_q0']:.2e} sei={m['sei_rel_q0']:.2e}")
        del c, cont, prev
        gc.collect()
    keys = ["max_soc_diff", "max_voltage_diff_V", "charge_Ah_diff_rel_q0", "discharge_Ah_diff_rel_q0",
            "sei_rel_q0", "plated_rel_q0", "dead_rel_q0"]
    REPORT["B_split"] = {"levels": levels,
                         "trend": {k: [lv["metrics"][k] for lv in levels] for k in keys}}
    flush()
    return levels


# ── C ───────────────────────────────────────────────────────────────
def part_c(q0):
    levels = []
    for f in FACTORS:
        t0 = time.perf_counter()
        c = B.Cell(**tol(f))
        S0 = c.initial_state()
        b0 = c.boundaries(B.last_step(S0))
        far = B.x_from_soc(0.99, b0)
        one_steps = [pybamm.step.current(-I_CHECK, duration=3 * 3600, termination=f"{B.V_MAX} V"),
                     pybamm.step.voltage(B.V_MAX, duration=120)]
        A, _, wa = c.solve("ONE", one_steps, S0.last_state)
        a_cc, a_cv = A.cycles[-1].steps[-2:]
        prev, t, n_cc, w = S0, float(S0.t[-1]), 0, 0.0
        while True:
            new, st, wi = c.solve("PCC", [B.step_cc(I_CHECK, 60)], prev.last_state,
                                  {B.T_STOP: t + 60.0, B.X_HI: far})
            w, n_cc, prev, t = w + wi, n_cc + 1, new, float(st.t[-1])
            if B.classify(st) != "time":
                break
        cc_last, cc_class = st, B.classify(st)
        t_cv0, cv_steps = t, []
        for k in (1, 2):
            new, st, wi = c.solve("PCV", [B.step_cv(60)], prev.last_state,
                                  {B.T_STOP: t_cv0 + 60.0 * k, B.X_HI: far})
            w, prev, t = w + wi, new, float(st.t[-1])
            cv_steps.append(st)
        jump_ours = B.snap_diff(B.snapshot(cc_last, -1), B.snapshot(cv_steps[0], 0))
        jump_ref = B.snap_diff(B.snapshot(a_cc, -1), B.snapshot(a_cv, 0))
        da, db = c.degradation_Ah(a_cv), c.degradation_Ah(cv_steps[-1])
        m = {"cc_end_t_s": {"one": float(a_cc.t[-1]), "chunked": float(cc_last.t[-1])},
             "cv_end_t_s": {"one": float(a_cv.t[-1]), "chunked": float(cv_steps[-1].t[-1])},
             "cc_pieces": n_cc, "cc_last_class": cc_class,
             "transition_jump_ours_max_rel": max(v["max_rel_diff"] for v in jump_ours.values()),
             "transition_jump_pybamm_max_rel": max(v["max_rel_diff"] for v in jump_ref.values()),
             "cv_start_current_A": {"one": B.scalar(a_cv, "Current [A]", 0),
                                    "chunked": B.scalar(cv_steps[0], "Current [A]", 0)},
             "end_soc_diff": abs(c.soc(a_cv) - c.soc(cv_steps[-1])),
             "end_voltage_diff_V": abs(B.scalar(a_cv, "Voltage [V]") - B.scalar(cv_steps[-1], "Voltage [V]")),
             "end_charge_Ah_diff_rel_q0": abs(B.scalar(a_cv, "Discharge capacity [A.h]")
                                             - B.scalar(cv_steps[-1], "Discharge capacity [A.h]")) / q0,
             "end_degradation_Ah": {"one": da, "chunked": db},
             "end_degradation_abs_diff_Ah": {k: abs(da[k] - db[k]) for k in ("sei_Ah", "plated_Ah", "dead_Ah")},
             "transition_jump_ours": jump_ours}
        crit = {"cc_ended_on_voltage": cc_class == "voltage",
                "transition_no_reset": m["transition_jump_ours_max_rel"] <= LIM["transition_jump_rel"],
                "soc": m["end_soc_diff"] <= LIM["soc_abs"],
                "voltage": m["end_voltage_diff_V"] <= LIM["voltage_V"],
                "charge_Ah": m["end_charge_Ah_diff_rel_q0"] <= LIM["throughput_rel_q0"],
                **{f"{k}": v / q0 <= LIM["degradation_rel_q0"] for k, v in m["end_degradation_abs_diff_Ah"].items()}}
        levels.append({"tolerance": tol(f), "factor": f, "metrics": m, "criteria": crit, "pass": all(crit.values()),
                       "wall_s": {"one_experiment": wa, "chunked": w, "level_total": time.perf_counter() - t0}})
        dd = m["end_degradation_abs_diff_Ah"]
        log(f"  C x{f:g}: pass={levels[-1]['pass']} jump={m['transition_jump_ours_max_rel']:.1e} "
            f"soc={m['end_soc_diff']:.2e} plated={dd['plated_Ah'] / q0:.2e} dead={dd['dead_Ah'] / q0:.2e}")
        del c, A, prev
        gc.collect()
    REPORT["C_cc_cv"] = {"levels": levels}
    flush()
    return levels


def main():
    t0 = time.perf_counter()
    try:
        q0 = part_a()
        lb = part_b(q0)
        lc = part_c(q0)
        a_ok = REPORT["A_Q0"]["pass"]
        sel = None
        for b_lv, c_lv in zip(lb, lc):
            if b_lv["pass"] and c_lv["pass"]:
                sel = b_lv["tolerance"]
                break
        REPORT["selected_tolerance"] = sel if a_ok else None
        REPORT["all_pass"] = bool(a_ok and sel is not None)
        REPORT["selection_rule"] = ("largest tolerance (x1, x0.1, x0.01 order) at which B and C both pass, "
                                    "given A passes")
    except Exception:
        REPORT["error"] = traceback.format_exc()
        log(REPORT["error"])
    REPORT["wall_total_s"] = time.perf_counter() - t0
    REPORT["memory"] = B.memory_mb()
    flush()
    log(f"all_pass={REPORT.get('all_pass')} selected={REPORT.get('selected_tolerance')} "
        f"wall={REPORT['wall_total_s']:.1f} s")


if __name__ == "__main__":
    main()
