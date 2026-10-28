"""
simulation/verify_chunking.py

Pre-checks before any lifetime run (docs/modeling_decisions.md, section 14).

  check1_input_update   Changing the SOC-boundary input on the SAME built model
                        changes the real termination function value and the stop point.
  check1_real_operation 20->80 % charge (CC, CV if 4.2 V first), rest, 80->20 % discharge,
                        rest, in 60 s pieces with boundary updates. End SOC within +-0.001.
                        Also checks the bookkeeping that lives outside last_state.
  check2_continuity     Same initial state: 600 s continuous vs 10 x 60 s chained.
  check3_last_state     Full-solution chaining vs last_state chaining, then 100 pieces
                        with last_state while recording RAM and kept-solution size.

Run:     python simulation/verify_chunking.py
Output:  $BATTERY_OUTPUT_DIR/verify_chunking/{report.json, log.txt}
         (default <repo>/outputs/verify_chunking/)

Pass/fail never relies on printing an input value. It uses the termination reported
by the solver, the stop time, the SOC recomputed from the end state, and the actual
event expression of the built model evaluated on the end state.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import time
import traceback
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pybamm

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
OUT = Path(os.environ.get("BATTERY_OUTPUT_DIR", ROOT / "outputs")) / "verify_chunking"
OUT.mkdir(parents=True, exist_ok=True)

V = CFG["variables"]
VC = CFG["verification"]
DT = float(CFG["boundary_update_s"])
TOL = float(CFG["soc_tolerance"])
SOC_LO, SOC_HI = (float(s) for s in CFG["soc_window"])
TARGET = "x_target"  # InputParameter used by the SOC termination
F = float(pybamm.constants.F.evaluate())

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(str(msg))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def flush(report: dict) -> None:
    (OUT / "log.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable),
        encoding="utf-8",
    )


# ── memory (Windows working set, no extra dependency) ───────────────
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float | None:
    if sys.platform != "win32":
        return None
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return pmc.WorkingSetSize / 2**20


def solution_size(sol) -> dict:
    n = 0
    for y in sol.all_ys:
        shape = getattr(y, "shape", None)
        if shape:
            n += int(np.prod(shape))
    return {"n_cycles": len(sol.cycles), "n_t": len(sol.t), "n_y_elements": n}


# ── model / parameters / steps ──────────────────────────────────────
def make_param() -> pybamm.ParameterValues:
    p = pybamm.ParameterValues(CFG["parameter_set"])
    p["Ambient temperature [K]"] = CFG["temperature_K"]
    p["Initial temperature [K]"] = CFG["temperature_K"]
    return p


def make_model():
    return pybamm.lithium_ion.SPMe(options=CFG["options"])


def soc_termination(direction: str):
    x_name = V["neg_stoich"]
    if direction == "charge":
        def fn(v):  # positive before the event, zero at the target
            return pybamm.InputParameter(TARGET) - v[x_name]
        return pybamm.step.CustomTermination("SOC upper target", fn)

    def fn(v):
        return v[x_name] - pybamm.InputParameter(TARGET)
    return pybamm.step.CustomTermination("SOC lower target", fn)


def cc_charge_step(current_A: float, dt: float):
    return pybamm.step.current(
        -current_A, duration=dt,
        termination=[f"{CFG['voltage_max_V']} V", soc_termination("charge")],
    )


def cv_step(dt: float):
    return pybamm.step.voltage(
        CFG["voltage_max_V"], duration=dt, termination=[soc_termination("charge")]
    )


def discharge_step(current_A: float, dt: float):
    return pybamm.step.current(
        current_A, duration=dt,
        termination=[f"{CFG['voltage_min_V']} V", soc_termination("discharge")],
    )


def make_sim(step) -> pybamm.Simulation:
    return pybamm.Simulation(
        make_model(), parameter_values=make_param(), experiment=pybamm.Experiment([step])
    )


_esoh_model = make_model()
ESOH = pybamm.lithium_ion.ElectrodeSOHSolver(
    make_param(), param=_esoh_model.param, options=_esoh_model.options
)


# ── readout helpers ─────────────────────────────────────────────────
def last(sol, name: str) -> float:
    a = np.asarray(sol[name].entries, dtype=float)
    return float(a.reshape(-1)[-1]) if a.ndim <= 1 else float(np.mean(a[..., -1]))


def last_step(sol):
    return sol.cycles[-1].steps[-1]


def boundaries(step) -> dict:
    qn = last(step, V["Q_n"])
    qp = last(step, V["Q_p"])
    qli = last(step, V["n_Li_particles"]) * F / 3600.0
    t0 = time.perf_counter()
    r = ESOH.solve({"Q_n": qn, "Q_p": qp, "Q_Li": qli})
    wall = time.perf_counter() - t0
    if "x_0" not in r or "x_100" not in r:
        raise KeyError(f"eSOH output lacks x_0/x_100. keys={sorted(r)[:40]}")
    return {
        "x0": float(np.ravel(r["x_0"])[0]),
        "x100": float(np.ravel(r["x_100"])[0]),
        "Q_n_Ah": qn, "Q_p_Ah": qp, "Q_Li_Ah": qli, "esoh_wall_s": wall,
    }


def soc_from_x(x: float, b: dict) -> float:
    return (x - b["x0"]) / (b["x100"] - b["x0"])


def x_from_soc(s: float, b: dict) -> float:
    return b["x0"] + s * (b["x100"] - b["x0"])


def classify(step, dt: float) -> tuple[str, str]:
    term = str(step.termination)
    low = term.lower()
    if "soc upper target" in low or "soc lower target" in low:
        return "soc", term
    if "voltage" in low:
        return "voltage", term
    if "final time" in low:
        return "time", term
    dur = float(step.t[-1] - step.t[0])
    if abs(dur - dt) < 1e-6:
        return "time", term
    return "other", term


def solve_piece(sim, start, inputs: dict):
    t0 = time.perf_counter()
    sol = sim.solve(starting_solution=start, inputs=inputs, calc_esoh=False)
    return sol, last_step(sol), time.perf_counter() - t0


def event_value(sim, step, inputs: dict, event_name: str):
    """Evaluate the real termination expression of the built model on the end state."""
    try:
        y = np.asarray(step.all_ys[-1], dtype=float)[:, -1].reshape(-1, 1)
        for m in sim.steps_to_built_models.values():
            for ev in m.events:
                if event_name in ev.name:
                    try:
                        val = ev.expression.evaluate(t=float(step.t[-1]), y=y, inputs=inputs)
                        return float(np.ravel(val)[0])
                    except Exception:
                        continue
    except Exception as e:
        return f"evaluate failed: {type(e).__name__}: {e}"
    return "event not found or not evaluable"


def state_snapshot(step) -> dict:
    snap = {}
    for key, cands in V["state_compare"].items():
        snap[key] = None
        for name in cands:
            try:
                a = np.asarray(step[name].entries, dtype=float)
            except Exception:
                continue
            snap[key] = {"name": name, "value": a[..., -1] if a.ndim > 1 else a[-1:]}
            break
    return snap


def compare(sa: dict, sb: dict) -> dict:
    out = {}
    for key in sa:
        if sa[key] is None or sb[key] is None:
            out[key] = {"status": "variable not found"}
            continue
        a = np.ravel(sa[key]["value"])
        b = np.ravel(sb[key]["value"])
        if a.shape != b.shape:
            out[key] = {"status": f"shape mismatch {a.shape} vs {b.shape}"}
            continue
        d = float(np.max(np.abs(a - b)))
        s = float(np.max(np.abs(a)))
        out[key] = {
            "name": sa[key]["name"], "max_abs_diff": d, "ref_max_abs": s,
            "max_rel_diff": d / s if s > 0 else (0.0 if d == 0 else float("inf")),
            "nonzero_state": s > 0,
        }
    return out


# ── bookkeeping that is NOT inside last_state ───────────────────────
@dataclass
class RunState:
    sim_time_s: float = 0.0        # since start of this run (excludes the init rest)
    stage: str = "init"
    charge_start_s: float | None = None
    cv_start_s: float | None = None
    rest_remaining_s: float | None = None
    discharge_Ah: float = 0.0      # main-trajectory discharge throughput
    charge_Ah: float = 0.0
    efc: float | None = None       # needs Q0 from the reference test
    efc_note: str = "Q0 not measured yet (reference test not run)"
    wall_time_s: float = 0.0
    pieces: int = 0
    piece_durations: list = field(default_factory=list)


# ── checks ──────────────────────────────────────────────────────────
def initial_state():
    sim = make_sim(pybamm.step.rest(duration=VC["init_rest_s"]))
    sol = sim.solve(initial_soc=SOC_LO, calc_esoh=False)
    st = last_step(sol)
    b = boundaries(st)
    x = last(st, V["neg_stoich"])
    missing = []
    for key in ("neg_stoich", "Q_n", "Q_p", "n_Li_particles", "current", "voltage",
                "discharge_capacity"):
        try:
            st[V[key]]
        except Exception:
            missing.append(V[key])
    info = {
        "init_rest_s": VC["init_rest_s"],
        "soc_requested": SOC_LO,
        "soc_from_esoh": soc_from_x(x, b),
        "boundaries": b,
        "equilibrium_window_Ah": b["Q_n_Ah"] * (b["x100"] - b["x0"]),
        "voltage_V": last(st, V["voltage"]),
        "missing_variables": missing,
        "t_end_s": float(st.t[-1]),
    }
    return sol, info


def check1_input_update(S0) -> dict:
    I = VC["check_current_A"]
    sim = make_sim(cc_charge_step(I, DT))   # built once, reused for all three solves
    st0 = last_step(S0)
    b0 = boundaries(st0)
    soc0 = soc_from_x(last(st0, V["neg_stoich"]), b0)
    runs = {}
    cases = [("A", VC["check1_dsoc"][0]), ("B", VC["check1_dsoc"][1]),
             ("control", VC["check1_control_dsoc"])]
    for label, dsoc in cases:
        s_t = soc0 + dsoc
        xt = x_from_soc(s_t, b0)
        sol, st, wall = solve_piece(sim, S0.last_state, {TARGET: xt})
        cls, term = classify(st, DT)
        x_end = last(st, V["neg_stoich"])
        b_end = boundaries(st)
        runs[label] = {
            "target_soc": s_t, "x_target": xt,
            "termination": term, "class": cls,
            "duration_s": float(st.t[-1] - st.t[0]),
            "event_fn_model_at_end": event_value(sim, st, {TARGET: xt}, "SOC upper target"),
            "event_fn_manual_at_end": xt - x_end,
            "soc_end_start_boundaries": soc_from_x(x_end, b0),
            "soc_end_end_boundaries": soc_from_x(x_end, b_end),
            "wall_s": wall,
        }
        runs[label]["soc_error"] = runs[label]["soc_end_end_boundaries"] - s_t
    a, b, c = runs["A"], runs["B"], runs["control"]
    criteria = {
        "A_and_B_stopped_by_soc_event": a["class"] == "soc" and b["class"] == "soc",
        "stop_time_differs_A_lt_B": a["duration_s"] < b["duration_s"],
        "control_ran_full_piece": c["class"] == "time" and abs(c["duration_s"] - DT) < 1e-6,
        "A_soc_error_within_tol": abs(a["soc_error"]) <= TOL,
        "B_soc_error_within_tol": abs(b["soc_error"]) <= TOL,
    }
    return {"soc_start": soc0, "boundaries_start": b0, "runs": runs,
            "criteria": criteria, "pass": all(criteria.values())}


def check1_real_operation(S0) -> dict:
    I_ch = VC["check_current_A"]
    I_dis = CFG["currents_A"]["discharge"]
    rest_s = float(CFG["rest_s"])
    if abs(rest_s / DT - round(rest_s / DT)) > 1e-9:
        raise ValueError("rest_s must be a multiple of boundary_update_s")
    sims = {
        "CC": make_sim(cc_charge_step(I_ch, DT)),
        "CV": make_sim(cv_step(DT)),
        "REST": make_sim(pybamm.step.rest(duration=DT)),
        "DIS": make_sim(discharge_step(I_dis, DT)),
    }
    rs = RunState()
    t_origin = float(last_step(S0).t[-1])
    prev = S0
    rows = []
    max_n = int(VC["max_pieces_per_phase"])

    def piece(stage_key: str, soc_target: float | None):
        nonlocal prev
        b = boundaries(last_step(prev))   # boundary update every piece, rest included
        inputs = {} if soc_target is None else {TARGET: x_from_soc(soc_target, b)}
        sol, st, wall = solve_piece(sims[stage_key], prev.last_state, inputs)
        dur = float(st.t[-1] - st.t[0])
        qd = np.asarray(st[V["discharge_capacity"]].entries, dtype=float)
        ah = float(qd[-1] - qd[0])
        rs.sim_time_s += dur
        rs.wall_time_s += wall + b["esoh_wall_s"]
        rs.pieces += 1
        rs.piece_durations.append(dur)
        if ah >= 0:
            rs.discharge_Ah += ah
        else:
            rs.charge_Ah += -ah
        cls, term = classify(st, DT)
        x_end = last(st, V["neg_stoich"])
        rows.append({
            "piece": rs.pieces, "stage": rs.stage, "model": stage_key,
            "t_abs_end_s": float(st.t[-1]), "duration_s": dur, "class": cls,
            "termination": term, "I_end_A": last(st, V["current"]),
            "V_end_V": last(st, V["voltage"]),
            "soc_end_piece_start_boundaries": soc_from_x(x_end, b),
            "ah": ah, "wall_s": wall, "esoh_wall_s": b["esoh_wall_s"],
        })
        prev = sol
        return st, cls

    # charge: CC, switch to CV on 4.2 V, stop on SOC 80 %
    rs.stage, rs.charge_start_s = "CHARGE_CC", rs.sim_time_s
    stage_key, charge_outcome = "CC", "max_pieces"
    for _ in range(max_n):
        st, cls = piece(stage_key, SOC_HI)
        if cls == "soc":
            charge_outcome = "soc_target"
            break
        if cls == "voltage" and stage_key == "CC":
            stage_key, rs.stage, rs.cv_start_s = "CV", "CHARGE_CV", rs.sim_time_s
            continue
        if cls == "time":
            if stage_key == "CV" and abs(last(st, V["current"])) < VC["cv_stall_current_A"]:
                charge_outcome = "cv_current_stall"
                break
            continue
        charge_outcome = f"unexpected:{cls}"
        break
    st_ch = last_step(prev)
    soc_charge_end = soc_from_x(last(st_ch, V["neg_stoich"]), boundaries(st_ch))
    charge_time_s = rs.sim_time_s - rs.charge_start_s
    charge_pieces_sum = sum(r["duration_s"] for r in rows)

    rs.stage, rs.rest_remaining_s = "REST_AFTER_CHARGE", rest_s
    while rs.rest_remaining_s > 1e-9:
        st, _ = piece("REST", None)
        rs.rest_remaining_s -= float(st.t[-1] - st.t[0])

    rs.stage = "DISCHARGE"
    discharge_outcome = "max_pieces"
    n_before = len(rows)
    for _ in range(max_n):
        st, cls = piece("DIS", SOC_LO)
        if cls == "soc":
            discharge_outcome = "soc_target"
            break
        if cls == "voltage":
            discharge_outcome = "early_voltage_cutoff"
            break
        if cls == "time":
            continue
        discharge_outcome = f"unexpected:{cls}"
        break
    st_dis = last_step(prev)
    soc_discharge_end = soc_from_x(last(st_dis, V["neg_stoich"]), boundaries(st_dis))
    discharge_time_s = sum(r["duration_s"] for r in rows[n_before:])

    rs.stage, rs.rest_remaining_s = "REST_AFTER_DISCHARGE", rest_s
    while rs.rest_remaining_s > 1e-9:
        st, _ = piece("REST", None)
        rs.rest_remaining_s -= float(st.t[-1] - st.t[0])
    rs.stage = "DONE"

    t_abs_end = float(last_step(prev).t[-1])
    book = {
        "pybamm_time_continuity_error_s": t_abs_end - (t_origin + rs.sim_time_s),
        "charge_time_vs_piece_sum_error_s": charge_time_s - charge_pieces_sum,
        "rest_remaining_end_s": rs.rest_remaining_s,
    }
    criteria = {
        "charge_stopped_by_soc_target": charge_outcome == "soc_target",
        "charge_end_soc_within_tol": abs(soc_charge_end - SOC_HI) <= TOL,
        "discharge_stopped_by_soc_target": discharge_outcome == "soc_target",
        "discharge_end_soc_within_tol": abs(soc_discharge_end - SOC_LO) <= TOL,
        "pybamm_time_continuous": abs(book["pybamm_time_continuity_error_s"]) < 1e-6,
        "charge_time_bookkeeping_consistent": abs(book["charge_time_vs_piece_sum_error_s"]) < 1e-6,
        "rest_fully_consumed": abs(rs.rest_remaining_s) < 1e-6,
    }
    sim_h = rs.sim_time_s / 3600.0
    return {
        "charge_current_A": I_ch, "discharge_current_A": I_dis,
        "charge_outcome": charge_outcome, "charge_time_min": charge_time_s / 60.0,
        "cv_entered": rs.cv_start_s is not None,
        "soc_charge_end": soc_charge_end,
        "discharge_outcome": discharge_outcome, "discharge_time_min": discharge_time_s / 60.0,
        "soc_discharge_end": soc_discharge_end,
        "charge_Ah": rs.charge_Ah, "discharge_Ah": rs.discharge_Ah,
        "bookkeeping": book,
        "run_state_end": {k: v for k, v in asdict(rs).items() if k != "piece_durations"},
        "cost": {"pieces": rs.pieces, "sim_hours": sim_h, "wall_s": rs.wall_time_s,
                 "wall_s_per_sim_hour": rs.wall_time_s / sim_h if sim_h > 0 else None,
                 "esoh_wall_s_total": sum(r["esoh_wall_s"] for r in rows)},
        "criteria": criteria, "pass": all(criteria.values()),
        "pieces": rows,
        "note": "First charge of a fresh cell at 25 degC only.",
    }


def check2_continuity(S0) -> dict:
    I = VC["check_current_A"]
    total = float(VC["check2_total_s"])
    n = int(VC["check2_pieces"])
    vmax = f"{CFG['voltage_max_V']} V"
    sim_long = make_sim(pybamm.step.current(-I, duration=total, termination=[vmax]))
    sim_short = make_sim(pybamm.step.current(-I, duration=total / n, termination=[vmax]))
    _, st_long, w_long = solve_piece(sim_long, S0.last_state, {})
    prev, w_short = S0.last_state, 0.0
    for _ in range(n):
        sol, st_short, w = solve_piece(sim_short, prev, {})
        prev, w_short = sol, w_short + w   # full-solution chaining
    diff = compare(state_snapshot(st_long), state_snapshot(st_short))
    plating_nonzero = any(diff.get(k, {}).get("nonzero_state") for k in ("plated_li", "dead_li"))
    return {
        "current_A": I, "total_s": total, "pieces": n,
        "t_end_long_s": float(st_long.t[-1]), "t_end_chained_s": float(st_short.t[-1]),
        "termination_long": str(st_long.termination),
        "state_diff": diff,
        "plating_state_nonzero": plating_nonzero,
        "plating_transfer_verified": plating_nonzero,
        "wall_s": {"continuous": w_long, "chained": w_short},
        "note": "CC only. The CC->CV transition is not part of this comparison.",
    }


def check3_last_state(S0) -> dict:
    I = VC["check_current_A"]
    sim_c = make_sim(cc_charge_step(I, DT))
    sim_d = make_sim(discharge_step(I, DT))

    def alternate(n: int, use_last_state: bool, record_mem: bool):
        prev, mem, classes = S0, [], []
        for k in range(n):
            charging = k % 2 == 0
            b = boundaries(last_step(prev))
            xt = x_from_soc(0.99 if charging else 0.01, b)   # far targets: pieces run full DT
            start = prev.last_state if use_last_state else prev
            sol, st, wall = solve_piece(sim_c if charging else sim_d, start, {TARGET: xt})
            classes.append(classify(st, DT)[0])
            prev = sol
            if record_mem:
                gc.collect()
                mem.append({"piece": k + 1, "rss_mb": rss_mb(), "wall_s": wall,
                            **solution_size(sol)})
        return prev, mem, classes

    n_cmp = int(VC["check3_compare_pieces"])
    full, _, cls_full = alternate(n_cmp, False, False)
    lst, _, cls_last = alternate(n_cmp, True, False)
    diff = compare(state_snapshot(last_step(full)), state_snapshot(last_step(lst)))

    n_mem = int(VC["check3_memory_pieces"])
    gc.collect()
    rss_before = rss_mb()
    _, mem, cls_mem = alternate(n_mem, True, True)
    rss = [m["rss_mb"] for m in mem if m["rss_mb"] is not None]
    slope = None
    if len(rss) >= 20:
        xs = np.arange(10, len(rss))
        slope = float(np.polyfit(xs, np.asarray(rss[10:]), 1)[0])
    return {
        "compare_pieces": n_cmp,
        "t_end_full_s": float(last_step(full).t[-1]),
        "t_end_last_state_s": float(last_step(lst).t[-1]),
        "classes_full": cls_full, "classes_last_state": cls_last,
        "state_diff_full_vs_last_state": diff,
        "full_solution_size_after_compare": solution_size(full),
        "memory": {
            "pieces": n_mem, "rss_before_mb": rss_before,
            "rss_piece10_mb": rss[9] if len(rss) >= 10 else None,
            "rss_last_mb": rss[-1] if rss else None,
            "rss_slope_mb_per_piece_after10": slope,
            "kept_solution_size_last": {k: mem[-1][k] for k in ("n_cycles", "n_t", "n_y_elements")}
            if mem else None,
            "non_time_classes": [c for c in cls_mem if c != "time"],
            "trace": mem,
        },
    }


def main() -> None:
    report = {
        "pybamm_version": pybamm.__version__,
        "python": sys.version,
        "config": CFG,
        "applied_options": {k: str(make_model().options[k]) for k in CFG["options"]},
        "checks": {},
    }
    t0 = time.perf_counter()
    try:
        S0, info = initial_state()
        report["initial_state"] = info
        log(f"initial state: soc={info['soc_from_esoh']:.5f}, "
            f"window={info['equilibrium_window_Ah']:.4f} Ah, missing={info['missing_variables']}")
    except Exception:
        report["initial_state"] = {"status": "error", "traceback": traceback.format_exc()}
        log(report["initial_state"]["traceback"])
        flush(report)
        return
    flush(report)

    for name, fn in (("check1_input_update", check1_input_update),
                     ("check1_real_operation", check1_real_operation),
                     ("check2_continuity", check2_continuity),
                     ("check3_last_state", check3_last_state)):
        log(f"--- {name}")
        t = time.perf_counter()
        try:
            res = fn(S0)
        except Exception:
            res = {"status": "error", "traceback": traceback.format_exc()}
            log(res["traceback"])
        res["wall_total_s"] = time.perf_counter() - t
        report["checks"][name] = res
        if "pass" in res:
            log(f"    pass={res['pass']}  criteria={res.get('criteria')}")
        flush(report)

    report["wall_total_s"] = time.perf_counter() - t0
    flush(report)
    log(f"report: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
