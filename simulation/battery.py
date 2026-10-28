"""
simulation/battery.py

Shared simulation code. Verification scripts and experiment runners import this module,
so the code that is verified is the code that runs (revised plan section 12).
Physical conditions come only from configs/*.json.

Control (revised plan sections 4-6)
  * One Experiment model per step kind (CC, CV, REST, DIS), built once and reused.
  * A piece ends at the first of: SOC event, voltage event, or the absolute time
    input `t_stop`. `t_stop` is the next scheduled point (boundary update, rest end,
    EFC target), so piece lengths follow the schedule instead of a fixed 60 s.
  * eSOH boundaries are updated every `dt` of simulated time and at every phase
    transition. If the updated boundaries show the SOC target is already met, the
    phase ends without solving another piece.
  * CC charge -> CV on 4.2 V. CV ends on SOC target. SOC first -> CV skipped.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pybamm

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
DIAG_CFG = json.loads((ROOT / "configs" / "diagnostics.json").read_text(encoding="utf-8"))

VARS = MODEL_CFG["variables"]
X = VARS["neg_stoich"]
F = float(pybamm.constants.F.evaluate())
DEFAULT_RTOL, DEFAULT_ATOL = 1e-4, 1e-6        # IDAKLUSolver defaults (idaklu_solver.py:240)
SOC_LO, SOC_HI = (float(s) for s in MODEL_CFG["soc_window"])
SOC_TOL = float(MODEL_CFG["soc_tolerance"])
V_MAX, V_MIN = float(MODEL_CFG["voltage_max_V"]), float(MODEL_CFG["voltage_min_V"])
T_STOP, X_HI, X_LO = "t_stop", "x_target_hi", "x_target_lo"

# variables compared for state continuity (differential states + their derived values)
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


# ── parameters / model ──────────────────────────────────────────────
def make_param(overrides: dict | None = None) -> pybamm.ParameterValues:
    p = pybamm.ParameterValues(MODEL_CFG["parameter_set"])
    p["Ambient temperature [K]"] = MODEL_CFG["temperature_K"]
    p["Initial temperature [K]"] = MODEL_CFG["temperature_K"]
    if overrides:
        p.update(overrides)
    return p


def make_model():
    return pybamm.lithium_ion.SPMe(options=MODEL_CFG["options"])


# ── steps ───────────────────────────────────────────────────────────
def _soc_term(direction: str):
    if direction == "charge":
        return pybamm.step.CustomTermination(
            "SOC upper target", lambda v: pybamm.InputParameter(X_HI) - v[X])
    return pybamm.step.CustomTermination(
        "SOC lower target", lambda v: v[X] - pybamm.InputParameter(X_LO))


def _piece_end():
    return pybamm.step.CustomTermination(
        "piece end", lambda v: pybamm.InputParameter(T_STOP) - pybamm.t)


def step_cc(I, dt, period=None, piece=True):
    term = [f"{V_MAX} V", _soc_term("charge")] + ([_piece_end()] if piece else [])
    return pybamm.step.current(-abs(I), duration=dt, period=period, termination=term)


def step_cv(dt, period=None, piece=True):
    term = [_soc_term("charge")] + ([_piece_end()] if piece else [])
    return pybamm.step.voltage(V_MAX, duration=dt, period=period, termination=term)


def step_rest(dt, period=None, piece=True):
    term = [_piece_end()] if piece else None
    return pybamm.step.rest(duration=dt, period=period, termination=term)


def step_dis(I, dt, period=None, piece=True):
    term = [f"{V_MIN} V", _soc_term("discharge")] + ([_piece_end()] if piece else [])
    return pybamm.step.current(abs(I), duration=dt, period=period, termination=term)


def classify(step) -> str:
    t = str(step.termination).lower()
    if "soc upper target" in t or "soc lower target" in t:
        return "soc"
    if "piece end" in t or "final time" in t:
        return "time"
    if "voltage" in t:
        return "voltage"
    if "current" in t:
        return "current"
    return "other"


# ── readout ─────────────────────────────────────────────────────────
def last_step(sol):
    return sol.cycles[-1].steps[-1] if sol.cycles else sol


def val(step, name, index=-1) -> np.ndarray:
    a = np.asarray(step[name].entries, dtype=float)
    return np.ravel(a[..., index])


def scalar(step, name, index=-1) -> float:
    return float(val(step, name, index)[0])


def y_end(sol) -> np.ndarray:
    return np.asarray(sol.all_ys[-1], dtype=float)[:, -1].copy()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(a)))
    return {"max_abs_diff": d, "ref_max_abs": s,
            "max_rel_diff": d / s if s > 0 else (0.0 if d == 0 else float("inf"))}


def snapshot(step, index=-1) -> dict:
    return {k: val(step, n, index) for k, n in STATE_VARS.items()}


def snap_diff(sa: dict, sb: dict) -> dict:
    return {k: compare(sa[k], sb[k]) if sa[k].shape == sb[k].shape
            else {"status": f"shape {sa[k].shape} vs {sb[k].shape}"} for k in sa}


# ── memory ──────────────────────────────────────────────────────────
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_mb() -> dict:
    if sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = {"peak_rss_mb": peak / (2**20 if sys.platform == "darwin" else 1024)}
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    out["rss_mb"] = float(line.split()[1]) / 1024
                    break
        return out
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return {"rss_mb": pmc.WorkingSetSize / 2**20, "peak_rss_mb": pmc.PeakWorkingSetSize / 2**20}


def jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── operating ledger (state outside last_state) ─────────────────────
@dataclass
class RunState:
    cell_id: int
    trajectory_id: str
    q0_Ah: float | None = None
    stage: str = "init"
    t_abs_s: float = 0.0
    next_boundary_t_s: float | None = None
    cycle: int = 0
    charge_start_s: float | None = None
    cv_start_s: float | None = None
    rest_remaining_s: float | None = None
    charge_Ah: float = 0.0
    discharge_Ah: float = 0.0
    efc: float = 0.0
    time_violations: list = field(default_factory=list)
    wall_time_s: float = 0.0
    diagnostic_wall_s: float = 0.0
    pieces: int = 0


# ── cell ────────────────────────────────────────────────────────────
class Cell:
    def __init__(self, overrides=None, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, mesh_scale=1.0):
        self.overrides = dict(overrides or {})
        self.param = make_param(self.overrides)
        self.rtol, self.atol, self.mesh_scale = rtol, atol, mesh_scale
        m = make_model()
        self.var_pts = None if mesh_scale == 1 else {
            k: int(round(v * mesh_scale)) for k, v in m.default_var_pts.items()}
        self.esoh = pybamm.lithium_ion.ElectrodeSOHSolver(self.param, param=m.param, options=m.options)
        self.esoh_calls, self.esoh_wall_s = 0, 0.0
        p = self.param
        self.neg_volume_m3 = (p["Negative electrode thickness [m]"] * p["Electrode height [m]"]
                              * p["Electrode width [m]"])
        self._sims: dict = {}
        self.builds = 0

    def sim(self, key, steps) -> pybamm.Simulation:
        if key not in self._sims:
            self._sims[key] = pybamm.Simulation(
                make_model(), parameter_values=self.param.copy(),
                experiment=pybamm.Experiment([tuple(steps)]),
                solver=pybamm.IDAKLUSolver(rtol=self.rtol, atol=self.atol),
                var_pts=self.var_pts)
            self.builds += 1
        return self._sims[key]

    def solve(self, key, steps, start, inputs=None):
        sim = self.sim(key, steps)
        t0 = time.perf_counter()
        sol = sim.solve(starting_solution=start, inputs=inputs or {}, calc_esoh=False)
        return sol, last_step(sol), time.perf_counter() - t0

    def initial_state(self, soc=SOC_LO, rest_s=1.0):
        sim = self.sim(("INIT", rest_s), [pybamm.step.rest(duration=rest_s)])
        return sim.solve(initial_soc=soc, calc_esoh=False)

    # eSOH boundaries from the current electrode state (revised plan section 5)
    def boundaries_from(self, qn, qp, n_li) -> dict:
        t0 = time.perf_counter()
        r = self.esoh.solve({"Q_n": qn, "Q_p": qp, "Q_Li": n_li * F / 3600.0})
        self.esoh_wall_s += time.perf_counter() - t0
        self.esoh_calls += 1
        return {"x0": float(np.ravel(r["x_0"])[0]), "x100": float(np.ravel(r["x_100"])[0])}

    def boundaries(self, step, index=-1) -> dict:
        return self.boundaries_from(scalar(step, VARS["Q_n"], index), scalar(step, VARS["Q_p"], index),
                                    scalar(step, VARS["n_Li_particles"], index))

    def soc(self, step, index=-1, b=None) -> float:
        b = b or self.boundaries(step, index)
        return soc_from_x(scalar(step, X, index), b)

    def degradation_Ah(self, step, index=-1) -> dict:
        k = self.neg_volume_m3 * F / 3600.0
        plated = scalar(step, "Volume-averaged negative lithium plating concentration [mol.m-3]", index) * k
        dead = scalar(step, "Volume-averaged negative dead lithium concentration [mol.m-3]", index) * k
        return {"sei_Ah": scalar(step, "Loss of capacity to negative SEI [A.h]", index),
                "plated_Ah": plated, "dead_Ah": dead,
                "plating_total_check_Ah": scalar(step, "Loss of capacity to negative lithium plating [A.h]", index)
                - (plated + dead)}

    # ── diagnostics ────────────────────────────────────────────────
    def reference_test(self, start) -> dict:
        r = DIAG_CFG["reference_test"]
        steps = [
            pybamm.step.current(-r["charge_current_A"], duration=6 * 3600,
                                termination=f"{r['charge_voltage_V']} V"),
            pybamm.step.voltage(r["charge_voltage_V"], duration=6 * 3600,
                                termination=f"{r['cv_cutoff_current_A']} A"),
            pybamm.step.rest(duration=r["rest_s"]),
            pybamm.step.current(r["discharge_current_A"], duration=24 * 3600,
                                termination=f"{r['discharge_cutoff_V']} V"),
        ]
        sol, _, wall = self.solve("REF", steps, start)
        st = sol.cycles[-1].steps
        names = ["cc_charge", "cv_hold", "rest", "discharge"]
        out = {n: {"termination": str(s.termination), "duration_s": float(s.t[-1] - s.t[0])}
               for n, s in zip(names, st)}
        dis = st[3]
        t = np.asarray(dis["Time [s]"].entries, dtype=float)
        i = np.asarray(dis["Current [A]"].entries, dtype=float)
        q = np.asarray(dis["Discharge capacity [A.h]"].entries, dtype=float)
        out["Q_trapz_Ah"] = float(np.trapezoid(i, t) / 3600.0)
        out["Q_cum_Ah"] = float(q[-1] - q[0])
        out["ok"] = (classify(st[0]) == "voltage" and classify(st[1]) == "current"
                     and classify(st[3]) == "voltage")
        out["wall_s"] = wall
        return out

    def r10_test(self, start_sol) -> dict:
        """SOC 50 % at 0.5 A (implementation choice), 30 min rest, 5 A for 10 s."""
        st0 = last_step(start_sol)
        b = self.boundaries(st0)
        soc0 = soc_from_x(scalar(st0, X), b)
        x50 = x_from_soc(0.5, b)
        if soc0 > 0.5:
            pos = pybamm.step.current(0.5, duration=48 * 3600,
                                      termination=[f"{V_MIN} V", _soc_term("discharge")])
            key, inputs = "R10_dis", {X_LO: x50}
        else:
            pos = pybamm.step.current(-0.5, duration=48 * 3600,
                                      termination=[f"{V_MAX} V", _soc_term("charge")])
            key, inputs = "R10_chg", {X_HI: x50}
        steps = [pos, pybamm.step.rest(duration=1800),
                 pybamm.step.current(5.0, duration=10, period=0.1)]
        sol, _, wall = self.solve(key, steps, start_sol.last_state.copy(), inputs)
        p, r, pulse = sol.cycles[-1].steps
        v_before = scalar(r, VARS["voltage"])
        v_10 = scalar(pulse, VARS["voltage"])
        return {"soc_before_positioning": soc0,
                "soc_after_positioning": self.soc(p),
                "positioning_termination": str(p.termination),
                "rest_termination": str(r.termination),
                "pulse_duration_s": float(pulse.t[-1] - pulse.t[0]),
                "V_before_V": v_before, "V_10s_V": v_10,
                "R10_ohm": (v_before - v_10) / 5.0, "wall_s": wall,
                "ok": classify(p) == "soc" and abs(float(pulse.t[-1] - pulse.t[0]) - 10) < 1e-6}


def soc_from_x(x, b):
    return (x - b["x0"]) / (b["x100"] - b["x0"])


def x_from_soc(s, b):
    return b["x0"] + s * (b["x100"] - b["x0"])


# ── daily operation controller ──────────────────────────────────────
def run_operation(cell: Cell, sol, rs: RunState, I_ch: float, I_dis: float, dt: float,
                  max_cycles: int | None = None, efc_stop: float | None = None,
                  charge_limit_s: float | None = None, max_pieces: int = 200000,
                  cv_stall_A: float = 1e-3, output_period_s=None,
                  on_piece=None, diagnostic_efcs=None, on_diagnostic=None):
    """Returns (solution, cycles, status, checks). Requires rs.q0_Ah for EFC."""
    rest_s = float(MODEL_CFG["rest_s"])
    cycles, status = [], None
    chk = {"max_same_model_join_y_jump": 0.0, "transition_jumps": [], "nonfinite_pieces": 0,
           "time_join_max_error_s": 0.0, "min_values": {}, "porosity_range": [1.0, 0.0],
           "pieces_by_class": {}}
    state = {"sol": sol, "b": None, "prev_kind": None}
    targets = iter(sorted(set(float(x) for x in (diagnostic_efcs or [])
                              if float(x) > rs.efc + 1e-9)))
    next_diagnostic = next(targets, None)

    def update_boundaries():
        state["b"] = cell.boundaries(last_step(state["sol"]))
        rs.next_boundary_t_s = rs.t_abs_s + dt

    def track_values(st):
        for k in ("plated_li", "dead_li", "sei_thickness", "electrolyte_conc",
                  "neg_particle_conc", "pos_particle_conc"):
            v = float(np.min(np.asarray(st[STATE_VARS[k]].entries, dtype=float)))
            chk["min_values"][k] = min(chk["min_values"].get(k, np.inf), v)
        por = np.asarray(st[STATE_VARS["neg_porosity"]].entries, dtype=float)
        chk["porosity_range"][0] = min(chk["porosity_range"][0], float(por.min()))
        chk["porosity_range"][1] = max(chk["porosity_range"][1], float(por.max()))

    def piece(kind, steps, phase_end_t, soc_target, x_name):
        nonlocal next_diagnostic
        if rs.pieces >= max_pieces:
            return "max_pieces", None
        if rs.t_abs_s >= rs.next_boundary_t_s - 1e-9:
            update_boundaries()
        b = state["b"]
        if soc_target is not None:
            s_now = soc_from_x(scalar(last_step(state["sol"]), X), b)
            if (x_name == X_HI and s_now >= soc_target) or (x_name == X_LO and s_now <= soc_target):
                return "soc_already", None
        t_stop = min(rs.next_boundary_t_s, phase_end_t)
        if t_stop - rs.t_abs_s <= 1e-9:
            return "zero_length", None
        inputs = {T_STOP: t_stop}
        if soc_target is not None:
            inputs[x_name] = x_from_soc(soc_target, b)
        prev_sol, prev_step = state["sol"], last_step(state["sol"])
        # one built model per (kind, current, dt); steps differ in current and duration
        key = (kind, I_ch if kind in ("CC", "CV") else (I_dis if kind == "DIS" else 0.0), dt,
               output_period_s)
        new, st, wall = cell.solve(key, [steps], prev_sol.last_state, inputs)
        rs.wall_time_s += wall
        rs.pieces += 1
        dur = float(st.t[-1] - st.t[0])
        chk["time_join_max_error_s"] = max(chk["time_join_max_error_s"], abs(float(st.t[0]) - rs.t_abs_s))
        y_new = np.asarray(st.all_ys[0], dtype=float)
        if not np.all(np.isfinite(np.asarray(st.all_ys[-1], dtype=float))):
            chk["nonfinite_pieces"] += 1
        if state["prev_kind"] == kind:
            y_prev = np.asarray(prev_step.all_ys[-1], dtype=float)[:, -1]
            if y_prev.shape == y_new[:, 0].shape:
                chk["max_same_model_join_y_jump"] = max(
                    chk["max_same_model_join_y_jump"], float(np.max(np.abs(y_new[:, 0] - y_prev))))
        elif state["prev_kind"] is not None:
            d = snap_diff(snapshot(prev_step, -1), snapshot(st, 0))
            chk["transition_jumps"].append({
                "from": state["prev_kind"], "to": kind, "t_s": rs.t_abs_s,
                "max_rel": max((v.get("max_rel_diff", 0.0) for k2, v in d.items()
                                if k2 != "discharge_capacity"), default=0.0)})
            track_values(prev_step)
        q = np.asarray(st[VARS["discharge_capacity"]].entries, dtype=float)
        dq = float(q[-1] - q[0])
        if dq >= 0:
            rs.discharge_Ah += dq
        else:
            rs.charge_Ah += -dq
        rs.efc = rs.discharge_Ah / rs.q0_Ah if rs.q0_Ah else 0.0
        rs.t_abs_s = float(st.t[-1])
        state["sol"], state["prev_kind"] = new, kind
        cls = classify(st) if dur > 1e-9 else "no_progress"
        chk["pieces_by_class"][f"{kind}:{cls}"] = chk["pieces_by_class"].get(f"{kind}:{cls}", 0) + 1
        if on_piece is not None:
            on_piece(new, st, kind, b, rs)
        if kind == "DIS" and next_diagnostic is not None and rs.efc >= next_diagnostic - 1e-9:
            if on_diagnostic is not None:
                on_diagnostic(new, rs, next_diagnostic)
            next_diagnostic = next(targets, None)
        return cls, st

    rs.next_boundary_t_s = rs.t_abs_s  # force an update on the first piece
    while status is None:
        cyc = {"cycle": rs.cycle + 1, "efc_start": rs.efc}
        # ---- charge
        rs.stage, rs.charge_start_s, rs.cv_start_s = "CHARGE_CC", rs.t_abs_s, None
        rs.next_boundary_t_s = rs.t_abs_s
        kind, cc_s, cv_s, end = "CC", step_cc(I_ch, dt, output_period_s), step_cv(dt, output_period_s), None
        while end is None:
            cls, st = piece(kind, cc_s if kind == "CC" else cv_s, np.inf, SOC_HI, X_HI)
            if cls in ("soc", "soc_already"):
                end = cls
            elif cls == "voltage" and kind == "CC":
                kind, rs.stage, rs.cv_start_s = "CV", "CHARGE_CV", rs.t_abs_s
                rs.next_boundary_t_s = rs.t_abs_s
            elif cls == "time":
                if kind == "CV" and abs(scalar(st, VARS["current"])) < cv_stall_A:
                    end, status = "cv_current_stall", "failed:cv_current_stall"
            else:
                end, status = cls, f"failed:charge:{cls}"
        track_values(last_step(state["sol"]))
        update_boundaries()
        cyc.update({
            "charge_end": end, "cv_entered": rs.cv_start_s is not None,
            "charge_time_s": rs.t_abs_s - rs.charge_start_s,
            "cv_time_s": (rs.t_abs_s - rs.cv_start_s) if rs.cv_start_s is not None else 0.0,
            "soc_charge_end": soc_from_x(scalar(last_step(state["sol"]), X), state["b"])})
        cyc["soc_charge_error"] = cyc["soc_charge_end"] - SOC_HI
        if charge_limit_s is not None and cyc["charge_time_s"] > charge_limit_s:
            rs.time_violations.append({"cycle": cyc["cycle"], "charge_time_s": cyc["charge_time_s"]})
        if status:
            cycles.append(cyc)
            break
        # ---- rest, discharge, rest
        for phase in ("REST_AFTER_CHARGE", "DISCHARGE", "REST_AFTER_DISCHARGE"):
            rs.stage = phase
            if phase == "DISCHARGE":
                rs.next_boundary_t_s = rs.t_abs_s   # update at the charge/discharge transition
            if phase.startswith("REST"):
                rest_end = rs.t_abs_s + rest_s
                rs.rest_remaining_s = rest_s
                while rs.rest_remaining_s > 1e-9:
                    cls, st = piece("REST", step_rest(dt, output_period_s), rest_end, None, None)
                    rs.rest_remaining_s = rest_end - rs.t_abs_s
                    if cls != "time":
                        status = f"failed:{phase}:{cls}"
                        break
                if status:
                    break
                continue
            d_ah0, end = rs.discharge_Ah, None
            while end is None:
                lim = np.inf
                if efc_stop is not None:
                    lim = rs.t_abs_s + (efc_stop * rs.q0_Ah - rs.discharge_Ah) / I_dis * 3600.0
                if next_diagnostic is not None:
                    lim = min(lim, rs.t_abs_s + (next_diagnostic * rs.q0_Ah - rs.discharge_Ah) / I_dis * 3600.0)
                cls, st = piece("DIS", step_dis(I_dis, dt, output_period_s), lim, SOC_LO, X_LO)
                if cls in ("soc", "soc_already"):
                    end = cls
                elif cls == "voltage":
                    end, status = "early_voltage_cutoff", "operating_condition_not_met:early_voltage_cutoff"
                elif cls == "time":
                    if efc_stop is not None and rs.discharge_Ah >= efc_stop * rs.q0_Ah - 1e-12:
                        end, status = "efc_target", "efc_target"
                elif cls == "zero_length" and efc_stop is not None:
                    end, status = "efc_target", "efc_target"
                else:
                    end, status = cls, f"failed:discharge:{cls}"
            track_values(last_step(state["sol"]))
            update_boundaries()
            cyc.update({"discharge_end": end, "discharge_Ah": rs.discharge_Ah - d_ah0,
                        "soc_discharge_end": soc_from_x(scalar(last_step(state["sol"]), X), state["b"])})
            cyc["soc_discharge_error"] = cyc["soc_discharge_end"] - SOC_LO
            if status:
                break
        cycles.append(cyc)
        if status:
            break
        rs.cycle += 1
        if max_cycles is not None and rs.cycle >= max_cycles:
            status = "max_cycles"
    rs.stage = "STOPPED"
    chk["esoh_calls"] = cell.esoh_calls
    return state["sol"], cycles, status, chk


def ledger(rs: RunState) -> dict:
    return asdict(rs)
