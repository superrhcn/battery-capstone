"""
simulation/verify_r10_mesh.py

R10 mesh convergence (user request). For a given cell id, run mesh scale x1, x2, x4:
  - own Q0 from a cloned reference test of that mesh's own initial state,
  - degrade to 5 EFC with the 1.5 A / 5 A grading schedule on that same mesh,
  - R10 on a clone of the 5 EFC state (0.5 A -> SOC 50 %, 30 min rest, 5 A 10 s).
No state vector is carried across meshes; each mesh runs from its own initial state.
Solver tolerance is held fixed (default) across the three meshes so only the mesh changes.

Each mesh runs in its own subprocess so peak RAM and crashes are per mesh.
The x4 run has a 30-minute wall limit (verification-only cap, enforced by the parent
as a process timeout). On failure or timeout we record config, last EFC reached,
error, wall time and peak RAM, then stop. No automatic mesh fallback.

Run (one mesh, child):   python simulation/verify_r10_mesh.py --cell 4 --mesh 4
Run (driver, one cell):  python simulation/verify_r10_mesh.py --cell 4
Output: outputs/verify_r10_mesh/cell_<id>_mesh_<scale>.json  and  cell_<id>.json (merged)
"""

from __future__ import annotations

import argparse
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
from run_pilot9 import operation, save  # noqa: E402

CFG = json.loads((B.ROOT / "configs" / "cells_pilot9.json").read_text(encoding="utf-8"))
OUT = Path(os.environ.get("R10_MESH_OUT", B.ROOT / "outputs" / "verify_r10_mesh"))
OUT.mkdir(parents=True, exist_ok=True)
VERIFY = B.ROOT / "outputs" / "verify_criteria" / "report.json"
X4_WALL_LIMIT_S = 1800.0  # 30 min, verification-only cap for the x4 pre-check


def run_one(cell_id: int, mesh_scale: float, tolerance: dict):
    """One mesh for one cell, in its own process. Writes a per-mesh json and returns."""
    t0 = time.perf_counter()
    spec = next(x for x in CFG["cells"] if x["id"] == cell_id)
    ov = {"Negative particle radius [m]": spec["negative_particle_radius_m"],
          "SEI solvent diffusivity [m2.s-1]": spec["sei_solvent_diffusivity_m2s"]}
    g = CFG["grading_run"]
    res = {"cell": spec, "overrides": ov, "mesh_scale": mesh_scale, "tolerance": tolerance,
           "pybamm_version": pybamm.__version__, "wall_limit_s": X4_WALL_LIMIT_S if mesh_scale == 4 else None,
           "failures": [], "last_efc": 0.0}
    path = OUT / f"cell_{cell_id}_mesh_{int(mesh_scale)}.json"
    try:
        c = B.Cell(ov, mesh_scale=mesh_scale, **tolerance)
        res["var_pts"] = c.var_pts
        S0 = c.initial_state()
        ref0 = c.reference_test(S0.last_state.copy())
        q0 = ref0["Q_trapz_Ah"]
        r10_0 = c.r10_test(S0)
        res["Q0_Ah"] = q0
        res["R10_initial_ohm"] = r10_0["R10_ohm"]
        res["R10_initial_ok"] = r10_0["ok"]
        # accumulate real compute time spent on diagnostics (separate from main-trajectory physics)
        diag_wall = ref0["wall_s"] + r10_0["wall_s"]
        save(path, res)

        sol5, rs5, info5 = operation(c, S0, cell_id, f"grade_1.5A_mesh{int(mesh_scale)}", q0,
                                     g["charge_current_A"], g["discharge_current_A"],
                                     float(g["update_interval_s"]), efc_stop=g["efc_stop"])
        res["last_efc"] = B.ledger(rs5).get("efc")
        res["grading_status"] = info5["status"]
        res["grading_wall_s"] = info5["wall_s"]
        res["grading_max_abs_soc_error"] = info5["max_abs_soc_error"]
        res["grading_checks"] = info5["checks"]
        res["end_state"] = info5["end_state"]
        if info5["status"] != "efc_target":
            res["failures"].append(f"grading:{info5['status']}")
        else:
            import copy as _copy
            led_b, y_b = _copy.deepcopy(B.ledger(rs5)), B.y_end(sol5)
            ref5 = c.reference_test(sol5.last_state.copy())
            r10 = c.r10_test(sol5)
            res["Q5_Ah"] = ref5["Q_trapz_Ah"]
            res["SOH5_percent"] = 100 * ref5["Q_trapz_Ah"] / q0
            res["R10_5EFC_ohm"] = r10["R10_ohm"]
            res["R10_5EFC_soc_after_positioning"] = r10["soc_after_positioning"]
            res["R10_5EFC_ok"] = r10["ok"]
            # diagnostics must not touch the main state or the time/EFC ledger
            res["main_state_max_abs_change"] = float(np.max(np.abs(B.y_end(sol5) - y_b)))
            res["main_state_unchanged"] = res["main_state_max_abs_change"] == 0.0
            res["ledger_unchanged"] = B.ledger(rs5) == led_b
            res["ledger_after_diag"] = B.ledger(rs5)
            diag_wall += ref5["wall_s"] + r10["wall_s"]
            if not (ref5["ok"] and r10["ok"]):
                res["failures"].append("diagnostic_at_5EFC_termination")
            if not (res["main_state_unchanged"] and res["ledger_unchanged"]):
                res["failures"].append("diagnostic_altered_main_state_or_ledger")
        # diagnostics do not change the main physical time / EFC, but their compute
        # time is counted separately so it can be added to the 12-hour compute budget
        res["diagnostic_wall_total_s"] = diag_wall
    except Exception:
        res["failures"].append("exception")
        res["traceback"] = traceback.format_exc()
    res["wall_s"] = time.perf_counter() - t0
    res["memory"] = B.memory_mb()
    save(path, res)


def drive(cell_id: int, tolerance: dict):
    results = {}
    for mesh in (1, 2, 4):
        path = OUT / f"cell_{cell_id}_mesh_{mesh}.json"
        if path.exists():
            path.unlink()
        timeout = X4_WALL_LIMIT_S if mesh == 4 else None
        t0 = time.perf_counter()
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--cell", str(cell_id), "--mesh", str(mesh)],
                           check=False, timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        wall = time.perf_counter() - t0
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = {"mesh_scale": mesh, "failures": ["no_output"], "last_efc": None}
        if timed_out:
            r.setdefault("failures", []).append(f"wall_limit_{int(X4_WALL_LIMIT_S)}s_exceeded")
            r["driver_wall_s"] = wall
        results[f"mesh_{mesh}"] = r
        save(OUT / f"cell_{cell_id}_mesh_{mesh}.json", r)
        # stop the whole sequence if this mesh did not reach 5 EFC
        if timed_out or r.get("failures"):
            results["stopped_after_mesh"] = mesh
            break
    save(OUT / f"cell_{cell_id}.json", results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--mesh", type=float)
    args = ap.parse_args()
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    if not v.get("all_pass"):
        raise SystemExit("verify_criteria did not pass")
    tolerance = v["selected_tolerance"]
    if args.mesh is not None:
        run_one(args.cell, args.mesh, tolerance)
    else:
        drive(args.cell, tolerance)


if __name__ == "__main__":
    main()
