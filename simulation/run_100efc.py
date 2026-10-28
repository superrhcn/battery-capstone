"""Fixed-current, single-cell first-100-EFC runner. No ML or optimisation.

Outputs: per-cycle Parquet observations, JSON summaries, atomic trusted-local
checkpoint. Diagnostics run on copied states and never count towards EFC.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

# One solver thread per independent trajectory; set before importing numerical code.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import casadi
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pybamm

import battery as B


class BudgetExpired(RuntimeError):
    pass


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=B.jsonable,
                              allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def identity(args):
    files = [*sorted((B.ROOT / "configs").glob("*.json")),
             Path(B.__file__), Path(__file__)]
    versions = {n: importlib.metadata.version(n) for n in
                ("pybamm", "pybammsolvers", "casadi", "numpy", "scipy", "pyarrow")}
    return {"cell": args.cell, "current_A": args.current, "mesh_scale": args.mesh_scale,
            "efc_target": args.efc, "output_period_s": 1.0,
            "source_sha256": {str(p.relative_to(B.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files}, "versions": versions,
            "python": platform.python_version(),
            "tolerance": {"rtol": B.DEFAULT_RTOL, "atol": B.DEFAULT_ATOL}}


def save_checkpoint(path, state):
    tmp = path.with_name(path.name + ".tmp")
    with casadi.global_pickle_context():
        with tmp.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path):
    # Only load checkpoints created locally by this runner. Pickle is executable.
    with casadi.global_unpickle_context():
        with path.open("rb") as f:
            return pickle.load(f)


class Observations:
    """Write each solver piece immediately; no full trajectory in RAM."""
    def __init__(self, path):
        self.path = path
        self.tmp = path.with_name(path.name + ".partial")
        self.writer = None
        self.rows = 0

    def add(self, sol, st, kind, b, rs):
        t = np.asarray(st.t, dtype=float)
        q = np.asarray(st[B.VARS["discharge_capacity"]].entries, dtype=float).reshape(-1)
        x = np.asarray(st[B.X].entries, dtype=float).reshape(-1)
        dq = float(q[-1] - q[0])
        start_dis = rs.discharge_Ah - max(dq, 0)
        efc = (start_dis + np.maximum(q - q[0], 0)) / rs.q0_Ah
        count = len(t)
        tab = pa.table({
            "cell_id": np.full(count, rs.cell_id, dtype=np.int32),
            "trajectory_id": [rs.trajectory_id] * count,
            "cycle": np.full(count, rs.cycle + 1, dtype=np.int32),
            "piece": np.full(count, rs.pieces, dtype=np.int32),
            "phase": [kind] * count, "time_s": t,
            "current_A": np.asarray(st[B.VARS["current"]].entries, dtype=float).reshape(-1),
            "voltage_V": np.asarray(st[B.VARS["voltage"]].entries, dtype=float).reshape(-1),
            "temperature_C": np.full(count, B.MODEL_CFG["temperature_K"] - 273.15),
            "soc_fraction": B.soc_from_x(x, b), "efc": efc,
            "discharge_capacity_Ah": q,
        })
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp, tab.schema, compression="zstd")
        self.writer.write_table(tab)
        self.rows += count

    def close(self, commit=True):
        if self.writer:
            self.writer.close()
            self.writer = None
            if commit:
                self.tmp.replace(self.path)


def worker(args):
    out = Path(args.output_dir).resolve()
    cp_path = out / "checkpoint.pkl"
    sig = identity(args)
    if args.resume:
        if not cp_path.is_file():
            raise RuntimeError("Resume requested but checkpoint.pkl is missing")
        state = load_checkpoint(cp_path)
        if state["identity"] != sig:
            raise RuntimeError("Checkpoint config/source/dependency identity mismatch")
        if state.get("terminal"):
            print("Checkpoint is already terminal; nothing to resume", flush=True)
            return
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("Output directory is not empty; use a new directory or --resume")
        out.mkdir(parents=True, exist_ok=True)
        state = {"identity": sig, "cycles": [], "diagnostics": [], "files": [],
                 "elapsed_wall_s": 0.0, "terminal": False, "checks": [], "row_count": 0}
    elapsed_prior = state["elapsed_wall_s"]
    for previous in (out / "FAILED.json", out / "TIMEOUT.json"):
        if previous.is_file():
            elapsed_prior = max(elapsed_prior, json.loads(previous.read_text(encoding="utf-8")).get("elapsed_wall_s", 0))
    started = time.perf_counter()
    obs = None

    def budget():
        if elapsed_prior + time.perf_counter() - started > args.max_wall_s:
            raise BudgetExpired("Cumulative trajectory wall-time limit exceeded")

    specs = json.loads((B.ROOT / "configs/cells_pilot9.json").read_text(encoding="utf-8"))
    spec = next(s for s in specs["cells"] if s["id"] == args.cell)
    cell = B.Cell({"Negative particle radius [m]": spec["negative_particle_radius_m"],
                   "SEI solvent diffusivity [m2.s-1]": spec["sei_solvent_diffusivity_m2s"]},
                  mesh_scale=args.mesh_scale)
    if args.resume:
        rs = B.RunState(**state["ledger"])
        sol = state["solution"]
        # Roll back uncommitted output only inside this run's dedicated directory.
        committed = set(state["files"])
        for p in out.glob("cycle_*.parquet*"):
            if p.name not in committed:
                p.unlink()
    else:
        sol = cell.initial_state()
        rs = B.RunState(args.cell, f"cell{args.cell}_{args.current:g}A_mesh{args.mesh_scale:g}",
                        t_abs_s=float(sol.t[-1]))

    def diagnose(main, ledger, target):
        budget()
        before = B.y_end(main)
        book = copy.deepcopy(B.ledger(ledger))
        result = cell.reference_test(main.last_state.copy())
        change = float(np.max(np.abs(B.y_end(main) - before)))
        same = B.ledger(ledger) == book
        q = result["Q_cum_Ah"]
        err = abs(result["Q_trapz_Ah"] - q)
        if not result["ok"] or not np.isfinite(q) or q <= 0 or err / q > 1e-4:
            raise RuntimeError(f"Reference test failed at {target:g} EFC")
        if change != 0 or not same:
            raise RuntimeError("Cloned diagnostic changed main state or ledger")
        denom = ledger.q0_Ah if ledger.q0_Ah is not None else q
        state["diagnostics"].append({"efc": float(ledger.efc), "target_efc": target,
            "Qref_Ah": q, "SOH_percent": 100 * q / denom,
            "state_max_abs_change": change, "ledger_unchanged": same,
            "output_period": "solver/experiment default; config 1s is not applied by reference_test",
            "reference": result})
        save_json(out / "diagnostics.json", state["diagnostics"])
        print(f"diagnostic EFC={ledger.efc:.9f} SOH={100*q/denom:.7f}%", flush=True)
        budget()
        return q

    def checkpoint(terminal=False):
        state.update({"ledger": B.ledger(rs), "solution": sol.last_state,
                      "elapsed_wall_s": elapsed_prior + time.perf_counter() - started,
                      "terminal": terminal})
        save_checkpoint(cp_path, state)
        summary = {k: v for k, v in state.items() if k != "solution"}
        summary.update({"memory": B.memory_mb(), "backend": "local or cloud Python process",
                        "actual_cycles_completed": rs.cycle,
                        "all_completed_charges_under_60min": (not bool(rs.time_violations)) if state["cycles"] else None,
                        "model_builds": cell.builds, "esoh_calls": cell.esoh_calls,
                        "numerical_threads": {k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                        "limitations": ["100 EFC does not establish end-of-life RUL",
                            "RPT period remains experiment default",
                            "Main 1s output includes event endpoints and duplicate phase boundaries",
                            "No 0.1s transition trace in this runner"]})
        save_json(out / "summary.json", summary)

    try:
        if not args.resume:
            rs.q0_Ah = diagnose(sol, rs, 0.0)
            checkpoint()
        targets = sorted(set([5.0, *map(float, range(10, 101, 10)), args.efc]))
        targets = [x for x in targets if 0 < x <= args.efc]
        while rs.efc < args.efc - 1e-9:
            budget()
            cycle_id = rs.cycle + 1
            obs = Observations(out / f"cycle_{cycle_id:06d}.parquet")

            def on_piece(main, st, kind, b, ledger):
                budget()
                obs.add(main, st, kind, b, ledger)

            sol, cycles, status, checks = B.run_operation(
                cell, sol, rs, args.current, B.MODEL_CFG["currents_A"]["discharge"],
                B.MODEL_CFG["boundary_update_s"], max_cycles=rs.cycle + 1,
                efc_stop=args.efc, charge_limit_s=3600, output_period_s=1.0,
                diagnostic_efcs=targets, on_diagnostic=diagnose, on_piece=on_piece)
            obs.close()
            state["files"].append(obs.path.name)
            state["row_count"] += obs.rows
            obs = None
            state["cycles"].extend(cycles)
            state["checks"].append(checks)
            bad_soc = any(abs(c.get(k, 0)) > B.SOC_TOL
                for c in cycles for k, end in (("soc_charge_error", "charge_end"),
                                               ("soc_discharge_error", "discharge_end"))
                if c.get(end) in ("soc", "soc_already"))
            invalid_values = (any(v < 0 for v in checks["min_values"].values())
                              or not (0 < checks["porosity_range"][0] <= checks["porosity_range"][1] < 1))
            bad_join = (checks["time_join_max_error_s"] > 1e-6 or
                        any(j["max_rel"] > 1e-10 for j in checks["transition_jumps"]))
            if checks["nonfinite_pieces"] or bad_soc or invalid_values or bad_join or status not in ("max_cycles", "efc_target"):
                raise RuntimeError(f"Operation failed checks: status={status}, SOC={bad_soc}")
            state["status"] = status
            checkpoint(terminal=(status == "efc_target"))
            print(f"cycle={rs.cycle} EFC={rs.efc:.7f} status={status} memory={B.memory_mb()}", flush=True)
            if status == "efc_target":
                (out / "DONE").write_text(f"EFC target {args.efc:g} reached\n", encoding="utf-8")
                break
            if args.smoke_cycle:
                print("Smoke complete: one full operating cycle; checkpoint remains resumable", flush=True)
                break
    except Exception:
        if obs:
            obs.close(commit=False)
        save_json(out / "FAILED.json", {"error": traceback.format_exc(), "ledger": B.ledger(rs),
                    "elapsed_wall_s": elapsed_prior + time.perf_counter() - started,
                    "memory": B.memory_mb()})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=int, default=4, choices=range(1, 10))
    p.add_argument("--current", type=float, required=True, choices=[3.0, 3.1, 3.2, 3.3])
    p.add_argument("--mesh-scale", type=float, default=4.0, choices=[1, 2, 4, 8])
    p.add_argument("--efc", type=float, default=100.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-cycle", action="store_true")
    p.add_argument("--max-wall-s", type=float, default=43200)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (0 < args.efc <= 100) or not (0 < args.max_wall_s <= 43200):
        p.error("0 < efc <= 100 and 0 < max-wall-s <= 43200 required")
    if args._worker:
        worker(args)
        return
    # External watchdog also catches a single solver call that never returns.
    prior = 0.0
    summary = Path(args.output_dir) / "summary.json"
    if args.resume:
        for previous in (summary, Path(args.output_dir) / "FAILED.json", Path(args.output_dir) / "TIMEOUT.json"):
            if previous.is_file():
                prior = max(prior, json.loads(previous.read_text(encoding="utf-8")).get("elapsed_wall_s", 0))
    remaining = args.max_wall_s - prior
    if remaining <= 0:
        raise SystemExit("Cumulative wall budget already exhausted")
    out = Path(args.output_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent / (out.name + ".runner.lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit(f"Run lock exists: {lock}; another process or stale lock requires inspection")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    proc = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--_worker"])
    try:
        code = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / "TIMEOUT.json", {"max_wall_s": args.max_wall_s,
                                       "message": "Worker killed by wall-time watchdog",
                                       "elapsed_wall_s": args.max_wall_s})
        raise SystemExit(124)
    finally:
        lock.unlink(missing_ok=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
