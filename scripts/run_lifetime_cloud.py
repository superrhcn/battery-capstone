"""Supervise one lifetime trajectory independently of the chat/SSM session.

Run via launch_lifetime_service.sh. No AWS API, SMTP password, or instance SNS
permission is used. EventBridge/SNS observes EC2 stopped outside this process.
Default mode actually calculates and shuts down EC2; --plan is read-only.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

GIB = 1024 ** 3


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def runner_command(args):
    cmd = [str(Path(args.python).resolve()),
        str(Path(args.repo_root).resolve() / "simulation/run_lifetime.py"),
        "--cell", str(args.cell), "--current", str(args.current),
        "--mesh-scale", str(args.mesh_scale), "--efc", str(args.efc),
        "--max-wall-s", str(args.max_wall_s),
        "--rpt-every-efc", str(args.rpt_every_efc),
        "--output-dir", str(Path(args.run_root).resolve() / "trajectory")]
    if args.resume:
        cmd.append("--resume")
    else:
        cmd.extend(["--import-100efc-dir", str(Path(args.import_100efc_dir).resolve())])
    return cmd


def hard_stop_command(args):
    return ["systemd-run", "--unit=" + args.guard_unit,
        "--on-active=" + str(int(args.max_wall_s + args.hard_stop_extra_s)) + "s",
        "--timer-property=AccuracySec=1s", "/usr/sbin/shutdown", "-h", "now"]


def memory_available_fraction():
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
        fields = {line.split(":", 1)[0]: int(line.split()[1]) for line in rows}
        return fields["MemAvailable"] / fields["MemTotal"]
    except (OSError, KeyError, ValueError):
        return None


def process_group_alive(pgid):
    """Check live Linux group members, ignoring already-dead zombies."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def terminate_tree(proc, grace_s=75, kill_wait_s=15):
    """Wait for the whole group, not just the runner supervisor parent PID."""
    result = {"term_sent": False, "kill_sent": False, "group_empty": False}
    if process_group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            result["term_sent"] = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_s
        while process_group_alive(proc.pid) and time.monotonic() < deadline:
            proc.poll()  # Reap the parent if it exits before its worker.
            time.sleep(.1)
        if process_group_alive(proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                result["kill_sent"] = True
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + kill_wait_s
            while process_group_alive(proc.pid) and time.monotonic() < deadline:
                proc.poll()
                time.sleep(.1)
    result["group_empty"] = not process_group_alive(proc.pid)
    if proc.poll() is not None:
        proc.wait(timeout=1)
    return result


def compact_outcome(outcome, summary, root):
    """Keep history on EBS; console/control JSON stay bounded at long horizons."""
    summary = summary or {}
    source = outcome or summary
    ledger = source.get("ledger") or summary.get("ledger") or {}
    diagnostics = source.get("diagnostics") or summary.get("diagnostics") or []
    last = diagnostics[-1] if diagnostics else {}
    eol = source.get("eol") or {}
    eol_keys = ("threshold_percent", "observed_at_EFC", "interval_lower_EFC",
                "interval_upper_EFC", "exact_crossing_known", "rul_from_100_observation_EFC")
    return {"status": source.get("status"),
        "last_valid_EFC": source.get("last_valid_EFC", ledger.get("efc")),
        "last_SOH_percent": last.get("SOH_percent"),
        "eol": {key: eol[key] for key in eol_keys if key in eol} or None,
        "elapsed_wall_s": source.get("elapsed_wall_s", summary.get("elapsed_wall_s")),
        "checkpoint_path": str(root / "trajectory/checkpoint.pkl"),
        "outcome_path": str(root / "trajectory/outcome.json"),
        "summary_path": str(root / "trajectory/summary.json")}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(args):
    command = runner_command(args)
    guard_command = hard_stop_command(args)
    if args.plan:
        print(json.dumps({"runner_command": command,
            "hard_stop_command": guard_command,
            "run_root": str(Path(args.run_root).resolve()),
            "trajectory_wall_limit_s": args.max_wall_s,
            "server_timer_s": args.max_wall_s + args.hard_stop_extra_s,
            "note": "Read-only plan. Does not install a unit, run the solver, or shut down."}, indent=2))
        return 0
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("Cloud supervisor requires Linux root; use --plan elsewhere")
    repo = Path(args.repo_root).resolve()
    root = Path(args.run_root).resolve()
    if not (repo / "simulation/run_lifetime.py").is_file():
        raise FileNotFoundError("Reviewed run_lifetime.py is missing")
    if not Path(args.python).is_file():
        raise FileNotFoundError("Selected Python executable is missing")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "cloud_status.json"
    if status_path.exists() and not args.resume:
        raise RuntimeError("Existing cloud run root; use a new root or explicit --resume")
    # Register first; if later setup or Python fails, server stop remains independent.
    subprocess.run(guard_command, check=True, timeout=30)
    log_path = root / "cloud_supervisor.log"
    proc = None
    reason = None
    termination = None
    exit_code = 1
    started = time.monotonic()
    state = {"status": "starting", "started_utc": utc_now(),
        "command": command, "server_guard": guard_command,
        "trajectory_wall_limit_s": args.max_wall_s,
        "server_hard_stop_s": args.max_wall_s + args.hard_stop_extra_s,
        "chat_required": False,
        "notification": "EC2 stopped/terminated EventBridge event; not experiment success proof"}

    def emit(value):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)
        try:
            with open("/dev/console", "a", encoding="utf-8") as console:
                console.write(text + "\n")
        except OSError:
            pass

    try:
        if shutil.disk_usage(root).free < args.minimum_free_gib * GIB:
            raise RuntimeError("Insufficient free disk before calculation")
        save_json(status_path, state)
        child_log_path = root / "runner.log"
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with child_log_path.open("ab", buffering=0) as child_log:
            proc = subprocess.Popen(command, cwd=repo, env=env,
                stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running", pid=proc.pid)
            save_json(status_path, state)
            emit({"stage": "lifetime_started", "pid": proc.pid, "utc": utc_now()})
            next_progress = 0.0
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                free = shutil.disk_usage(root).free
                available = memory_available_fraction()
                if free < args.minimum_free_gib * GIB:
                    reason = "disk_guard"
                elif available is not None and available < args.minimum_available_ram_fraction:
                    reason = "memory_guard"
                # Runner also enforces the cumulative trajectory budget. This outer
                # cap catches a wedged runner/supervisor with additional flush time.
                elif elapsed > args.max_wall_s + 120:
                    reason = "supervisor_wall_guard"
                if reason:
                    emit({"stage": "guard_stop", "reason": reason,
                          "elapsed_s": elapsed, "free_bytes": free,
                          "available_ram_fraction": available})
                    termination = terminate_tree(proc)
                    break
                if elapsed >= next_progress:
                    summary = read_json(root / "trajectory/summary.json") or {}
                    ledger = summary.get("ledger") or {}
                    emit({"stage": "lifetime_progress", "elapsed_s": round(elapsed, 1),
                        "efc": ledger.get("efc"), "runner_status": summary.get("status"),
                        "free_GiB": round(free / GIB, 3),
                        "available_ram_fraction": available})
                    next_progress = elapsed + 300
                time.sleep(5)
            exit_code = proc.wait(timeout=15)
        outcome = read_json(root / "trajectory/outcome.json")
        compact = compact_outcome(outcome, read_json(root / "trajectory/summary.json"), root)
        state.update(status=reason or ((outcome or {}).get("status")) or
                     ("runner_exit_zero_without_outcome" if exit_code == 0 else "failed"),
                     runner_exit_code=exit_code, outcome=compact)
    except BaseException as exc:
        state.update(status="supervisor_failed", error=str(exc), traceback=traceback.format_exc())
        if proc is not None:
            termination = terminate_tree(proc)
        exit_code = 1
    finally:
        if proc is not None and (termination is None or not termination.get("group_empty")):
            termination = terminate_tree(proc)
        state["termination"] = termination
        state["outcome"] = compact_outcome(read_json(root / "trajectory/outcome.json"), read_json(root / "trajectory/summary.json"), root)
        if termination and (termination["kill_sent"] or not termination["group_empty"]):
            state["forced_termination"] = True
            state["final_state_note"] = "Last committed checkpoint is authoritative; final partial state is not validated."
        state.update(finished_utc=utc_now(), supervisor_wall_s=time.monotonic() - started,
            shutdown_requested=True,
            result_root=str(root),
            result_note="Raw result files remain on EBS. Final partial cycle may require runner rollback on resume.")
        try:
            save_json(status_path, state)
            emit({"stage": "lifetime_final", "status": state["status"],
                "runner_exit_code": state.get("runner_exit_code"),
                "supervisor_wall_s": state["supervisor_wall_s"],
                "termination": termination, "forced_termination": state.get("forced_termination", False),
                **{key:value for key,value in state["outcome"].items() if key != "status"}})
        finally:
            # Shutdown is not conditional on notification delivery, result success,
            # network connectivity, or the current chat remaining open.
            try:
                subprocess.run(["sync"], timeout=30, check=False)
            finally:
                subprocess.run(["/usr/sbin/shutdown", "-h", "now"], timeout=30, check=False)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--import-100efc-dir")
    source.add_argument("--resume", action="store_true")
    parser.add_argument("--cell", type=int, default=4)
    parser.add_argument("--current", type=float, default=3.2)
    parser.add_argument("--mesh-scale", type=int, default=4)
    parser.add_argument("--efc", type=float, default=3000)
    parser.add_argument("--max-wall-s", type=int, default=72000)
    parser.add_argument("--rpt-every-efc", type=float, default=10)
    parser.add_argument("--hard-stop-extra-s", type=int, default=900)
    parser.add_argument("--guard-unit", default="capstone-lifetime-hard-stop")
    parser.add_argument("--minimum-free-gib", type=float, default=2)
    parser.add_argument("--minimum-available-ram-fraction", type=float, default=.08)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not (0 < args.max_wall_s <= 72000) or not (180 <= args.hard_stop_extra_s <= 900):
        parser.error("wall budget <=72000s; hard stop grace 180..900s required")
    if not (0 < args.efc <= 3000) or args.rpt_every_efc <= 0:
        parser.error("max EFC <=3000 and positive diagnostic interval required")
    if not (1 <= args.minimum_free_gib <= 10) or not (0 < args.minimum_available_ram_fraction < .5):
        parser.error("invalid resource guard")
    if not all(c.isalnum() or c in "-_" for c in args.guard_unit):
        parser.error("invalid guard unit")
    raise SystemExit(run(args))

if __name__ == "__main__":
    main()
