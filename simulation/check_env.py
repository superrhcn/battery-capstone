"""
simulation/check_env.py

Step 1 of docs/modeling_decisions.md section 14:
  - the six pre-existing packages kept their versions
  - pyarrow / scikit-learn / torch import and report the pinned versions
  - a short SPMe run with the project options builds and solves

Run:     python simulation/check_env.py
Output:  outputs/check_env.json
"""

import json
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

EXPECTED = {
    "pybamm": "26.8.0.0",
    "pybammsolvers": "0.9.1",
    "casadi": "3.7.2",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "pandas": "3.0.5",
    "pyarrow": "26.0.0",
    "scikit-learn": "1.9.1",
    "torch": "2.14.1+cpu",
}

report = {"python": sys.version, "versions": {}, "imports": {}, "spme": {}}

for dist, want in EXPECTED.items():
    try:
        got = version(dist)
    except Exception as e:
        got = f"missing: {e}"
    report["versions"][dist] = {"expected": want, "installed": got, "ok": got == want}

for mod in ("pyarrow", "sklearn", "torch", "pybamm", "numpy", "scipy", "pandas"):
    try:
        m = __import__(mod)
        report["imports"][mod] = {"ok": True, "__version__": getattr(m, "__version__", None)}
    except Exception:
        report["imports"][mod] = {"ok": False, "traceback": traceback.format_exc()}

try:
    import pybamm

    cfg = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    model = pybamm.lithium_ion.SPMe(options=cfg["options"])
    p = pybamm.ParameterValues(cfg["parameter_set"])
    p["Ambient temperature [K]"] = cfg["temperature_K"]
    p["Initial temperature [K]"] = cfg["temperature_K"]
    exp = pybamm.Experiment(["Charge at 1.5 A for 10 minutes"])
    sim = pybamm.Simulation(model, parameter_values=p, experiment=exp)
    t0 = time.perf_counter()
    sol = sim.solve(initial_soc=0.2)
    report["spme"] = {
        "ok": True,
        "wall_s": time.perf_counter() - t0,
        "termination": str(sol.termination),
        "t_end_s": float(sol["Time [s]"].entries[-1]),
        "V_end": float(sol["Voltage [V]"].entries[-1]),
        "applied_options": {k: str(model.options[k]) for k in cfg["options"]},
    }
except Exception:
    report["spme"] = {"ok": False, "traceback": traceback.format_exc()}

report["all_ok"] = (
    all(v["ok"] for v in report["versions"].values())
    and all(v["ok"] for v in report["imports"].values())
    and report["spme"].get("ok", False)
)
(OUT / "check_env.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
