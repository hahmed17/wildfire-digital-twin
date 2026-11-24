"""
Pre-flight for the Fixed vs Rule arms. Run from the repo root before run_arms.py:

    python arms_preflight.py

Twelve arms share one machine, one quota, and one copy of src/, so a problem that
costs minutes in a single run costs a day here. Each check below is something that
has already gone wrong at least once in this project.
"""
from __future__ import annotations

import importlib
import inspect
import os
import shutil
import sys
from pathlib import Path

# Importing the package puts the repo's src/ on sys.path (see orchestrator/
# __init__.py), which is what makes the plain `import config` below work from
# any directory. The order matters.
import orchestrator  # noqa: F401
import config

# This script deliberately does NOT chdir. An earlier version resolved the repo
# root and chdir'd there, which made every path check pass from any directory --
# and thereby hid the bug it existed to catch: replay.py held its perimeter and
# hotspot paths as strings relative to the working directory, so a run launched
# from a subdirectory died on the first read while the pre-flight reported all
# clear. The pre-flight has to see what the run will see, so paths are checked
# exactly as the modules resolve them.

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


# --- 1. the FARSITE disk leak ---------------------------------------------
# Without the fix each sub-step leaves ~13 files behind. One arm at N=300 leaves
# roughly 600,000; twelve arms would leave millions. This is the check that
# matters most, because the failure mode is a quota error hours in.
try:
    import farsite
    src = inspect.getsource(farsite)
    per_substep = src.count("cleanup_farsite_outputs(farsite.id")
    logs_cleaned = 'farsite_logs").glob' in src or "farsite_logs') / " in src \
        or 'farsite_logs"' in src.split("def cleanup_farsite_outputs")[1][:800]
    check("FARSITE cleans each sub-step (disk leak fixed)",
          per_substep >= 2,
          "sub-step files accumulate for the whole run; a single arm leaves ~600k files")
    check("FARSITE cleanup also clears farsite_logs/",
          logs_cleaned,
          "logs live in a subdirectory the {run_id}_* glob never matched")
except Exception as exc:                                        # noqa: BLE001
    check("FARSITE module imports", False, f"{type(exc).__name__}: {exc}")

# --- 2. the orchestrator modules -------------------------------------------
for mod in ("orchestrator.replay", "orchestrator.metrics", "orchestrator.run_arms",
            "orchestrator.steps", "orchestrator.adapter"):
    try:
        importlib.import_module(mod)
        check(f"{mod} imports", True)
    except Exception as exc:                                    # noqa: BLE001
        check(f"{mod} imports", False, f"{type(exc).__name__}: {exc}")

# --- 3. steps.py is the version with the forecast implemented --------------
try:
    from orchestrator import steps
    check("steps.py has seed_state / make_forecast / apply_assimilation",
          all(hasattr(steps, n) for n in
              ("seed_state", "make_forecast", "apply_assimilation")),
          "an older steps.py is installed; forecast still raises NotImplementedError")
except Exception as exc:                                        # noqa: BLE001
    check("steps.py loads", False, str(exc))

# --- 4. the sub-step rule matches the notebook -----------------------------
# If the baseline and the Fixed arm disagree here they differ in FARSITE
# sub-step length as well as policy, which confounds the comparison.
try:
    from orchestrator import steps
    check("sub-step rule matches the notebook (15 min above 25 mph, else 30)",
          steps.HIGH_WIND_MPH == 25.0
          and steps.FARSITE_DT_HIGH_WIND == 15
          and steps.FARSITE_DT_NORMAL == 30,
          f"orchestrator uses {steps.FARSITE_DT_HIGH_WIND}/{steps.FARSITE_DT_NORMAL} min "
          f"above {steps.HIGH_WIND_MPH} mph")
except Exception as exc:                                        # noqa: BLE001
    check("sub-step rule readable", False, str(exc))

# --- 5. inputs present -----------------------------------------------------
try:
    from orchestrator import adapter, replay
    lcp = Path(str(adapter.LCP_PATH))
    check("landscape file present", lcp.exists(), f"missing: {lcp}")
    check("perimeter file present", Path(replay.PERIMETER_FILE).exists(),
          f"missing: {replay.PERIMETER_FILE}")
    check("FIRMS hotspots present", Path(replay.HOTSPOT_FILE).exists(),
          f"missing: {replay.HOTSPOT_FILE}; agrees_with_firms will stay None")
except Exception as exc:                                        # noqa: BLE001
    check("input paths readable", False, str(exc))

# --- 5b. path constants are absolute --------------------------------------
# The failure mode this catches: a module-level path stored relative to the
# working directory. It resolves once, at import, against wherever the process
# happened to start, so `python -m orchestrator.run_arms` works from the repo
# root and dies instantly from anywhere else.
try:
    from orchestrator import replay
    relative = [n for n in ("PERIMETER_FILE", "HOTSPOT_FILE", "OUTPUT_ROOT")
                if not Path(str(getattr(replay, n))).is_absolute()]
    check("replay path constants are absolute (run from any directory)",
          not relative,
          f"relative: {', '.join(relative)}; these resolve against the working "
          "directory, so the arms only run from the repo root")
except Exception as exc:                                        # noqa: BLE001
    check("replay path constants readable", False, str(exc))

# --- 6. tmp is clear and the disk has room ---------------------------------
tmp = Path(config.SRC_DIR) / "tmp"
n_tmp = sum(1 for _ in tmp.rglob("*")) if tmp.exists() else 0
check("src/tmp is clear before starting", n_tmp < 100,
      f"{n_tmp:,} files left over; clear with: find src/tmp -mindepth 1 -delete")

# With the sub-step cleanup in place the arms need very little: each N=300 arm
# writes about 1 MB of ensembles and logs, each N=48 arm about 150 KB, and peak
# scratch is twelve workers holding one sub-step each. The 5 GB figure used
# before was sized for the leak, not for the fixed code.
usage = shutil.disk_usage(".")
free_gb = usage.free / 1e9
check("at least 2 GB free", free_gb >= 2,
      f"only {free_gb:.1f} GB free; the arms need roughly 100 MB with the "
      "FARSITE cleanup in place, but leave headroom for the simulator's scratch")

# --- 7. the baseline to compare against ------------------------------------
base = config.OUTPUT_DIR / "Border_2_Real"
steps_csv = base / "steps.csv"
n_steps = 0
if steps_csv.exists():
    n_steps = max(0, sum(1 for _ in open(steps_csv)) - 1)
check("baseline present with all three steps", n_steps >= 3,
      f"steps.csv has {n_steps} step rows; the N=300 arms are compared against it")

# --- 8. arms will not write into the baseline ------------------------------
try:
    from orchestrator import replay
    out_root = Path(str(replay.OUTPUT_ROOT))
    check("arms write outside the baseline directory",
          base.resolve() not in out_root.resolve().parents
          and out_root.resolve() != base.resolve(),
          f"arms would write into {out_root}")
except Exception as exc:                                        # noqa: BLE001
    check("arm output root readable", False, str(exc))

# --- report ----------------------------------------------------------------
width = max(len(n) for n, _, _ in checks)
print("Arm pre-flight")
print("=" * (width + 10))
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        -> {detail}")
print("=" * (width + 10))

failed = [n for n, ok, _ in checks if not ok]
if failed:
    print(f"\n{len(failed)} check(s) failed. Fix before running run_arms.py --- "
          "twelve arms multiply any problem by twelve.")
    sys.exit(1)

print(f"\nAll {len(checks)} checks passed.")
print(f"Free disk: {free_gb:.1f} GB   src/tmp: {n_tmp} files")
print("\nNext:  python -m orchestrator.run_arms --smoke     (~0.5 h)")
print("then:  python -m orchestrator.run_arms              (resumable)")