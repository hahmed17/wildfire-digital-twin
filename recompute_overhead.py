"""
Recompute agent overhead and the guardrail override rate from a finished run's
decisions.csv, separating the two things the old behaviour_summary conflated.

    python recompute_overhead.py                 # every run under the arm output root
    python recompute_overhead.py <run_dir> ...   # named runs only

The forecast and synthesize stages log a record for provenance but take no
decision, and the forecast's record carries its entire runtime -- the whole
ensemble, all of FARSITE -- in the agent_latency_s field. Summing that field
across every row therefore reports wall-clock time as agent overhead. The split
below is on whether a row actually chose something.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import orchestrator  # noqa: F401  -- puts src/ on sys.path
import config


def summarise(run_dir: Path) -> dict | None:
    path = run_dir / "decisions.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    def lat(r):
        try:
            return float(r.get("agent_latency_s") or 0.0)
        except ValueError:
            return 0.0

    def truthy(v):
        return str(v).strip().lower() in ("1", "true", "yes")

    decided = [r for r in rows if (r.get("action") or "").strip() not in ("", "None")]
    steps = [r for r in rows if r not in decided]
    overrides = sum(truthy(r.get("overridden_by_guardrail")) for r in decided)

    return {
        "run": run_dir.name,
        "records": len(rows),
        "decisions": len(decided),
        "agent_s_total": sum(lat(r) for r in decided),
        "agent_s_max": max((lat(r) for r in decided), default=0.0),
        "step_exec_s": sum(lat(r) for r in steps),
        "overrides": overrides,
        "override_rate": overrides / len(decided) if decided else None,
    }


def main(argv: list[str]) -> int:
    if argv:
        dirs = [Path(a) for a in argv]
    else:
        root = config.OUTPUT_DIR / "Border_2_Orchestrator"
        dirs = sorted(d for d in root.glob("*") if (d / "decisions.csv").exists())

    out = [s for d in dirs if (s := summarise(d))]
    if not out:
        print("No runs with a decisions.csv found.")
        return 1

    hdr = (f"{'run':34s} {'dec':>4s} {'agent s':>9s} {'max s':>8s} "
           f"{'steps s':>9s} {'overhead':>9s} {'ovr':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for s in out:
        total = s["agent_s_total"] + s["step_exec_s"]
        frac = s["agent_s_total"] / total if total else 0.0
        print(f"{s['run']:34s} {s['decisions']:4d} {s['agent_s_total']:9.3f} "
              f"{s['agent_s_max']:8.3f} {s['step_exec_s']:9.1f} "
              f"{frac:8.2%} {s['overrides']:5d}")

    print("\nagent s  = time inside the policies, summed over decisions")
    print("steps s  = execution time of the stages that take no decision")
    print("overhead = agent time as a fraction of the two combined")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
