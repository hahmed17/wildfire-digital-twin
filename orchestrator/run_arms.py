"""
Run the Fixed vs Rule experiment arms for Border 2.

    # from the repo root, after the baseline finishes (it holds all 12 workers)
    python -m orchestrator.run_arms --smoke          # ~0.5 h, proves the FARSITE path
    python -m orchestrator.run_arms                  # the 10 arms, ~41 h
    python -m orchestrator.run_arms --table          # just reprint the results table

Design
------
Two groups, because decisions and accuracy need different ensemble sizes:

  decisions  N=48, all four scenarios, both policies   8 runs, ~2.0 h each
             The agents' choices barely depend on N -- containment, growth, obs age
             and wind shift never touch the ensemble, and normalized_innovation is
             stable across N -- so a small ensemble buys the full decision table
             cheaply.

  accuracy   N=300, clean only, both policies          2 runs, ~12.6 h each
             IoU against the true perimeter does depend on N, and N=300 is what the
             baseline used, so these are the runs that compare to it.

The cheap group runs first: it surfaces any problem in the FARSITE path in 2 hours
rather than 12, and leaves a useful result on disk if the long runs are interrupted.

N is a multiple of the worker count on purpose. Wall time per step is
ceil(N/workers) batches, so N=48 and N=60 both cost 5 batches at 12 workers --
48 wastes nothing, 50 leaves a batch almost idle.

Why N and resolution are pinned
-------------------------------
RulePolicy would otherwise size the ensemble itself from the operator deadline.
Both are overridden here so the ONLY difference between the Fixed and Rule arms is
the decisions the agents make; otherwise an accuracy gap could just as well be the
two arms having run different ensemble sizes. Deadline-driven plan sizing is worth
measuring, but as its own experiment (sweep deadline_min, hold the policy fixed).

Resumable: an arm whose directory already holds metrics.json is skipped, so a
failure part-way costs only the arm that failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from orchestrator import metrics, replay

# Matches border2_experiment.ipynb's RANDOM_SEED, so the arms draw their member
# winds from the same stream the baseline did.
SEED = 1234

WORKERS = 12
RESOLUTION_M = 150        # the baseline's DIST_RES / PERIM_RES
DEADLINE_MIN = 60.0

SCENARIOS = ["clean", "shifted_obs", "mislabeled_obs", "stale_obs", "weather_failure"]
FAULT_SCENARIOS = [s for s in SCENARIOS if s != "clean"]

# (label, policies, scenario, ensemble_size)
# `clean` is included in the N=48 group as a within-group control: it is what shows
# RulePolicy does NOT reject good observations. Without it the decision table only
# demonstrates true positives, and "does it cry wolf?" is the first thing a reader
# will ask. Two extra runs, ~4 h. Drop them with --no-clean-control if the budget
# is tight; the N=300 clean arms cover the same ground at a different N.
DECISION_ARMS = [(f"{p}/{s}", p, s, 48)
                 for s in SCENARIOS for p in ("fixed", "rule")]
DECISION_ARMS_NO_CONTROL = [a for a in DECISION_ARMS if a[2] != "clean"]
ACCURACY_ARMS = [(f"{p}/clean@300", p, "clean", 300) for p in ("fixed", "rule")]


def _already_done(out_root: Path, policies: str, scenario: str,
                  ensemble_size: int) -> Path | None:
    """An arm's directory, if a completed run for it already exists."""
    for d in sorted(out_root.glob(f"{policies}_{scenario}_*")):
        mpath = d / "metrics.json"
        cpath = d / "config.json"
        if not (mpath.exists() and cpath.exists()):
            continue
        try:
            cfg = json.loads(cpath.read_text())
        except Exception:
            continue
        if cfg.get("ensemble_size_override") == ensemble_size:
            return d
    return None


def run_arm(label: str, policies: str, scenario: str, ensemble_size: int,
            out_root: Path, dry_run: bool = False) -> dict | None:
    done = _already_done(out_root, policies, scenario, ensemble_size)
    if done is not None:
        print(f"  SKIP  {label:24s} already complete -> {done.name}", flush=True)
        return metrics.load_metrics(done / "metrics.json")

    if dry_run:
        print(f"  PLAN  {label:24s} N={ensemble_size} res={RESOLUTION_M} "
              f"scenario={scenario}", flush=True)
        return None

    print(f"  RUN   {label:24s} N={ensemble_size} res={RESOLUTION_M} ...",
          flush=True)
    t0 = time.time()
    try:
        result = replay.replay(
            policies=policies, scenario=scenario,
            ensemble_size=ensemble_size, resolution_m=RESOLUTION_M,
            workers=WORKERS, deadline_min=DEADLINE_MIN,
            output_root=out_root, seed=SEED)
    except Exception:
        print(f"  FAIL  {label:24s} after {timedelta(seconds=int(time.time()-t0))}",
              flush=True)
        traceback.print_exc()
        return None

    print(f"  DONE  {label:24s} {timedelta(seconds=int(time.time()-t0))}  "
          f"-> {result.out_dir.name}", flush=True)
    print(result.summary(), flush=True)
    return result.scored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny run (N=4) to prove the FARSITE path, then stop")
    ap.add_argument("--decisions-only", action="store_true",
                    help="the 8 N=48 arms only (~16 h)")
    ap.add_argument("--accuracy-only", action="store_true",
                    help="the 2 N=300 arms only (~25 h)")
    ap.add_argument("--table", action="store_true",
                    help="reprint the table from completed runs, run nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run, without running it")
    ap.add_argument("--no-clean-control", action="store_true",
                    help="drop the two N=48 clean arms (10 runs instead of 12, ~41 h)")
    ap.add_argument("--output-root", default=str(replay.OUTPUT_ROOT))
    args = ap.parse_args(argv)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    decision = (DECISION_ARMS_NO_CONTROL if args.no_clean_control
                else DECISION_ARMS)
    if args.smoke:
        arms = [("smoke fixed/clean", "fixed", "clean", 4)]
    elif args.decisions_only:
        arms = decision
    elif args.accuracy_only:
        arms = ACCURACY_ARMS
    else:
        arms = decision + ACCURACY_ARMS           # cheap group first, on purpose

    est = sum(2.0 if n == 48 else 12.6 if n == 300 else 0.5 for _, _, _, n in arms)
    print(f"{len(arms)} arm(s), roughly {est:.0f} h if none are already complete")
    print(f"output root: {out_root}\n")

    scored: dict[str, dict] = {}
    if not args.table:
        for label, policies, scenario, n in arms:
            got = run_arm(label, policies, scenario, n, out_root,
                          dry_run=args.dry_run)
            if got is not None:
                scored[label] = got
            print(flush=True)

    # Pick up every completed run, including ones from earlier invocations.
    for label, policies, scenario, n in (decision + ACCURACY_ARMS):
        if label in scored:
            continue
        done = _already_done(out_root, policies, scenario, n)
        if done is not None:
            scored[label] = metrics.load_metrics(done / "metrics.json")

    if not scored:
        print("No completed runs to tabulate.")
        return 0

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    table = metrics.compare_runs(scored)
    try:
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(table.to_string())
        table.to_csv(out_root / "arms_comparison.csv")
        print(f"\nwritten: {out_root / 'arms_comparison.csv'}")
    except ImportError:
        for row in table:
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
