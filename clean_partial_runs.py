"""
Find (and optionally delete) arm directories left behind by an interrupted run.

    python clean_partial_runs.py            # list only, deletes nothing
    python clean_partial_runs.py --delete   # remove the incomplete ones

A run writes its metrics.json and then its config.json as the last two acts of
replay(). A directory missing either was interrupted, and run_arms will ignore it
and start that arm again in a fresh directory -- so the leftover is harmless but
occupies disk and makes the output root hard to read. A directory with both files
is a completed arm and is never touched here.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import orchestrator  # noqa: F401  -- puts src/ on sys.path
import config


def main(argv: list[str]) -> int:
    delete = "--delete" in argv
    root = config.OUTPUT_DIR / "Border_2_Orchestrator"
    if not root.exists():
        print(f"no output root at {root}")
        return 1

    complete: list[Path] = []
    partial: list[tuple[Path, list[str]]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        missing = [f for f in ("metrics.json", "config.json")
                   if not (d / f).exists()]
        (complete.append(d) if not missing else partial.append((d, missing)))

    def size_mb(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

    print(f"output root: {root}\n")
    print(f"complete ({len(complete)}) -- these are kept and will be SKIPPED on resume:")
    for d in complete:
        print(f"  {d.name}")

    if not partial:
        print("\nNo interrupted directories found. Re-run the same run_arms "
              "command and it will resume.")
        return 0

    total = sum(size_mb(d) for d, _ in partial)
    print(f"\nincomplete ({len(partial)}) -- {total:.1f} MB, these arms will be "
          "re-run from the start:")
    for d, missing in partial:
        print(f"  {d.name:44s} missing {', '.join(missing)}  ({size_mb(d):.1f} MB)")

    if not delete:
        print("\nNothing deleted. Pass --delete to remove the incomplete ones.")
        return 0

    for d, _ in partial:
        shutil.rmtree(d)
        print(f"  removed {d.name}")
    print(f"\nFreed about {total:.1f} MB. Re-run the same run_arms command to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
