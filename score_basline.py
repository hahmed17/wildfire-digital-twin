"""
Score the notebook baseline, mapping each saved state to a cycle by its recorded
t_obs rather than by the step number in its filename.

    python score_b2.py

A filename cannot say whether a state was saved before or after its step's
assimilation, and the two differ by a whole cycle. Scoring a saved seed against
the first assimilated perimeter reports a large under-prediction that is really
just the seed being smaller than the fire later became -- which reads as a
plausible first-cycle result rather than as a mapping error. t_obs settles it.
"""
import numpy as np
import orchestrator  # noqa: F401  -- puts src/ on sys.path
import config
from orchestrator import metrics, replay

base = config.OUTPUT_DIR / "Border_2_Real"
cyc_of = {str(t)[:16]: c for c, t in enumerate(replay.OBS_TIMESTAMPS[1:], 1)}
seed = str(replay.OBS_TIMESTAMPS[0])[:16]
truth = replay._truth_perimeters(replay.PERIMETER_FILE, 20)

print(f"baseline: {base}")
print(f"seed time: {seed}")
print(f"cycle times: {cyc_of}\n")

scored = {}
for f in sorted(base.glob("state_step*.npz")):
    d = np.load(f, allow_pickle=False)
    if "t_obs" not in d.files:
        print(f"{f.name}  no t_obs recorded -- cannot place it")
        continue
    stamp = str(d["t_obs"])[:16]
    cyc = cyc_of.get(stamp)
    if stamp == seed:
        print(f"{f.name}  t_obs={stamp}  SEED, not scored")
    elif cyc is None:
        print(f"{f.name}  t_obs={stamp}  matches no cycle")
    else:
        s = metrics.score_perimeter(np.asarray(d["state"]).ravel(), truth[cyc])
        scored[cyc] = s
        print(f"{f.name}  t_obs={stamp}  cycle {cyc}   "
              f"iou={s['iou']:.4f}  area_err={s['area_error_frac']:+.4f}  "
              f"centroid={s['centroid_offset_m']:.0f} m")

if scored:
    ks = sorted(scored)
    ious = [scored[k]["iou"] for k in ks]
    print(f"\niou_mean      {np.mean(ious):.6f}   (over cycles {ks})")
    print(f"iou_last      {ious[-1]:.6f}   (cycle {ks[-1]})")
    print(f"abs_area_err  {np.mean([abs(scored[k]['area_error_frac']) for k in ks]):.6f}")
    print(f"centroid_m    {np.mean([scored[k]['centroid_offset_m'] for k in ks]):.1f}")
    absent = [c for c in cyc_of.values() if c not in scored]
    if absent:
        print(f"\nNo analysis saved for cycle(s) {absent}. A final-cycle "
              "comparison with the\narms is therefore not like-for-like: compare "
              "per cycle, not on the aggregates.")
else:
    print("\nNothing scored.")
