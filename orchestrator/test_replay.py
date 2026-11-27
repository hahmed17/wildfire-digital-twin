"""
Tests for replay.py (§9) and metrics.py (§10), with FARSITE stubbed.

Most of these pin bugs found while building the two modules, so they are written as
regressions rather than as happy-path coverage:

  * a geometry fault must move the geometry, not just set a label
  * the previous perimeter must be loaded with its feature id
  * accuracy must be scored against ground truth, not against what was assimilated
  * the baseline in outputs/Border_2_Real must never be written to

    python -m pytest orchestrator/test_replay.py -v
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Point

from orchestrator import adapter, metrics, replay


# ---------------------------------------------------------------------------
# Stub FARSITE: grow the perimeter about its own centroid.
# (Scaling the raw EPSG:5070 coordinates instead translates the fire hundreds of
# km — a mistake worth not repeating in a fixture.)
# ---------------------------------------------------------------------------

def _stub_ensemble(growth_per_hour: float = 0.03):
    def _run(state_vec, cov_X, plan, weather, start_time, dt, rng, n_vertex=20, **kw):
        v = np.asarray(state_vec).reshape(-1, 1)
        hours = max(dt.total_seconds() / 3600.0, 0.1)
        scale = 1.0 + growth_per_hour * hours
        cx, cy = v[0::2].mean(), v[1::2].mean()
        g = v.copy()
        g[0::2] = cx + (v[0::2] - cx) * scale
        g[1::2] = cy + (v[1::2] - cy) * scale
        # Seed from the timestamp itself, NOT hash(str(...)): str hashing is
        # randomised per process, which made this stub produce different ensembles
        # on every run and turned arm-vs-arm comparisons into coin flips.
        r = np.random.default_rng(int(start_time.timestamp()) % 2**32)
        Z = g + r.normal(0, 120.0, (len(v), plan.ensemble_size))
        raw = {"zkphat_ensemble": Z,
               "xkhat_ensemble": v + r.normal(0, 80.0, (len(v), plan.ensemble_size)),
               "ykhat_ensemble": Z.copy(), "zero_samples": [],
               "member_results": [{"wall_s": 1.0} for _ in range(plan.ensemble_size)]}
        return adapter.EnsembleResult(
            raw=raw, plan=plan,
            wind_speeds=np.full(plan.ensemble_size, 10.0),
            wind_directions=np.full(plan.ensemble_size, 225.0),
            wall_s=float(plan.ensemble_size))
    return _run


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(adapter, "run_ensemble", _stub_ensemble())


@pytest.fixture
def perimeters_available():
    p = Path(replay.PERIMETER_FILE)
    if not p.exists():
        pytest.skip(f"{p} not present")
    return str(p)


def run(tmp_path, policies="fixed", scenario="clean", **kw):
    opts = dict(ensemble_size=6, workers=2, resolution_m=300,
                lcp_path="unused.lcp", output_root=tmp_path / "out", seed=42)
    opts.update(kw)
    return replay.replay(policies=policies, scenario=scenario, **opts)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_iou_of_identical_shapes_is_one(self):
        c = Point(0, 0).buffer(1000)
        assert metrics.iou(c, c) == pytest.approx(1.0)

    def test_iou_of_disjoint_shapes_is_zero(self):
        assert metrics.iou(Point(0, 0).buffer(100),
                           Point(10_000, 0).buffer(100)) == pytest.approx(0.0)

    def test_iou_penalises_displacement_that_area_error_misses(self):
        """The reason IoU is the headline number: a perimeter of exactly the right
        size in the wrong place has zero area error but poor IoU."""
        truth = Point(0, 0).buffer(1000)
        moved = translate(truth, xoff=1200.0)
        assert metrics.area_error(moved, truth)["area_error_m2"] == pytest.approx(0.0, abs=1.0)
        assert metrics.iou(moved, truth) < 0.4

    def test_area_error_is_signed(self):
        truth = Point(0, 0).buffer(1000)
        assert metrics.area_error(Point(0, 0).buffer(1200), truth)["area_error_m2"] > 0
        assert metrics.area_error(Point(0, 0).buffer(800), truth)["area_error_m2"] < 0

    def test_centroid_offset(self):
        truth = Point(0, 0).buffer(1000)
        off = metrics.centroid_offset_m(translate(truth, xoff=500.0), truth)
        assert off == pytest.approx(500.0, abs=1.0)

    def test_missing_geometry_gives_none_not_zero(self):
        """None means 'could not score'; 0.0 would read as a total miss."""
        assert metrics.iou(None, Point(0, 0).buffer(10)) is None
        assert metrics.centroid_offset_m(None, None) is None

    def test_non_decision_nodes_stay_out_of_the_counts(self):
        """forecast logs for provenance but decides nothing, so it must not appear
        as a 'forecast: None' row in decisions_per_node."""
        from orchestrator.state import DecisionRecord
        from datetime import datetime, timezone
        t = datetime(2025, 1, 1, tzinfo=timezone.utc)
        log = [
            DecisionRecord(sim_time=t, node="observation", policy="RulePolicy",
                           inputs={}, decision={"action": "accept"},
                           overridden_by_guardrail=False, agent_latency_s=0.1),
            DecisionRecord(sim_time=t, node="forecast", policy="rule", inputs={},
                           decision={"ensemble_path": "x.npz"},
                           overridden_by_guardrail=False, agent_latency_s=1.0),
        ]
        summary = metrics.behaviour_summary(log)
        assert "forecast" not in summary["decisions_per_node"]
        assert summary["decisions_per_node"]["observation"] == {"accept": 1}
        # but the row survives in the CSV, for provenance
        assert any(r["node"] == "forecast" for r in metrics.decision_rows(log))


# ---------------------------------------------------------------------------
# build_events
# ---------------------------------------------------------------------------

class TestBuildEvents:

    def test_seed_is_separate_from_the_observations(self, perimeters_available):
        seed, events = replay.build_events("clean")
        assert len(events) == len(replay.SELECTED_IDX) - 1 == 3
        assert seed.feature_id == str(replay.SELECTED_IDX[0])
        assert seed.obs_time == replay.OBS_TIMESTAMPS[0]
        assert seed.feature_id not in [e.feature_id for e in events]

    def test_timestamps_match_the_notebook(self, perimeters_available):
        _, events = replay.build_events("clean")
        assert [e.obs_time for e in events] == replay.OBS_TIMESTAMPS[1:]

    def test_geometry_fault_without_inject_dir_raises(self, perimeters_available):
        """It must not silently fall back to a label-only fault — that scenario
        would behave exactly like 'clean' and test nothing."""
        with pytest.raises(ValueError, match="inject_dir"):
            replay.build_events("shifted_obs")

    def test_shifted_obs_really_moves_the_geometry(self, tmp_path,
                                                   perimeters_available):
        import geopandas as gpd
        _, events = replay.build_events("shifted_obs", inject_dir=tmp_path)
        faulted = [e for e in events if e.injected == "shifted_2km"]
        assert len(faulted) == 1
        ev = faulted[0]
        assert Path(ev.path) != Path(replay.PERIMETER_FILE), "must not edit the source"

        src = gpd.read_file(replay.PERIMETER_FILE).to_crs(epsg=5070)
        got = gpd.read_file(ev.path).to_crs(epsg=5070).geometry.iloc[0]
        orig = src.geometry.iloc[replay.SELECTED_IDX[replay.FAULT_CYCLE]]
        assert got.centroid.x - orig.centroid.x == pytest.approx(2000.0, abs=1.0)
        assert got.area == pytest.approx(orig.area, rel=1e-9)   # same size, moved

    def test_mislabeled_obs_substitutes_an_earlier_perimeter(self, tmp_path,
                                                             perimeters_available):
        import geopandas as gpd
        _, events = replay.build_events("mislabeled_obs", inject_dir=tmp_path)
        ev = [e for e in events if e.injected == "mislabeled"][0]
        src = gpd.read_file(replay.PERIMETER_FILE).to_crs(epsg=5070)
        got = gpd.read_file(ev.path).to_crs(epsg=5070).geometry.iloc[0]
        assert got.area == pytest.approx(
            src.geometry.iloc[replay.SELECTED_IDX[0]].area, rel=1e-9)

    def test_stale_obs_delays_arrival_past_the_threshold(self, perimeters_available):
        _, events = replay.build_events("stale_obs")
        stale = [e for e in events if e.injected == "stale"][0]
        assert stale.latency_min > 360.0

    def test_source_file_is_never_modified(self, tmp_path, perimeters_available):
        before = hashlib.md5(Path(replay.PERIMETER_FILE).read_bytes()).hexdigest()
        for sc in ("clean", "shifted_obs", "mislabeled_obs", "stale_obs"):
            replay.build_events(sc, inject_dir=tmp_path)
        assert hashlib.md5(Path(replay.PERIMETER_FILE).read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

class TestReplay:

    def test_writes_its_own_directory_with_the_expected_files(
            self, tmp_path, stubbed, perimeters_available):
        r = run(tmp_path)
        for name in ("config.json", "steps.csv", "decisions.csv", "metrics.json",
                     "state.npz"):
            assert (r.out_dir / name).exists(), f"{name} missing"
        assert list((r.out_dir / "ensembles").glob("step*.npz"))
        assert r.out_dir.name.startswith("fixed_clean_")

    def test_baseline_outputs_are_never_touched(self, tmp_path, stubbed,
                                                perimeters_available):
        base = Path("outputs/Border_2_Real")
        if not base.exists():
            pytest.skip("baseline outputs not present")
        snap = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
                for p in sorted(base.iterdir()) if p.is_file()}
        run(tmp_path, policies="rule", scenario="shifted_obs")
        after = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
                 for p in sorted(base.iterdir()) if p.is_file()}
        assert snap == after

    def test_config_records_what_was_run(self, tmp_path, stubbed,
                                         perimeters_available):
        r = run(tmp_path, policies="rule", scenario="stale_obs")
        cfg = json.loads((r.out_dir / "config.json").read_text())
        assert cfg["policies"] == "rule" and cfg["scenario"] == "stale_obs"
        assert cfg["selected_idx"] == replay.SELECTED_IDX
        assert len(cfg["obs_timestamps"]) == 4

    def test_first_cycle_forecasts_from_the_seed_not_a_default_window(
            self, tmp_path, stubbed, perimeters_available):
        """The regression this exists for: the first cycle silently forecast one
        hour instead of the 26.2 between the seed and the first observation.

        cache["good_obs"] is only filled once an observation has been accepted, so
        on the first cycle forecast_window has no analysis time to work from and
        falls back to DEFAULT_WINDOW_HOURS unless cache["seed_obs_time"] is set.
        The bug was invisible three ways over: the scores stayed plausible, the
        cheap first ensemble looked like a small fire rather than a short forecast,
        and a fix that set the key immediately after initial_state was discarded by
        the next statement, which replaces the whole cache. Nothing in the run
        output recorded the interval, so this asserts on the recorded window
        rather than on anything derived from it.
        """
        import csv
        r = run(tmp_path)
        with open(r.out_dir / "decisions.csv") as f:
            weather = [row for row in csv.DictReader(f)
                       if row.get("node") == "weather"]
        assert weather, "no weather decisions recorded"
        first = float(weather[0]["window_hours"])
        seed_gap = (replay.OBS_TIMESTAMPS[replay.SELECTED_IDX.index(
            replay.SELECTED_IDX[1])] - replay.OBS_TIMESTAMPS[0])
        expected = seed_gap.total_seconds() / 3600
        assert first == pytest.approx(expected, abs=0.05), (
            f"first cycle forecast {first} h, expected {expected:.3f} h "
            "(seed to first observation)")
        assert first > 2.0, (
            f"first cycle forecast only {first} h -- this is the "
            "DEFAULT_WINDOW_HOURS fallback, so seed_obs_time never reached the "
            "weather agent")

    def test_seed_obs_time_survives_cache_construction(
            self, tmp_path, stubbed, perimeters_available):
        """replay assigns st["cache"] from a literal, so a key set before that
        assignment is thrown away. Guards the shape of the fix, not just its effect.
        """
        import inspect
        src = inspect.getsource(replay.replay)
        assigns = [i for i, line in enumerate(src.splitlines())
                   if 'st["cache"] =' in line]
        assert len(assigns) == 1, (
            f"{len(assigns)} assignments to st['cache']; a later one discards "
            "whatever an earlier one set")
        assert "seed_obs_time" in src

    def test_three_cycles_for_the_four_border2_perimeters(
            self, tmp_path, stubbed, perimeters_available):
        r = run(tmp_path)
        assert len(r.scored["cycles"]) == 3

    def test_steps_csv_uses_the_baseline_schema(self, tmp_path, stubbed,
                                                perimeters_available):
        """So an arm and the baseline concatenate for comparison."""
        import csv
        r = run(tmp_path)
        with open(r.out_dir / "steps.csv") as f:
            got = next(csv.reader(f))
        assert got == ["run_id", "step", "t_obs", "N", "n_valid",
                       "t_ensemble_s", "t_enkf_s", "vsize", "wsize"]

    def test_fixed_policy_uses_the_baseline_noise(self, tmp_path, stubbed,
                                                  perimeters_available):
        from orchestrator.state import BASELINE
        r = run(tmp_path, policies="fixed")
        for c in r.scored["cycles"]:
            assert c["vsize_m"] == BASELINE["vsize_m"]
            assert c["wsize_m"] == BASELINE["wsize_m"]


class TestFaultsAreDetectable:
    """The point of the scenarios: rule must catch what fixed misses. Without this
    the whole policy comparison is vacuous."""

    @pytest.mark.parametrize("scenario", ["shifted_obs", "mislabeled_obs", "stale_obs"])
    def test_rule_rejects_and_fixed_does_not(self, tmp_path, stubbed,
                                             perimeters_available, scenario):
        fixed = run(tmp_path, "fixed", scenario)
        rule = run(tmp_path, "rule", scenario)

        def rejects(r):
            return (r.scored["behaviour"]["decisions_per_node"]
                    .get("observation", {}).get("reject", 0))

        assert rejects(fixed) == 0, "FixedPolicy must accept everything (it is the control)"
        assert rejects(rule) == 1, f"RulePolicy missed the {scenario} fault"

    def test_synthetic_replacement_is_judged_as_synthetic(self, tmp_path, stubbed,
                                                          perimeters_available):
        """After a reject, assimilation must see the SYNTHETIC perimeter's signals, not the
        rejected observation's: source synthetic_perimeter, so the 500 m vsize applies.
        (synthesize used to leave the rejected observation's signals in the state.)"""
        import pandas as pd
        r = run(tmp_path, "rule", "shifted_obs")
        d = pd.read_csv(r.out_dir / "decisions.csv")
        assim = d[d.node == "assimilation"].reset_index(drop=True)
        cycle2 = assim.iloc[1]          # the faulted cycle (FAULT_CYCLE = 2)
        assert cycle2.vsize_m == 500.0, cycle2.reason
        assert assim.iloc[0].vsize_m == 200.0 and assim.iloc[2].vsize_m == 200.0

    def test_clean_scenario_has_no_rejections(self, tmp_path, stubbed,
                                              perimeters_available):
        """A policy that rejects real observations is worse than useless."""
        r = run(tmp_path, "rule", "clean")
        obs = r.scored["behaviour"]["decisions_per_node"].get("observation", {})
        assert obs.get("reject", 0) == 0

    def test_weather_failure_is_flagged_by_both_but_only_fixed_needs_a_guardrail(
            self, tmp_path, stubbed, perimeters_available):
        fixed = run(tmp_path, "fixed", "weather_failure")
        rule = run(tmp_path, "rule", "weather_failure")
        wx = lambda r: r.scored["behaviour"]["decisions_per_node"].get("weather", {})
        assert wx(fixed).get("fallback_weather", 0) == 3
        assert wx(rule).get("fallback_weather", 0) == 3
        # FixedPolicy says "normal" and is overridden; RulePolicy already says so
        assert fixed.scored["behaviour"]["guardrail_overrides"] == 3
        assert rule.scored["behaviour"]["guardrail_overrides"] == 0


class TestScoringAgainstGroundTruth:
    """Accuracy must be measured against the true perimeter, not against whatever
    the filter was handed. Scoring against the assimilated observation rewarded an
    arm for faithfully reproducing a fault: with a mislabeled observation, the arm
    that swallowed it scored 0.257 against the arm that rejected it at 0.164."""

    def test_truth_map_covers_every_cycle_and_ignores_faults(
            self, tmp_path, perimeters_available):
        truth = replay._truth_perimeters(replay.PERIMETER_FILE, replay.N_VERTEX)
        assert sorted(truth) == [1, 2, 3]

        clean = replay._truth_perimeters(replay.PERIMETER_FILE, replay.N_VERTEX)
        replay.build_events("shifted_obs", inject_dir=tmp_path)   # writes a fault
        after = replay._truth_perimeters(replay.PERIMETER_FILE, replay.N_VERTEX)
        for k in clean:
            assert np.allclose(np.asarray(clean[k]).ravel(),
                               np.asarray(after[k]).ravel()), (
                "ground truth must come from the source file, not the faulted copy")

    def test_scoring_ignores_the_fault_so_the_arms_are_comparable(
            self, tmp_path, stubbed, perimeters_available):
        """Both arms must be scored against the same ground truth.

        Deliberately NOT asserted here: that rejecting the bad observation yields a
        better IoU. Whether rejection helps is the experiment's finding, not a
        property of this code — and with a stub forecast that never translates the
        fire it comes out either way. Baking a hoped-for result into the suite
        would make the tests agree with us regardless of what the real run shows.
        """
        fixed = run(tmp_path, "fixed", "mislabeled_obs")
        rule = run(tmp_path, "rule", "mislabeled_obs")

        # identical ground truth for both arms
        truth = replay._truth_perimeters(replay.PERIMETER_FILE, replay.N_VERTEX)
        for arm in (fixed, rule):
            assert len(arm.scored["cycles"]) == len(truth)
            assert all(c.get("iou") is not None for c in arm.scored["cycles"])

        # and the arms differ somewhere, so the comparison is not vacuous
        assert (fixed.scored["behaviour"]["decisions_per_node"]
                != rule.scored["behaviour"]["decisions_per_node"])

    def test_cycles_record_how_far_the_assimilated_obs_was_from_truth(
            self, tmp_path, stubbed, perimeters_available):
        r = run(tmp_path, "rule", "mislabeled_obs")
        faulted = [c for c in r.scored["cycles"]
                   if c.get("assimilated_vs_truth_iou") is not None]
        assert faulted, "no assimilated-vs-truth comparison recorded"
        # the faulted cycle's assimilated perimeter is further from truth than a clean one
        ious = [c["assimilated_vs_truth_iou"] for c in faulted]
        assert min(ious) < max(ious)


class TestCompare:

    def test_compare_produces_one_row_per_arm(self, tmp_path, stubbed,
                                              perimeters_available):
        rs = [run(tmp_path, p, s) for p in ("fixed", "rule")
              for s in ("clean", "shifted_obs")]
        table = replay.compare(rs)
        try:
            import pandas as pd
            assert isinstance(table, pd.DataFrame) and len(table) == 4
            assert "iou_mean" in table.columns
        except ImportError:
            assert len(table) == 4

    def test_load_run_round_trips(self, tmp_path, stubbed, perimeters_available):
        r = run(tmp_path)
        loaded = replay.load_run(r.out_dir)
        assert loaded["config"]["run_id"] == r.run_id
        assert len(loaded["scored"]["cycles"]) == len(r.scored["cycles"])