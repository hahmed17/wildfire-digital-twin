"""
Tests for the forecast and update nodes (§8) — the half of the loop that touches
src/enkf.py.

FARSITE is stubbed: adapter.run_ensemble is replaced by a fake that returns an ensemble
of the right shape. Everything else is real — the real EnKF math in enkf.enkf_update,
the real .npz persistence, the real graph. So these tests exercise the plumbing a live
Border 2 run depends on, without the hours of compute.

    python -m pytest orchestrator/test_steps.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from orchestrator import adapter, steps
from orchestrator.graph import build_pipeline, run_pipeline
from orchestrator.placeholders import placeholder_nodes
from orchestrator.state import (
    BASELINE,
    AssimilationDecision,
    ForecastPlan,
    ObservationEvent,
    OperatorRequest,
    initial_state,
)

T0 = datetime(2025, 1, 23, 18, 22, tzinfo=timezone.utc)   # Border 2 t0
N_VERTEX = 20
N_STATES = 2 * N_VERTEX

CALM = {"wind_speeds": [8.0, 9.0], "wind_directions": [220.0, 225.0],
        "used_fallback": False}
HIGH = {"wind_speeds": [30.0, 31.0], "wind_directions": [220.0, 225.0],
        "used_fallback": False}


# ---------------------------------------------------------------------------
# Fixtures: a growing series of circular perimeters, as the real file is a
# growing series of mapped ones.
# ---------------------------------------------------------------------------

@pytest.fixture
def perimeters(tmp_path):
    """One GeoJSON with 4 concentric perimeters, feature i growing with i."""
    cx, cy = -2_050_000.0, 1_800_000.0          # plausible EPSG:5070 metres
    geoms = [Point(cx, cy).buffer(r) for r in (1500, 2600, 3300, 3500)]
    path = tmp_path / "perims.geojson"
    gpd.GeoDataFrame(geometry=geoms, crs="EPSG:5070").to_file(path, driver="GeoJSON")
    return str(path)


def event(perimeters: str, i: int, hours: float) -> ObservationEvent:
    t = T0 + timedelta(hours=hours)
    return ObservationEvent(obs_time=t, arrival_time=t + timedelta(minutes=5),
                            source="mapped_perimeter", path=perimeters,
                            feature_id=str(i))


def fake_run_ensemble(scale: float = 1.0, seed: int = 0):
    """Stand in for adapter.run_ensemble.

    Builds a forecast ensemble by perturbing the prior state outwards, which is what a
    real FARSITE ensemble does to a growing fire. `scale` controls how far it grows, so
    a test can make the forecast agree with or miss the observation.
    """
    def _run(state_vec, cov_X, plan, weather, start_time, dt, rng,
             n_vertex=N_VERTEX, **kw):
        n = plan.ensemble_size
        v = np.asarray(state_vec).reshape(-1, 1)
        centre = np.array([[v[0::2].mean()], [v[1::2].mean()]])
        grown = v.copy()
        grown[0::2] = centre[0] + (v[0::2] - centre[0]) * scale
        grown[1::2] = centre[1] + (v[1::2] - centre[1]) * scale

        r = np.random.default_rng(seed)
        Z = grown + r.normal(0, 60.0, (len(v), n))
        raw = {"zkphat_ensemble": Z,
               "xkhat_ensemble": v + r.normal(0, 30.0, (len(v), n)),
               "ykhat_ensemble": Z.copy(),
               "zero_samples": [],
               "member_results": [{"wall_s": 1.0} for _ in range(n)]}
        return adapter.EnsembleResult(raw=raw, plan=plan,
                                      wind_speeds=np.full(n, 10.0),
                                      wind_directions=np.full(n, 225.0),
                                      wall_s=1.0)
    return _run


def base_state(perimeters, tmp_path, step=1, **over):
    ev = event(perimeters, 1, 26.2)
    st = dict(initial_state([ev], OperatorRequest(deadline_min=60),
                            str(tmp_path / "state.npz")))
    st.update(current=ev, pending=[], clock=ev.arrival_time, step=step,
              plan=ForecastPlan(ensemble_size=8, resolution_m=300, workers=2,
                                reason="test"),
              cache={"weather_last": CALM, "run_dir": str(tmp_path / "run")})
    st.update(over)
    return st


# ---------------------------------------------------------------------------
# seed_state
# ---------------------------------------------------------------------------

class TestSeedState:

    def test_writes_a_state_the_forecast_can_load(self, perimeters, tmp_path):
        path = tmp_path / "state.npz"
        steps.seed_state(path, event(perimeters, 0, 0), n_vertex=N_VERTEX, seed=7)

        vec, cov, rng = adapter.load_state(path)
        assert vec.reshape(-1, 1).shape == (N_STATES, 1)
        assert cov.shape == (N_STATES, N_STATES)
        assert rng is not None

    def test_initial_covariance_matches_the_notebook(self, perimeters, tmp_path):
        """The notebook sets X = 1000 * I before the first step."""
        path = tmp_path / "state.npz"
        steps.seed_state(path, event(perimeters, 0, 0), n_vertex=N_VERTEX)
        _, cov, _ = adapter.load_state(path)
        assert np.allclose(cov, 1000.0 * np.eye(N_STATES))

    def test_seeded_state_is_the_first_perimeter(self, perimeters, tmp_path):
        path = tmp_path / "state.npz"
        ev = event(perimeters, 0, 0)
        steps.seed_state(path, ev, n_vertex=N_VERTEX)
        vec, _, _ = adapter.load_state(path)
        assert np.allclose(np.asarray(vec).ravel(),
                           np.asarray(adapter.load_observation(ev, N_VERTEX)).ravel())


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

class TestForecast:

    def test_missing_state_file_says_how_to_seed_it(self, perimeters, tmp_path):
        node = steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)
        with pytest.raises(FileNotFoundError, match="seed_state"):
            node(base_state(perimeters, tmp_path))

    def test_missing_plan_names_the_observation_agent(self, perimeters, tmp_path):
        steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX)
        node = steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)
        with pytest.raises(ValueError, match="observation agent"):
            node(base_state(perimeters, tmp_path, plan=None))

    def test_missing_weather_names_the_weather_agent(self, perimeters, tmp_path):
        steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX)
        node = steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)
        with pytest.raises(ValueError, match="weather agent"):
            node(base_state(perimeters, tmp_path, cache={}))

    def test_writes_an_ensemble_update_can_read(self, perimeters, tmp_path, monkeypatch):
        monkeypatch.setattr(adapter, "run_ensemble", fake_run_ensemble(1.2))
        steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX)

        out = steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)(
            base_state(perimeters, tmp_path))

        data = np.load(out["ensemble_path"], allow_pickle=True)
        for key in ("zkphat_ensemble", "xkhat_ensemble", "ykhat_ensemble",
                    "zero_samples", "wind_speeds", "wind_directions",
                    "ensemble_size", "resolution_m", "workers"):
            assert key in data, f"{key} missing from the saved ensemble"
        assert data["zkphat_ensemble"].shape == (N_STATES, 8)
        assert len(out["log"]) == 1 and out["log"][0].node == "forecast"

    def test_forecast_window_matches_the_weather_window(self, perimeters, tmp_path,
                                                        monkeypatch):
        """The ensemble must be driven over the same interval the weather covers."""
        seen = {}

        def spy(state_vec, cov_X, plan, weather, start_time, dt, rng, **kw):
            seen["start"], seen["dt"] = start_time, dt
            return fake_run_ensemble()(state_vec, cov_X, plan, weather,
                                       start_time, dt, rng, **kw)

        monkeypatch.setattr(adapter, "run_ensemble", spy)
        steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX)

        prev = event(perimeters, 0, 0)
        st = base_state(perimeters, tmp_path)
        st["cache"] = {**st["cache"], "good_obs": [prev.model_dump(mode="json")]}
        steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)(st)

        from orchestrator.agents.weather import forecast_window
        exp_start, exp_end = forecast_window(st)
        assert seen["start"] == exp_start
        assert seen["dt"] == exp_end - exp_start

    @pytest.mark.parametrize("weather,expected", [
        (CALM, steps.FARSITE_DT_NORMAL),
        (HIGH, steps.FARSITE_DT_HIGH_WIND),
    ])
    def test_substep_shortens_in_high_wind(self, perimeters, tmp_path, monkeypatch,
                                           weather, expected):
        """The notebook uses 15 min sub-steps on high-wind steps, 30 otherwise."""
        seen = {}

        def spy(state_vec, cov_X, plan, weather, start_time, dt, rng,
                farsite_dt_minutes=30, **kw):
            seen["dt_min"] = farsite_dt_minutes
            return fake_run_ensemble()(state_vec, cov_X, plan, weather,
                                       start_time, dt, rng, **kw)

        monkeypatch.setattr(adapter, "run_ensemble", spy)
        steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX)
        steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)(
            base_state(perimeters, tmp_path, cache={"weather_last": weather,
                                                     "run_dir": str(tmp_path / "run")}))
        assert seen["dt_min"] == expected


# ---------------------------------------------------------------------------
# apply_assimilation — the four actions
# ---------------------------------------------------------------------------

def _forecast_once(perimeters, tmp_path, monkeypatch, scale=1.2, seed=5):
    """Seed, forecast once, and return the state ready for update.

    `seed` is passed through to seed_state so the RNG the update draws its observation
    perturbations from is fixed — without it two otherwise identical runs differ and
    comparisons between them are meaningless.
    """
    monkeypatch.setattr(adapter, "run_ensemble", fake_run_ensemble(scale))
    steps.seed_state(tmp_path / "state.npz", event(perimeters, 0, 0), N_VERTEX,
                     seed=seed)
    st = base_state(perimeters, tmp_path)
    out = steps.make_forecast(lcp_path="unused.lcp", n_vertex=N_VERTEX)(st)
    return {**st, **out}


def aligned_pair(state):
    """The observation and prior as the forecast aligned them.

    Tests must compare against these, not against a freshly loaded observation: the
    forecast rotates both onto a common vertex ordering, so the raw vectors differ in
    traversal order even when the shapes are identical.
    """
    data = np.load(state["ensemble_path"], allow_pickle=True)
    return data["aligned_obs"], data["aligned_prior"]


def verdict(action="assimilate", **kw):
    kw.setdefault("vsize_m", BASELINE["vsize_m"])
    kw.setdefault("wsize_m", BASELINE["wsize_m"])
    return AssimilationDecision(action=action, reason="test", **kw)


class TestApplyAssimilation:

    def test_nothing_to_do_without_a_verdict(self, perimeters, tmp_path):
        assert steps.apply_assimilation(base_state(perimeters, tmp_path)) == {}

    def test_placeholder_forecast_leaves_nothing_to_apply(self, perimeters, tmp_path):
        """A placeholder ensemble_path points at no file; update must not blow up,
        and must not need FARSITE to find that out."""
        st = base_state(perimeters, tmp_path,
                        ensemble_path="placeholder_step1.npz",
                        assim_verdict=verdict())
        out = steps.apply_assimilation(st)
        assert out["assim_applied"] is False

    def test_assimilate_moves_the_state_towards_the_observation(
            self, perimeters, tmp_path, monkeypatch):
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        obs, before = aligned_pair(st)

        st["assim_verdict"] = verdict("assimilate")
        out = steps.apply_assimilation(st, n_vertex=N_VERTEX)
        after, _, _ = adapter.load_state(st["state_path"])

        assert out["assim_applied"] is True
        d_before = np.linalg.norm(np.asarray(before).ravel() - np.asarray(obs).ravel())
        d_after = np.linalg.norm(np.asarray(after).ravel() - np.asarray(obs).ravel())
        assert d_after < d_before, (
            f"the analysis should sit closer to the observation: "
            f"{d_after:.0f} vs {d_before:.0f}")

    def test_inflation_moves_the_state_further(self, perimeters, tmp_path,
                                               monkeypatch):
        """Widening the prior spread lets the update travel further towards the
        observation, which is the point of inflate_and_assimilate."""
        obs = None
        moves = {}
        for label, v in [("plain", verdict("assimilate")),
                         ("inflated", verdict("inflate_and_assimilate",
                                              inflation=2.0))]:
            st = _forecast_once(perimeters, tmp_path, monkeypatch)
            before, _, _ = adapter.load_state(st["state_path"])
            obs = adapter.load_observation(st["current"], N_VERTEX)
            st["assim_verdict"] = v
            steps.apply_assimilation(st, n_vertex=N_VERTEX)
            after, _, _ = adapter.load_state(st["state_path"])
            moves[label] = np.linalg.norm(np.asarray(after).ravel()
                                          - np.asarray(before).ravel())
        assert moves["inflated"] > moves["plain"]

    def test_skip_keeps_the_forecast_and_ignores_the_observation(
            self, perimeters, tmp_path, monkeypatch):
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        data = np.load(st["ensemble_path"])
        forecast_mean = data["zkphat_ensemble"].mean(axis=1)

        st["assim_verdict"] = verdict("skip")
        out = steps.apply_assimilation(st, n_vertex=N_VERTEX)
        after, _, _ = adapter.load_state(st["state_path"])

        assert out["assim_applied"] is True
        assert np.allclose(np.asarray(after).ravel(), forecast_mean)

    def test_reinitialize_sets_the_state_to_the_observation(
            self, perimeters, tmp_path, monkeypatch):
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        obs, _ = aligned_pair(st)

        st["assim_verdict"] = verdict("reinitialize_from_obs", vsize_m=200.0)
        steps.apply_assimilation(st, n_vertex=N_VERTEX)
        after, cov, _ = adapter.load_state(st["state_path"])

        assert np.allclose(np.asarray(after).ravel(), np.asarray(obs).ravel())
        # default reinit_cov="observation" -> X = vsize^2 * I
        assert np.allclose(cov, 200.0 ** 2 * np.eye(N_STATES))

    def test_reinitialize_can_use_the_notebook_prior_instead(
            self, perimeters, tmp_path, monkeypatch):
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        st["assim_verdict"] = verdict("reinitialize_from_obs", vsize_m=200.0)
        steps.apply_assimilation(st, n_vertex=N_VERTEX, reinit_cov="initial")
        _, cov, _ = adapter.load_state(st["state_path"])
        assert np.allclose(cov, steps.REINIT_COV_INITIAL * np.eye(N_STATES))

    def test_metrics_carry_the_areas(self, perimeters, tmp_path, monkeypatch):
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        st["assim_verdict"] = verdict("assimilate")
        out = steps.apply_assimilation(st, n_vertex=N_VERTEX)
        assert out["analysis_area_m2"] > 0
        assert out["observed_area_m2"] > 0


class TestCovarianceIsCarriedForward:
    """adapter.enkf_update used to read the prior covariance off the ensemble dict,
    which enkf.run_ensemble does not populate — so it silently fell back to 1000 * I
    every cycle and the filter never accumulated covariance. Pin the fix."""

    def test_prior_covariance_reaches_enkf_update(self, perimeters, tmp_path,
                                                  monkeypatch):
        import enkf as enkf_mod
        seen = {}
        real = enkf_mod.enkf_update

        def spy(**kw):
            seen["X"] = np.array(kw["X"], copy=True)
            seen["initial_state"] = np.array(kw["initial_state"], copy=True)
            return real(**kw)

        monkeypatch.setattr(enkf_mod, "enkf_update", spy)

        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        # Seed a distinctive covariance so the old 1000 * I fallback would be obvious
        prior_vec, _, rng = adapter.load_state(st["state_path"])
        distinctive = 7777.0 * np.eye(N_STATES)
        adapter.save_state(st["state_path"], prior_vec, distinctive, rng)

        st["assim_verdict"] = verdict("assimilate")
        steps.apply_assimilation(st, n_vertex=N_VERTEX)

        assert np.allclose(seen["X"], distinctive), (
            "the carried-forward covariance did not reach enkf_update")
        # initial_state must be the prior the forecast was driven from (aligned),
        # not an ensemble mean
        _, aligned_prior = aligned_pair(st)
        assert np.allclose(np.asarray(seen["initial_state"]).ravel(),
                           np.asarray(aligned_prior).ravel()), (
            "initial_state should be the previous analysis, not an ensemble mean")

    def test_covariance_changes_between_cycles(self, perimeters, tmp_path,
                                               monkeypatch):
        """A filter that works updates its covariance; one that resets does not."""
        st = _forecast_once(perimeters, tmp_path, monkeypatch)
        _, cov_before, _ = adapter.load_state(st["state_path"])
        st["assim_verdict"] = verdict("assimilate")
        steps.apply_assimilation(st, n_vertex=N_VERTEX)
        _, cov_after, _ = adapter.load_state(st["state_path"])
        assert not np.allclose(cov_before, cov_after)


# ---------------------------------------------------------------------------
# Full replay through the graph, FARSITE stubbed
# ---------------------------------------------------------------------------

class TestFullReplayWithStubbedFarsite:
    """The shape of a real Border 2 run: perimeter 0 seeds the state, perimeters 1..3
    arrive as observations, three assimilation cycles follow."""

    @staticmethod
    def _run(perimeters, tmp_path, monkeypatch, obs_policy, assim_policy, scale=1.2):
        from orchestrator.agents.observation import make_observation_agent
        from orchestrator.agents.weather import make_weather_agent
        from orchestrator.agents.weather import RulePolicy as WxRule
        from orchestrator.agents.assimilation import make_assimilation_agent

        monkeypatch.setattr(adapter, "run_ensemble", fake_run_ensemble(scale))

        state_path = tmp_path / "state.npz"
        steps.seed_state(state_path, event(perimeters, 0, 0), N_VERTEX, seed=11)

        # Border 2's real observation times: t1, t2, t3
        events = [event(perimeters, 1, 26.2),
                  event(perimeters, 2, 33.45),
                  event(perimeters, 3, 77.83)]

        nodes = placeholder_nodes()
        nodes["observation"] = make_observation_agent(obs_policy)
        nodes["weather"] = make_weather_agent(WxRule(), fetch=lambda s, e: CALM)
        nodes["assimilation"] = make_assimilation_agent(assim_policy)
        nodes["forecast"] = steps.make_forecast(lcp_path="unused.lcp",
                                                n_vertex=N_VERTEX)

        st = initial_state(events, OperatorRequest(deadline_min=60), str(state_path))
        st["cache"] = {"run_dir": str(tmp_path / "run")}
        return run_pipeline(build_pipeline(nodes), st), state_path

    def test_three_cycles_complete_and_each_applies_an_update(
            self, perimeters, tmp_path, monkeypatch):
        from orchestrator.agents.observation import FixedPolicy as ObsFixed
        from orchestrator.agents.assimilation import FixedPolicy as AsFixed

        final, state_path = self._run(perimeters, tmp_path, monkeypatch,
                                      ObsFixed(), AsFixed())
        assert len(final["metrics"]) == 3
        for m in final["metrics"]:
            assert m["assim_applied"] is True, f"step {m['step']} applied no update"
            assert m["assim_action"] == "assimilate"

    def test_the_analysis_tracks_the_growing_fire(self, perimeters, tmp_path,
                                                 monkeypatch):
        from orchestrator.agents.observation import FixedPolicy as ObsFixed
        from orchestrator.agents.assimilation import FixedPolicy as AsFixed

        final, _ = self._run(perimeters, tmp_path, monkeypatch, ObsFixed(), AsFixed())
        areas = [m["analysis_area_m2"] for m in final["metrics"]]
        observed = [m["observed_area_m2"] for m in final["metrics"]]
        assert areas == sorted(areas), f"analysis area should grow, got {areas}"
        assert observed == sorted(observed)

    def test_fixed_uses_baseline_noise_every_step(self, perimeters, tmp_path,
                                                  monkeypatch):
        """FixedPolicy must reproduce the baseline run's vsize/wsize, which is what
        makes it a valid control arm."""
        from orchestrator.agents.observation import FixedPolicy as ObsFixed
        from orchestrator.agents.assimilation import FixedPolicy as AsFixed

        final, _ = self._run(perimeters, tmp_path, monkeypatch, ObsFixed(), AsFixed())
        for m in final["metrics"]:
            assert m["vsize_m"] == BASELINE["vsize_m"] == 300.0
            assert m["wsize_m"] == BASELINE["wsize_m"] == 100.0
            assert m["inflation"] == 1.0

    def test_state_file_is_left_loadable_for_the_next_run(
            self, perimeters, tmp_path, monkeypatch):
        from orchestrator.agents.observation import FixedPolicy as ObsFixed
        from orchestrator.agents.assimilation import FixedPolicy as AsFixed

        _, state_path = self._run(perimeters, tmp_path, monkeypatch,
                                  ObsFixed(), AsFixed())
        vec, cov, rng = adapter.load_state(state_path)
        assert np.asarray(vec).reshape(-1, 1).shape == (N_STATES, 1)
        assert np.isfinite(cov).all()

    def test_rule_policies_also_complete_three_cycles(self, perimeters, tmp_path,
                                                      monkeypatch):
        from orchestrator.agents.observation import RulePolicy as ObsRule
        from orchestrator.agents.assimilation import RulePolicy as AsRule

        final, _ = self._run(perimeters, tmp_path, monkeypatch, ObsRule(), AsRule())
        assert len(final["metrics"]) == 3
        assert all(m["assim_applied"] for m in final["metrics"])

    def test_reinitialize_cannot_fire_in_three_cycles(self, perimeters, tmp_path,
                                                      monkeypatch):
        """Documents the limit discussed for the experiment design: the drift streak is
        read on entry and gains one entry per cycle, so with drift_steps=3 the rule
        needs a fourth cycle. A badly missing forecast still never reinitialises here."""
        from orchestrator.agents.observation import RulePolicy as ObsRule
        from orchestrator.agents.assimilation import RulePolicy as AsRule

        # scale=1.0 -> the forecast never grows, so every observation disagrees
        final, _ = self._run(perimeters, tmp_path, monkeypatch, ObsRule(), AsRule(),
                             scale=1.0)
        actions = [m["assim_action"] for m in final["metrics"]]
        assert "reinitialize_from_obs" not in actions, (
            f"expected the streak to stay below the threshold, got {actions}")
