"""
Tests for the three agent nodes (§6). No FARSITE, no network, no disk:
every external call the nodes make is injected.

    python -m pytest orchestrator/test_agents.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orchestrator.agents.assimilation import (
    make_assimilation_agent,
    FixedPolicy as AsFixed,
    RulePolicy as AsRule,
)
from orchestrator.agents.observation import (
    make_observation_agent,
    FixedPolicy as ObsFixed,
    RulePolicy as ObsRule,
)
from orchestrator.agents.weather import (
    FALLBACK_DIRECTIONS,
    FALLBACK_SPEEDS,
    forecast_window,
    make_weather_agent,
    FixedPolicy as WxFixed,
    RulePolicy as WxRule,
)
from orchestrator.graph import build_pipeline, run_pipeline
from orchestrator.placeholders import placeholder_nodes
from orchestrator.state import (
    BASELINE,
    VSIZE_PRESETS,
    ObservationEvent,
    OperatorRequest,
    Signals,
    WeatherVerdict,
    initial_state,
)

T0 = datetime(2025, 1, 23, 12, 0, tzinfo=timezone.utc)

CALM = {"wind_speeds": [8.0, 9.0], "wind_directions": [220.0, 225.0], "used_fallback": False}
SHIFT = {"wind_speeds": [30.0, 32.0], "wind_directions": [40.0, 45.0], "used_fallback": False}
HIGH_SAME_DIR = {"wind_speeds": [30.0, 31.0], "wind_directions": [220.0, 225.0],
                 "used_fallback": False}
FALLBACK = {"wind_speeds": list(FALLBACK_SPEEDS),
            "wind_directions": list(FALLBACK_DIRECTIONS), "used_fallback": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_event(i: int = 0, **kw) -> ObservationEvent:
    defaults = dict(
        obs_time=T0 + timedelta(hours=3 * i),
        arrival_time=T0 + timedelta(hours=3 * i, minutes=5),
        source="mapped_perimeter",
        path="p.geojson",
        feature_id=str(i),
    )
    defaults.update(kw)
    return ObservationEvent(**defaults)


@pytest.fixture
def state():
    """A mid-replay state: one event in hand, a previous observation on record."""
    ev = make_event(1)
    st = dict(initial_state([ev], OperatorRequest(deadline_min=60), "out.npz"))
    st["current"] = ev
    st["pending"] = []
    st["clock"] = ev.arrival_time
    st["prev_obs_path"] = "prev.geojson"   # an established state exists
    return st


def signals(**kw) -> Signals:
    """Signals with everything passing; override one field per test."""
    base = dict(
        normalized_innovation=0.8,
        agrees_with_firms=None,
        contains_previous=True,
        growth_plausible=True,
        obs_age_min=5.0,
        source="mapped_perimeter",
        wind_shift=False,
        consecutive_disagreements=0,
    )
    base.update(kw)
    return Signals(**base)


def ensemble_with_innovation(target: float, source: str = "mapped_perimeter", seed: int = 0):
    """An (ensemble, obs) pair whose normalized_innovation is approximately `target`."""
    rng = np.random.default_rng(seed)
    n_states, n_members = 40, 50
    Z = rng.normal(0, 100, (n_states, n_members))
    spread = np.linalg.norm(Z - Z.mean(axis=1, keepdims=True), axis=0).mean()
    offset = target * (spread + VSIZE_PRESETS[source]) / np.sqrt(n_states)
    obs = Z.mean(axis=1, keepdims=True) + offset
    return lambda _state: (Z, obs)


NO_ENSEMBLE = lambda _state: (None, None)   # noqa: E731


# ---------------------------------------------------------------------------
# Observation agent
# ---------------------------------------------------------------------------

class TestObservationAgent:

    def test_fixed_accepts_with_baseline_plan(self, state):
        out = make_observation_agent(ObsFixed())(state)
        assert out["obs_verdict"].action == "accept"
        assert out["plan"].ensemble_size == BASELINE["ensemble_size"]
        assert out["plan"].resolution_m == BASELINE["resolution_m"]
        assert out["plan"].workers == BASELINE["workers"]

    def test_returns_signals_and_one_log_record(self, state):
        out = make_observation_agent(ObsFixed())(state)
        assert isinstance(out["signals"], Signals)
        assert len(out["log"]) == 1
        assert out["log"][0].node == "observation"
        assert out["log"][0].policy == "FixedPolicy"

    def test_rule_rejects_stale_observation(self, state):
        # clock is 48 h after obs_time -> far past the 360 min default
        state["clock"] = state["current"].obs_time + timedelta(hours=48)
        out = make_observation_agent(ObsRule())(state)
        assert out["obs_verdict"].action == "reject"
        assert "old" in out["obs_verdict"].reason

    def test_rule_rejects_injected_fault_when_configured(self, state):
        state["current"] = make_event(1, injected="shifted_2km")
        out = make_observation_agent(ObsRule(reject_injected=True))(state)
        assert out["obs_verdict"].action == "reject"
        assert "shifted_2km" in out["obs_verdict"].reason

    def test_rule_ignores_injected_fault_by_default(self, state):
        state["current"] = make_event(1, injected="shifted_2km")
        out = make_observation_agent(ObsRule())(state)
        assert out["obs_verdict"].action == "accept"

    def test_guardrail_accepts_first_observation(self, state):
        """With no previous perimeter, a reject must be overridden: the first
        observation initialises the state, so it cannot be thrown away."""
        state["prev_obs_path"] = None
        state["clock"] = state["current"].obs_time + timedelta(hours=48)   # would reject
        out = make_observation_agent(ObsRule())(state)
        assert out["obs_verdict"].action == "accept"
        assert out["obs_verdict"].override_guardrail is True
        assert out["log"][0].overridden_by_guardrail is True

    @pytest.mark.parametrize("deadline,expected_n", [(10, 50), (30, 150), (60, 300)])
    def test_rule_scales_ensemble_to_deadline(self, state, deadline, expected_n):
        state["request"] = OperatorRequest(deadline_min=deadline)
        out = make_observation_agent(ObsRule())(state)
        assert out["plan"].ensemble_size == expected_n

    def test_rule_uses_lagged_weather_for_wind_shift(self, state):
        """wind_shift compares weather_prev to weather_last, because the current
        window has not been fetched yet when this node runs."""
        state["cache"] = {"weather_prev": CALM, "weather_last": SHIFT}
        out = make_observation_agent(ObsRule())(state)
        assert out["signals"].wind_shift is True


class TestGoodObsTimeParsing:
    """steps.update stores good_obs as model_dump(mode="json") dicts, so obs_time
    arrives as an ISO string. Reading it with getattr returns None, which silently
    zeroes dt_minutes and makes growth_plausible pass unconditionally. Pin both
    representations."""

    def test_reads_obs_time_from_a_json_dict(self):
        from orchestrator.agents.observation import _obs_time_of
        entry = make_event(0).model_dump(mode="json")
        assert isinstance(entry["obs_time"], str)          # the shape update writes
        assert _obs_time_of(entry) == make_event(0).obs_time

    def test_reads_obs_time_from_an_event_object(self):
        from orchestrator.agents.observation import _obs_time_of
        assert _obs_time_of(make_event(0)) == make_event(0).obs_time

    def test_unreadable_entry_gives_none_rather_than_raising(self):
        from orchestrator.agents.observation import _obs_time_of
        assert _obs_time_of({"obs_time": "not-a-date"}) is None
        assert _obs_time_of({}) is None

    def test_dt_reaches_growth_plausible_through_compute_signals(self, state, tmp_path):
        """End to end: a shrinking perimeter 3 h after the last accepted one must be
        caught. If dt_minutes were zeroed, growth_plausible would return True."""
        from orchestrator.agents.observation import compute_signals
        import geopandas as gpd
        from shapely.geometry import Point

        big = tmp_path / "big.geojson"
        small = tmp_path / "small.geojson"
        gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(3000)],
                         crs="EPSG:5070").to_file(big, driver="GeoJSON")
        gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(500)],
                         crs="EPSG:5070").to_file(small, driver="GeoJSON")

        prev_event = make_event(0)
        state["prev_obs_path"] = str(big)
        state["current"] = make_event(1, path=str(small), feature_id="0")
        state["cache"] = {"good_obs": [prev_event.model_dump(mode="json")],
                          "weather_last": CALM}

        sig = compute_signals(state)
        # the new perimeter is far smaller, so it cannot contain the previous one
        assert sig.contains_previous is False
        assert sig.independent_check_failed is True


# ---------------------------------------------------------------------------
# Weather agent
# ---------------------------------------------------------------------------

class TestWeatherAgent:

    def test_window_defaults_to_one_hour_on_first_cycle(self, state):
        start, end = forecast_window(state)
        assert end == state["current"].obs_time
        assert end - start == timedelta(hours=1)

    def test_window_spans_from_last_accepted_observation(self, state):
        state["cache"] = {"good_obs": [make_event(0).model_dump(mode="json")]}
        start, end = forecast_window(state)
        assert start == make_event(0).obs_time
        assert end == state["current"].obs_time

    def test_fixed_is_always_normal_but_records_the_shift(self, state):
        state["cache"] = {"weather_last": CALM}
        out = make_weather_agent(WxFixed(), fetch=lambda s, e: SHIFT)(state)
        v = out["weather_verdict"]
        assert v.action == "normal"      # the control condition does not act on it
        assert v.wind_shift is True      # but it is recorded for the metrics

    def test_rule_flags_wind_shift(self, state):
        state["cache"] = {"weather_last": CALM}
        out = make_weather_agent(WxRule(), fetch=lambda s, e: SHIFT)(state)
        assert out["weather_verdict"].action == "wind_shift_detected"

    def test_rule_reports_normal_when_steady(self, state):
        state["cache"] = {"weather_last": CALM}
        out = make_weather_agent(WxRule(), fetch=lambda s, e: CALM)(state)
        assert out["weather_verdict"].action == "normal"

    def test_rule_flags_fallback_weather(self, state):
        out = make_weather_agent(WxRule(), fetch=lambda s, e: FALLBACK)(state)
        v = out["weather_verdict"]
        assert v.action == "fallback_weather"
        assert v.used_fallback is True

    def test_cache_rotates_so_the_next_cycle_can_compare(self, state):
        state["cache"] = {"weather_last": CALM}
        out = make_weather_agent(WxRule(), fetch=lambda s, e: SHIFT)(state)
        assert out["cache"]["weather_prev"] == CALM
        assert out["cache"]["weather_last"] == SHIFT

    def test_no_weather_prev_on_the_first_cycle(self, state):
        out = make_weather_agent(WxRule(), fetch=lambda s, e: CALM)(state)
        assert "weather_prev" not in out["cache"]
        assert out["cache"]["weather_last"] == CALM

    def test_guardrail_on_fetch_failure(self, state):
        def boom(start, end):
            raise RuntimeError("firemap timeout")

        out = make_weather_agent(WxFixed(), fetch=boom)(state)
        v = out["weather_verdict"]
        assert v.action == "fallback_weather"
        assert v.used_fallback is True
        assert out["log"][0].overridden_by_guardrail is True
        # the replay continues on fallback values rather than dying
        assert out["cache"]["weather_last"]["wind_speeds"] == list(FALLBACK_SPEEDS)

    def test_overwrites_lagged_wind_shift_in_signals(self, state):
        """The observation agent's wind_shift is one cycle behind; the weather agent
        replaces it with the value for the window being forecast."""
        state["signals"] = signals(wind_shift=False)
        state["cache"] = {"weather_last": CALM}
        out = make_weather_agent(WxRule(), fetch=lambda s, e: SHIFT)(state)
        assert out["signals"].wind_shift is True

    def test_refines_resolution_in_high_wind(self, state):
        from orchestrator.state import ForecastPlan
        state["plan"] = ForecastPlan(ensemble_size=300, resolution_m=150, workers=12)
        state["cache"] = {"weather_last": HIGH_SAME_DIR}   # high but steady: no shift
        out = make_weather_agent(WxRule(), fetch=lambda s, e: HIGH_SAME_DIR)(state)
        assert out["plan"].resolution_m == 75

    def test_no_refinement_when_the_deadline_is_tight(self, state):
        from orchestrator.state import ForecastPlan
        state["plan"] = ForecastPlan(ensemble_size=300, resolution_m=150, workers=12)
        state["request"] = OperatorRequest(deadline_min=20)
        state["cache"] = {"weather_last": HIGH_SAME_DIR}
        out = make_weather_agent(WxRule(), fetch=lambda s, e: HIGH_SAME_DIR)(state)
        assert "plan" not in out        # the existing plan stands


# ---------------------------------------------------------------------------
# Assimilation agent
# ---------------------------------------------------------------------------

class TestAssimilationAgent:

    def test_fixed_reproduces_the_baseline(self, state):
        state["signals"] = signals()
        out = make_assimilation_agent(AsFixed(), load_ensemble=NO_ENSEMBLE)(state)
        v = out["assim_verdict"]
        assert v.action == "assimilate"
        assert v.vsize_m == BASELINE["vsize_m"]
        assert v.wsize_m == BASELINE["wsize_m"]
        assert v.inflation == 1.0

    def test_rule_plain_update_when_everything_agrees(self, state):
        state["signals"] = signals()
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(0.8))
        assert node(state)["assim_verdict"].action == "assimilate"

    def test_rule_skips_when_an_independent_check_fails(self, state):
        state["signals"] = signals(contains_previous=False)
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(0.8))
        v = node(state)["assim_verdict"]
        assert v.action == "skip"
        assert "contain" in v.reason

    def test_wind_shift_excuses_a_failed_independent_check(self, state):
        """A big jump right after a wind shift is believable, so inflate rather
        than skip."""
        state["signals"] = signals(contains_previous=False, wind_shift=True)
        state["weather_verdict"] = WeatherVerdict(action="wind_shift_detected",
                                                  wind_shift=True)
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(5.0))
        v = node(state)["assim_verdict"]
        assert v.action == "inflate_and_assimilate"
        assert v.inflation > 1.0

    def test_rule_inflates_on_a_surprising_but_clean_observation(self, state):
        state["signals"] = signals()
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(5.0))
        assert node(state)["assim_verdict"].action == "inflate_and_assimilate"

    def test_rule_reinitialises_after_a_drift_streak(self, state):
        state["signals"] = signals(consecutive_disagreements=3)
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(5.0))
        v = node(state)["assim_verdict"]
        assert v.action == "reinitialize_from_obs"
        assert "drifted" in v.reason

    def test_drift_streak_outranks_inflation(self, state):
        """Both conditions hold; reinitialise wins because the model is the problem."""
        state["signals"] = signals(consecutive_disagreements=5)
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(9.0))
        assert node(state)["assim_verdict"].action == "reinitialize_from_obs"

    def test_guardrail_refuses_to_skip_without_an_established_state(self, state):
        state["prev_obs_path"] = None
        state["signals"] = signals(contains_previous=False)
        node = make_assimilation_agent(AsRule(), load_ensemble=ensemble_with_innovation(0.8))
        out = node(state)
        v = out["assim_verdict"]
        assert v.action == "assimilate"
        assert v.vsize_m >= VSIZE_PRESETS["synthetic_perimeter"]   # widened instead
        assert out["log"][0].overridden_by_guardrail is True

    @pytest.mark.parametrize("source,expected", [
        ("mapped_perimeter", 200),
        ("synthetic_perimeter", 500),
        ("firms_viirs", 750),
        ("firms_modis", 1000),
    ])
    def test_vsize_follows_the_source(self, state, source, expected):
        state["signals"] = signals(source=source)
        node = make_assimilation_agent(AsRule(),
                                       load_ensemble=ensemble_with_innovation(0.8, source))
        assert node(state)["assim_verdict"].vsize_m == expected

    def test_missing_innovation_is_not_treated_as_a_low_one(self, state):
        """A high value left in signals from an earlier cycle must not be read as
        this cycle's innovation when no ensemble was available."""
        state["signals"] = signals(normalized_innovation=9.9)
        out = make_assimilation_agent(AsRule(), load_ensemble=NO_ENSEMBLE)(state)
        assert out["assim_verdict"].action == "assimilate"
        assert "unavailable" in out["assim_verdict"].reason
        assert out["cache"]["innovation_available"] is False

    def test_drift_history_accumulates_only_real_innovations(self, state):
        state["signals"] = signals()
        with_ens = make_assimilation_agent(AsRule(),
                                           load_ensemble=ensemble_with_innovation(3.0))(state)
        assert len(with_ens["drift_history"]) == 1

        without = make_assimilation_agent(AsRule(), load_ensemble=NO_ENSEMBLE)(state)
        assert without["drift_history"] == []

    def test_drift_history_is_bounded(self, state):
        from orchestrator.agents.assimilation import DRIFT_HISTORY_LEN
        state["signals"] = signals()
        state["drift_history"] = [1.0] * (DRIFT_HISTORY_LEN + 4)
        out = make_assimilation_agent(AsRule(),
                                      load_ensemble=ensemble_with_innovation(2.0))(state)
        assert len(out["drift_history"]) == DRIFT_HISTORY_LEN

    def test_builds_its_own_signals_if_none_were_provided(self, state):
        """A placeholder observation node does not write signals; the assimilation
        agent must still work."""
        state["signals"] = None
        out = make_assimilation_agent(AsRule(), load_ensemble=NO_ENSEMBLE)(state)
        assert isinstance(out["signals"], Signals)
        assert out["assim_verdict"] is not None


# ---------------------------------------------------------------------------
# All three together, through the graph
# ---------------------------------------------------------------------------

class TestFullPipeline:

    @staticmethod
    def _run(obs_policy, as_policy, weather_seq, innovations):
        events = [make_event(i) for i in range(len(weather_seq))]
        seq = list(weather_seq)

        def fetch(start, end):
            return seq.pop(0) if seq else weather_seq[-1]

        def load(st):
            target = innovations[min(st["step"] - 1, len(innovations) - 1)]
            return ensemble_with_innovation(target, seed=st["step"])(st)

        nodes = placeholder_nodes()
        nodes["observation"] = make_observation_agent(obs_policy)
        nodes["weather"] = make_weather_agent(WxRule(), fetch=fetch)
        nodes["assimilation"] = make_assimilation_agent(as_policy, load_ensemble=load)
        return run_pipeline(
            build_pipeline(nodes),
            initial_state(events, OperatorRequest(deadline_min=60), "out.npz"),
        )

    WEATHER = [CALM, CALM, SHIFT, SHIFT]
    INNOVATIONS = [0.6, 0.9, 4.5, 1.1]

    def test_fixed_policies_always_do_a_plain_update(self):
        final = self._run(ObsFixed(), AsFixed(), self.WEATHER, self.INNOVATIONS)
        assert len(final["metrics"]) == 4
        assert {m["assim_action"] for m in final["metrics"]} == {"assimilate"}

    def test_rule_policies_inflate_on_the_wind_shift_cycle(self):
        final = self._run(ObsRule(), AsRule(), self.WEATHER, self.INNOVATIONS)
        actions = [m["assim_action"] for m in final["metrics"]]
        assert actions == ["assimilate", "assimilate",
                           "inflate_and_assimilate", "assimilate"]

    def test_the_two_configurations_differ_on_identical_inputs(self):
        """The comparison the experiment rests on: same observations, same weather,
        same innovations; only the policies change."""
        fixed = self._run(ObsFixed(), AsFixed(), self.WEATHER, self.INNOVATIONS)
        rule = self._run(ObsRule(), AsRule(), self.WEATHER, self.INNOVATIONS)
        assert ([m["assim_action"] for m in fixed["metrics"]]
                != [m["assim_action"] for m in rule["metrics"]])

    def test_weather_cache_is_populated_for_the_whole_run(self):
        final = self._run(ObsRule(), AsRule(), self.WEATHER, self.INNOVATIONS)
        assert "weather_last" in final["cache"]
        assert "weather_prev" in final["cache"]

    def test_every_cycle_logs_all_three_agents(self):
        final = self._run(ObsRule(), AsRule(), self.WEATHER, self.INNOVATIONS)
        for node in ("observation", "weather", "assimilation"):
            assert sum(r.node == node for r in final["log"]) == 4