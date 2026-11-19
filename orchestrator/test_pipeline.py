"""Tests for the pipeline organizer (graph.py, routing.py, steps.py) and synthetic.py.

No FARSITE, no LLM: agents are placeholders. Run with `pytest tests/test_pipeline.py -v`.
"""
from datetime import datetime, timedelta

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from orchestrator import routing, steps
from orchestrator.graph import build_pipeline, memory_checkpointer, run_pipeline, PIPELINE
from orchestrator.placeholders import PlaceholderVerdict, placeholder_nodes
from orchestrator.state import ObservationEvent, OperatorRequest, initial_state
from orchestrator.synthetic import (equivalent_radius, grow_perimeter, length_to_breadth,
                                    observed_spread_rate, read_perimeter)

T0 = datetime(2025, 1, 23, 12, 0)
CYCLE = ["receive", "observation", "weather", "forecast", "assimilation", "update"]
REJECT_CYCLE = ["receive", "observation", "synthesize", "weather", "forecast", "assimilation", "update"]
WEATHER = {"wind_speeds": [15.0, 17.0, 14.0], "wind_directions": [0.0, 10.0, 350.0], "used_fallback": False}


# ---------------------------------------------------------------------------
# fixtures: a GeoJSON holding several growing perimeters (one feature per row, like Border 2)
# ---------------------------------------------------------------------------
@pytest.fixture
def perimeters(tmp_path):
    centre = gpd.GeoSeries([Point(-116.85, 32.60)], crs=4326).to_crs(5070).iloc[0]
    radii = [500, 900, 1300, 1700, 2100, 2500, 2900, 3300, 3700, 4100]
    path = tmp_path / "perims.geojson"
    gpd.GeoDataFrame({"i": range(len(radii))}, geometry=[centre.buffer(r) for r in radii],
                     crs=5070).to_crs(4326).to_file(path, driver="GeoJSON")
    return path


def make_events(path, n, injected=None):
    """n observations, hourly, each arriving 10 min after it was made."""
    injected = injected or {}
    return [ObservationEvent(obs_time=T0 + timedelta(hours=i), arrival_time=T0 + timedelta(hours=i, minutes=10),
                             source="mapped_perimeter", path=str(path), feature_id=str(i),
                             injected=injected.get(i)) for i in range(n)]


def traced(nodes, trace):
    """Wrap every node so the order of calls is recorded."""
    def wrap(name, fn):
        def node(state):
            trace.append(name)
            return fn(state)
        return node
    return {name: wrap(name, fn) for name, fn in nodes.items()}


def run(events, tmp_path, obs_action="use", trace=None):
    nodes = placeholder_nodes(obs_action=obs_action, weather=WEATHER)
    graph = build_pipeline(traced(nodes, trace) if trace is not None else nodes)
    state = initial_state(events, OperatorRequest(deadline_min=30), state_path=str(tmp_path / "state.npz"))
    state["cache"]["run_dir"] = str(tmp_path / "run")
    return run_pipeline(graph, state)


def reject_injected(state):
    return "reject" if state["current"].injected else "use"


# ---------------------------------------------------------------------------
# organizer
# ---------------------------------------------------------------------------
def test_pipeline_constant_matches_todo():
    assert PIPELINE == CYCLE


def test_clean_run_follows_fixed_order_and_ends(perimeters, tmp_path):
    trace = []
    final = run(make_events(perimeters, 3), tmp_path, trace=trace)
    assert trace == CYCLE * 3
    assert final["pending"] == []
    assert len(final["metrics"]) == 3
    assert [m["step"] for m in final["metrics"]] == [1, 2, 3]
    assert not any(m["synthetic"] for m in final["metrics"])


def test_rejected_observation_is_replaced_and_forecast_still_runs(perimeters, tmp_path):
    trace = []
    events = make_events(perimeters, 4, injected={2: "shifted_2km"})
    final = run(events, tmp_path, obs_action=reject_injected, trace=trace)

    assert trace == CYCLE * 2 + REJECT_CYCLE + CYCLE           # cycle 3 goes through synthesize
    m = final["metrics"][2]
    assert m["obs_action"] == "reject" and m["synthetic"] and m["source"] == "synthetic_perimeter"
    assert m["assim_action"] == "assimilate"                    # assimilation still ran

    syn = [r for r in final["log"] if r.node == "synthesize"]
    assert len(syn) == 1 and syn[0].decision["synthetic"] is not None
    assert syn[0].decision["rate_source"].startswith("observed")   # two earlier accepted perimeters
    # the rejected observation never becomes a "good" one
    good_ids = [e["feature_id"] for e in final["cache"]["good_obs"]]
    assert "2" not in good_ids and good_ids == ["1", "3"]


def test_synthetic_perimeter_grows_from_last_good_and_is_readable(perimeters, tmp_path):
    events = make_events(perimeters, 3, injected={2: "stale"})
    final = run(events, tmp_path, obs_action=reject_injected)
    rec = next(r for r in final["log"] if r.node == "synthesize")
    syn_event = ObservationEvent.model_validate(rec.decision["synthetic"])
    grown, last_good = read_perimeter(syn_event), read_perimeter(events[1])
    assert grown.buffer(1).contains(last_good)                  # fires don't shrink
    assert grown.area > last_good.area
    assert syn_event.obs_time == events[2].obs_time and syn_event.feature_id == "0"


def test_first_observation_rejected_has_nothing_to_grow(perimeters, tmp_path):
    trace = []
    events = make_events(perimeters, 2, injected={0: "mislabeled"})
    final = run(events, tmp_path, obs_action=reject_injected, trace=trace)
    assert trace[:4] == ["receive", "observation", "synthesize", "update"]   # skips the forecast
    assert trace[4:] == CYCLE
    rec = next(r for r in final["log"] if r.node == "synthesize")
    assert rec.decision["synthetic"] is None and "no earlier" in rec.decision["reason"]


def test_long_replay_is_not_cut_off_by_recursion_limit(perimeters, tmp_path):
    final = run(make_events(perimeters, 10), tmp_path)           # 60 steps > LangGraph's default 25
    assert len(final["metrics"]) == 10 and final["pending"] == []


def test_verdicts_are_cleared_each_cycle(perimeters, tmp_path):
    state = initial_state(make_events(perimeters, 2), OperatorRequest(deadline_min=30), "s.npz")
    state["obs_verdict"] = PlaceholderVerdict(action="reject")
    out = steps.receive(state)
    assert out["obs_verdict"] is None and out["weather_verdict"] is None and out["assim_verdict"] is None


def test_build_pipeline_rejects_missing_steps():
    nodes = placeholder_nodes()
    del nodes["synthesize"]
    with pytest.raises(ValueError, match="synthesize"):
        build_pipeline(nodes)


# ---------------------------------------------------------------------------
# routing (pure functions)
# ---------------------------------------------------------------------------
def test_routing_rules():
    assert routing.after_observation({"obs_verdict": PlaceholderVerdict(action="reject")}) == "synthesize"
    assert routing.after_observation({"obs_verdict": {"action": "downweight"}}) == "weather"
    assert routing.after_observation({"obs_verdict": None}) == "weather"
    assert routing.after_update({"pending": [1]}) == "receive"
    assert routing.after_update({"pending": []}) == "end"


# ---------------------------------------------------------------------------
# synthetic perimeter growth
# ---------------------------------------------------------------------------
def test_length_to_breadth():
    assert length_to_breadth(0) == pytest.approx(1.0, abs=1e-3)
    assert 1 < length_to_breadth(10) < length_to_breadth(20) < length_to_breadth(40)


def test_grow_without_wind_is_round_and_contains_previous():
    prev = Point(0, 0).buffer(1000)
    grown = grow_perimeter(prev, dt_h=2, rate_m_per_h=100)
    assert grown.contains(prev)
    assert equivalent_radius(grown) == pytest.approx(1200, rel=0.01)
    minx, miny, maxx, maxy = grown.bounds
    assert (maxx - minx) == pytest.approx(maxy - miny, rel=0.01)


def test_grow_with_north_wind_pushes_fire_south():
    prev = Point(0, 0).buffer(500)
    grown = grow_perimeter(prev, dt_h=3, rate_m_per_h=100, wind_speed_mph=20, wind_from_deg=0)
    assert grown.contains(prev)
    assert grown.centroid.y < -100                               # moved downwind (south)
    minx, miny, maxx, maxy = grown.bounds
    assert (maxy - miny) > (maxx - minx)                         # longer than wide, north-south


def test_observed_spread_rate():
    a, b = Point(0, 0).buffer(1000), Point(0, 0).buffer(1600)
    assert observed_spread_rate(a, b, dt_h=3) == pytest.approx(200, rel=0.01)
    assert observed_spread_rate(b, a, dt_h=3) == 0.0             # shrinking → 0, never negative


def test_checkpointer_keeps_every_state_and_loads_our_types(perimeters, tmp_path, recwarn):
    events = make_events(perimeters, 3, injected={2: "shifted_2km"})
    graph = build_pipeline(placeholder_nodes(obs_action=reject_injected, weather=WEATHER),
                           checkpointer=memory_checkpointer())
    state = initial_state(events, OperatorRequest(deadline_min=30), str(tmp_path / "s.npz"))
    state["cache"]["run_dir"] = str(tmp_path)
    cfg = {"configurable": {"thread_id": "t1"}}
    final = run_pipeline(graph, state, config=cfg)
    history = list(graph.get_state_history(cfg))
    assert len(history) > 3 * len(CYCLE)
    assert isinstance(history[0].values["current"], ObservationEvent)
    assert not [w for w in recwarn if "unregistered type" in str(w.message)]
    assert len(final["metrics"]) == 3
