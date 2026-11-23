"""
§9 replay.py — run one Border 2 experiment arm end to end.

    from orchestrator.replay import replay
    result = replay(policies="fixed", scenario="clean")

Each call builds the observation events, seeds the initial state, wires the nodes,
runs the graph, scores the result and writes everything to its own directory.

Output layout — kept entirely separate from the baseline, which stays untouched in
`outputs/Border_2_Real/`:

    outputs/Border_2_Orchestrator/<policies>_<scenario>_<run_id>/
        config.json      exactly what was run (policies, scenario, N, resolution, ...)
        steps.csv        same schema as the baseline's, so the two concatenate
        members.csv      same schema as the baseline's
        decisions.csv    one row per agent decision
        metrics.json     per-cycle accuracy + behaviour summary
        state.npz        final analysis and covariance
        ensembles/       per-step forecast ensembles
        synthetic/       synthetic perimeters, when an observation was rejected

The baseline's own numbers are never written to or read from, so an arm can be
re-run as often as needed without putting `outputs/Border_2_Real/` at risk.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from orchestrator import metrics, steps
from orchestrator.graph import build_pipeline, run_pipeline
from orchestrator.state import (
    BASELINE,
    ObservationEvent,
    OperatorRequest,
    initial_state,
)

# ---------------------------------------------------------------------------
# Border 2 scenario definition — matches border2_experiment.ipynb exactly
# ---------------------------------------------------------------------------

FIRE_NAME = "Border_2"
PERIMETER_FILE = "data/border2_inputs/border_2_perimeters.geojson"
HOTSPOT_FILE = "data/border2_inputs/firms_hotspots.csv"

# The notebook's SELECTED_IDX. Features 0, 1 and 2 of the GeoJSON are byte-identical
# (IoU 1.0), so only one of them is usable; the notebook takes index 1.
SELECTED_IDX = [1, 3, 4, 5]

# The notebook's OBS_TIMESTAMPS, anchored to CAL FIRE incident updates and a FIRMS
# ignition detection. Index 0 is the initial condition, not an observation.
OBS_TIMESTAMPS = [
    datetime(2025, 1, 23, 18, 22, tzinfo=timezone.utc),   # t0  FIRMS ignition anchor
    datetime(2025, 1, 24, 20, 36, tzinfo=timezone.utc),   # t1  CAL FIRE  4,250 ac
    datetime(2025, 1, 25,  3, 49, tzinfo=timezone.utc),   # t2  CAL FIRE  6,273 ac
    datetime(2025, 1, 27,  0, 12, tzinfo=timezone.utc),   # t3  CAL FIRE  6,625 ac
]

# Weather query location (fire centroid) and FARSITE params, from the notebook.
WX_LAT, WX_LON = 32.59588662537945, -116.85708741551528
FARSITE_PARAMS = {"temperature": 66, "humidity": 0.08}

N_VERTEX = 20
INITIAL_COV_SCALE = 1000.0      # the notebook's X = 1000 * I

OUTPUT_ROOT = Path("outputs") / "Border_2_Orchestrator"

Scenario = Literal["clean", "shifted_obs", "stale_obs", "mislabeled_obs",
                   "wind_shift", "weather_failure"]
Policies = Literal["fixed", "rule", "llm"]

# Which cycle carries the fault. Cycle 2 of 3, so there is an established state
# before it (the first-observation guardrail would otherwise force acceptance) and
# a cycle after it to show recovery.
FAULT_CYCLE = 2


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _write_faulted_perimeter(out_path: Path, perimeter_file: str,
                             feature_id: int, fault: str,
                             substitute_feature: int | None = None) -> None:
    """Write a single faulted perimeter to its own GeoJSON.

    The fault has to be in the geometry, not just in a label: `contains_previous`,
    `growth_plausible` and `agrees_with_firms` all read the polygon, so an event
    merely tagged "shifted_2km" is indistinguishable from a clean one and the
    scenario tests nothing. The source file is never modified — the faulted copy
    lands in the run directory, so it can be inspected afterwards.
    """
    import geopandas as gpd
    from shapely.affinity import translate

    gdf = gpd.read_file(perimeter_file).to_crs(epsg=5070)

    if fault == "shifted_2km":
        # Right size, wrong place: what contains_previous exists to catch.
        geom = translate(gdf.geometry.iloc[feature_id], xoff=2000.0, yoff=0.0)
    elif fault == "mislabeled":
        # A perimeter from a different time, presented as this one: the fire
        # appears not to have grown at all.
        if substitute_feature is None:
            raise ValueError("mislabeled needs substitute_feature")
        geom = gdf.geometry.iloc[substitute_feature]
    else:
        raise ValueError(f"unknown geometry fault {fault!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:5070").to_file(
        out_path, driver="GeoJSON")


def build_events(scenario: Scenario = "clean",
                 perimeter_file: str = PERIMETER_FILE,
                 latency_min: float = 5.0,
                 inject_dir: str | Path | None = None,
                 ) -> tuple[ObservationEvent, list[ObservationEvent]]:
    """Build the initial condition and the observation events for a scenario.

    Returns (seed_event, events). `seed_event` is the perimeter the state starts
    from — the notebook's `perimeters_gdf.iloc[0]`, which it never assimilates.
    `events` are the remaining perimeters, arriving `latency_min` after observation.

      clean            the observations as recorded
      shifted_obs      cycle 2's perimeter translated 2 km east (geometry really moved)
      stale_obs        cycle 2's observation arrives 10 h late
      mislabeled_obs   cycle 2 carries an earlier perimeter, so the fire looks static
      wind_shift       unchanged; the label marks the Santa Ana interval already in the data
      weather_failure  unchanged; replay() breaks the weather service instead

    Geometry faults are written as separate GeoJSON files under `inject_dir`
    (replay passes the run directory), so the source file is untouched and the
    exact faulted perimeter is kept with the run. Asking for a geometry fault
    without `inject_dir` raises rather than silently degrading to a label.
    """
    from datetime import timedelta

    geometry_fault = {"shifted_obs": "shifted_2km",
                      "mislabeled_obs": "mislabeled"}.get(scenario)
    if geometry_fault and inject_dir is None:
        raise ValueError(
            f"scenario {scenario!r} alters the perimeter geometry, so it needs "
            "inject_dir to write the faulted copy to. Without it the fault would "
            "be a label only and the scenario would behave exactly like 'clean'.")

    def mk(idx_in_file: int, obs_time: datetime, injected: str | None = None,
           extra_latency_min: float = 0.0,
           path: str | None = None, feature_id: str | None = None
           ) -> ObservationEvent:
        return ObservationEvent(
            obs_time=obs_time,
            arrival_time=obs_time + timedelta(minutes=latency_min + extra_latency_min),
            source="mapped_perimeter",
            path=path or perimeter_file,
            feature_id=feature_id if feature_id is not None else str(idx_in_file),
            injected=injected,
        )

    seed = mk(SELECTED_IDX[0], OBS_TIMESTAMPS[0])

    events: list[ObservationEvent] = []
    for i, (idx, t) in enumerate(zip(SELECTED_IDX[1:], OBS_TIMESTAMPS[1:]), start=1):
        if i != FAULT_CYCLE or scenario in ("clean", "wind_shift", "weather_failure"):
            events.append(mk(idx, t))
            continue

        if scenario == "stale_obs":
            # 10 h late, past the observation agent's 360 min threshold
            events.append(mk(idx, t, injected="stale", extra_latency_min=600.0))

        elif geometry_fault:
            faulted = Path(inject_dir) / f"{scenario}_cycle{i}.geojson"
            _write_faulted_perimeter(
                faulted, perimeter_file, idx, geometry_fault,
                substitute_feature=SELECTED_IDX[0] if geometry_fault == "mislabeled"
                else None)
            events.append(mk(idx, t, injected=geometry_fault,
                             path=str(faulted), feature_id="0"))
        else:
            events.append(mk(idx, t))

    return seed, events


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

def make_nodes(policies: Policies, run_dir: Path, lcp_path: str | Path,
               members_csv: Path, run_id: str,
               hotspot_path: str | None = HOTSPOT_FILE,
               weather_fetch=None) -> dict:
    """Wire a full nodes dict for the given policy family.

    'fixed' reproduces the baseline: accept everything, baseline FARSITE settings,
    a plain EnKF update at vsize=300/wsize=100. 'rule' uses the rule-based policies.
    An LLM family slots in here as a third branch without touching anything else.
    """
    from orchestrator.agents import assimilation as A
    from orchestrator.agents import observation as O
    from orchestrator.agents import weather as W

    if policies == "fixed":
        obs_p, wx_p, as_p = O.FixedPolicy(), W.FixedPolicy(), A.FixedPolicy()
    elif policies == "rule":
        obs_p, wx_p, as_p = O.RulePolicy(), W.RulePolicy(), A.RulePolicy()
    elif policies == "llm":
        # LLM assimilation agent; observation and weather stay rule-based until their
        # LLM policies exist. The model is fixed for every run (orchestrator/llm.py).
        from orchestrator.llm import get_llm
        obs_p, wx_p, as_p = O.RulePolicy(), W.RulePolicy(), A.LLMPolicy(get_llm())
    else:
        raise ValueError(f"unknown policies {policies!r}; expected 'fixed', 'rule' or 'llm'")

    nodes = {
        "receive": steps.receive,
        "synthesize": steps.synthesize,
        "update": steps.update,
        "observation": O.make_observation_agent(obs_p, hotspot_path=hotspot_path),
        "weather": W.make_weather_agent(wx_p, fetch=weather_fetch,
                                        lat=WX_LAT, lon=WX_LON),
        "assimilation": A.make_assimilation_agent(as_p),
        "forecast": steps.make_forecast(
            lcp_path=lcp_path, n_vertex=N_VERTEX, params=FARSITE_PARAMS,
            members_csv=members_csv, fire_name=FIRE_NAME, run_id=run_id),
    }
    return nodes


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    run_id: str
    policies: str
    scenario: str
    out_dir: Path
    final_state: dict
    scored: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        acc, beh = self.scored["accuracy"], self.scored["behaviour"]
        lines = [f"{self.policies}/{self.scenario}  run {self.run_id}",
                 f"  cycles              {len(self.scored['cycles'])}"]
        if acc.get("iou_mean") is not None:
            lines.append(f"  mean IoU            {acc['iou_mean']:.3f}")
            lines.append(f"  final IoU           {acc['iou_final']:.3f}")
            lines.append(f"  mean |area error|   {acc['abs_area_error_frac_mean']:.1%}")
        lines.append(f"  guardrail overrides {beh['guardrail_overrides']}")
        for node, counts in beh["decisions_per_node"].items():
            lines.append(f"  {node:13s} {counts}")
        lines.append(f"  output              {self.out_dir}")
        return "\n".join(lines)


def replay(policies: Policies = "fixed",
           scenario: Scenario = "clean",
           ensemble_size: int | None = None,
           resolution_m: int | None = None,
           workers: int | None = None,
           deadline_min: float = 60.0,
           lcp_path: str | Path | None = None,
           perimeter_file: str = PERIMETER_FILE,
           hotspot_path: str | None = HOTSPOT_FILE,
           output_root: str | Path = OUTPUT_ROOT,
           run_id: str | None = None,
           seed: int = 42,
           n_vertex: int = N_VERTEX) -> ReplayResult:
    """Run one experiment arm and write its results to its own directory.

    Args:
        policies: 'fixed' (baseline behaviour) or 'rule'
        scenario: see build_events
        ensemble_size, resolution_m, workers: override the plan the agents choose.
            Use small values for a smoke run (ensemble_size=4, workers=4,
            resolution_m=300) before committing to a full one.
        deadline_min: the operator budget the rule policies size the plan against
        lcp_path: FARSITE landscape; defaults to adapter.LCP_PATH
        output_root: parent of this run's directory. Defaults to
            outputs/Border_2_Orchestrator, never the baseline's directory.
        seed: RNG seed saved into the initial state, so a run is reproducible

    Returns:
        ReplayResult. Print `.summary()` for the short version.
    """
    from orchestrator import adapter

    run_id = run_id or str(uuid.uuid4())[:8]
    out_dir = Path(output_root) / f"{policies}_{scenario}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if lcp_path is None:
        lcp_path = adapter.LCP_PATH
    members_csv = out_dir / "members.csv"
    steps_csv = out_dir / "steps.csv"
    state_path = out_dir / "state.npz"

    # --- events + initial condition -----------------------------------------
    # Geometry faults are written under the run directory, so the source GeoJSON
    # stays untouched and the exact faulted perimeter is kept with the results.
    seed_event, events = build_events(scenario, perimeter_file,
                                      inject_dir=out_dir / "injected")
    steps.seed_state(state_path, seed_event, n_vertex=n_vertex,
                     cov_scale=INITIAL_COV_SCALE, seed=seed)

    # --- a broken weather service, for the weather_failure scenario ----------
    weather_fetch = None
    if scenario == "weather_failure":
        def weather_fetch(start, end):          # noqa: ANN001
            raise ConnectionError("weather_failure scenario: service unavailable")

    nodes = make_nodes(policies, out_dir, lcp_path, members_csv, run_id,
                       hotspot_path=hotspot_path, weather_fetch=weather_fetch)

    # --- plan overrides, for smoke runs -------------------------------------
    # The observation agent picks the plan; wrap it so an explicit override wins
    # without having to special-case every policy.
    if any(v is not None for v in (ensemble_size, resolution_m, workers)):
        inner = nodes["observation"]

        def observation_with_override(state):
            out = inner(state)
            plan = out.get("plan")
            if plan is not None:
                out["plan"] = plan.model_copy(update={
                    k: v for k, v in
                    (("ensemble_size", ensemble_size), ("resolution_m", resolution_m),
                     ("workers", workers)) if v is not None})
            return out

        nodes["observation"] = observation_with_override

    # --- run ----------------------------------------------------------------
    st = initial_state(events, OperatorRequest(deadline_min=deadline_min),
                       str(state_path))
    st["cache"] = {"run_dir": str(out_dir), "run_id": run_id,
                   "hotspot_path": hotspot_path}

    # Ground truth for scoring: always the clean perimeter for that cycle, read
    # straight from the source file. Accuracy must NOT be measured against the
    # observation the filter was given, because in a fault scenario that is the
    # faulted or synthetic perimeter — scoring against it rewards an arm for
    # faithfully reproducing the fault. With a mislabeled observation that inverted
    # the result: the arm that swallowed the bad perimeter scored better than the
    # arm that rejected it.
    truth_states: dict[int, Any] = _truth_perimeters(perimeter_file, n_vertex)

    assimilated_states: dict[int, Any] = {}   # what the filter actually saw
    analysis_states: dict[int, Any] = {}

    # update() saves the analysis to state_path and the forecast saved the aligned
    # observation, so read both from disk after each cycle rather than threading
    # arrays through TwinState.
    real_update = nodes["update"]

    def update_and_capture(state):
        out = real_update(state)
        step = state.get("step")
        ens_path = state.get("ensemble_path")
        try:
            if ens_path and Path(ens_path).exists():
                data = np.load(ens_path, allow_pickle=True)
                if "aligned_obs" in data:
                    assimilated_states[step] = data["aligned_obs"]
            vec, _, _ = adapter.load_state(state["state_path"])
            analysis_states[step] = vec
        except Exception:
            pass
        return out

    nodes["update"] = update_and_capture

    final = run_pipeline(build_pipeline(nodes), st)

    # --- steps.csv, in the baseline's schema so the two concatenate ----------
    _write_steps_csv(steps_csv, run_id, final, ensemble_size, resolution_m)

    # --- score + write ------------------------------------------------------
    # Accuracy against ground truth, not against what was assimilated.
    scored = metrics.score_run(final, truth_states, analysis_states)
    # Record how far the assimilated observation was from the truth, so a fault
    # scenario shows the size of the fault alongside its effect.
    for c in scored["cycles"]:
        step = c.get("step")
        if step in assimilated_states and step in truth_states:
            fault = metrics.score_perimeter(assimilated_states[step], truth_states[step])
            c["assimilated_vs_truth_iou"] = fault["iou"]
            c["assimilated_vs_truth_centroid_m"] = fault["centroid_offset_m"]
    metrics.write_run_metrics(out_dir, scored, final)

    config = {
        "run_id": run_id, "policies": policies, "scenario": scenario,
        "fire": FIRE_NAME, "n_vertex": n_vertex, "seed": seed,
        "deadline_min": deadline_min,
        "ensemble_size_override": ensemble_size,
        "resolution_m_override": resolution_m, "workers_override": workers,
        "baseline_reference": dict(BASELINE),
        "perimeter_file": perimeter_file, "hotspot_path": hotspot_path,
        "lcp_path": str(lcp_path),
        "selected_idx": SELECTED_IDX,
        "obs_timestamps": [t.isoformat() for t in OBS_TIMESTAMPS],
        "farsite_params": FARSITE_PARAMS,
        "wx_lat": WX_LAT, "wx_lon": WX_LON,
    }
    if policies == "llm":
        from orchestrator.llm import get_llm, llm_config
        config["llm"] = llm_config(get_llm())     # model alias + date: NRP can repoint aliases
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    return ReplayResult(run_id=run_id, policies=policies, scenario=scenario,
                        out_dir=out_dir, final_state=final, scored=scored,
                        config=config)


def _truth_perimeters(perimeter_file: str, n_vertex: int) -> dict[int, Any]:
    """{cycle: true perimeter state vector} from the unfaulted source file.

    Cycle numbering matches TwinState["step"], which is 1-based, so cycle 1 is
    SELECTED_IDX[1] — the first perimeter actually assimilated.
    """
    from orchestrator import adapter

    truth: dict[int, Any] = {}
    for cycle, (idx, t) in enumerate(zip(SELECTED_IDX[1:], OBS_TIMESTAMPS[1:]), start=1):
        try:
            truth[cycle] = adapter.load_observation(
                ObservationEvent(obs_time=t, arrival_time=t,
                                 source="mapped_perimeter", path=perimeter_file,
                                 feature_id=str(idx)),
                n_vertex=n_vertex)
        except Exception:
            pass
    return truth


def _write_steps_csv(path: Path, run_id: str, final: dict,
                     n_override: int | None, res_override: int | None) -> None:
    """Write steps.csv with the baseline's columns, so an arm and the baseline can
    be concatenated for comparison (outputs/Border_2_Real/steps.csv)."""
    import csv

    header = ["run_id", "step", "t_obs", "N", "n_valid",
              "t_ensemble_s", "t_enkf_s", "vsize", "wsize"]
    # Pull per-step ensemble facts out of the forecast log records.
    forecast_by_step: dict[int, dict] = {}
    for i, rec in enumerate([r for r in final.get("log", []) if r.node == "forecast"]):
        forecast_by_step[i + 1] = rec.decision | rec.inputs

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for m in final.get("metrics", []):
            step = m.get("step")
            fc = forecast_by_step.get(step, {})
            w.writerow({
                "run_id": run_id,
                "step": (step - 1) if isinstance(step, int) else step,  # 0-indexed, as baseline
                "t_obs": m.get("obs_time"),
                "N": fc.get("N", n_override),
                "n_valid": fc.get("n_valid"),
                "t_ensemble_s": fc.get("ensemble_wall_s"),
                "t_enkf_s": "",
                "vsize": m.get("vsize_m"),
                "wsize": m.get("wsize_m"),
            })


# ---------------------------------------------------------------------------
# Comparing arms
# ---------------------------------------------------------------------------

def compare(results: list[ReplayResult]):
    """Results table across arms. Pass the ReplayResults from several replay() calls."""
    return metrics.compare_runs(
        {f"{r.policies}/{r.scenario}": r.scored for r in results})


def load_run(out_dir: str | Path) -> dict[str, Any]:
    """Load a finished run's metrics.json and config.json from disk."""
    out_dir = Path(out_dir)
    return {"config": json.loads((out_dir / "config.json").read_text()),
            "scored": metrics.load_metrics(out_dir / "metrics.json")}
