"""
§6 observation_agent — decides whether to trust a new perimeter observation.

Two policies are provided:
  FixedPolicy   — always accept (reproduces the baseline Border 2 run)
  RulePolicy    — accept / reject based on checks.py signals; customizable thresholds

Both are written as callables that take TwinState and return a dict with:
    {"obs_verdict": ObservationVerdict, "signals": Signals, "log": [DecisionRecord]}

Usage
-----
    from orchestrator.agents.observation import make_observation_agent, RulePolicy

    # Baseline (reproduces Border_2_Real): all observations accepted
    obs_node = make_observation_agent(FixedPolicy())

    # Rule-based filtering: reject bad observations
    obs_node = make_observation_agent(RulePolicy())

    # Pass to build_pipeline:
    nodes = make_placeholder_nodes()
    nodes["observation"] = obs_node
    graph = build_pipeline(nodes)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from shapely.geometry import Polygon

from orchestrator import checks
from orchestrator.state import (
    DecisionRecord,
    ForecastPlan,
    ObservationVerdict,
    Signals,
    Source,
    TwinState,
    BASELINE,
    VSIZE_PRESETS,
)


# ---------------------------------------------------------------------------
# Shared helper: build the Signals dataclass from raw state + checks
# ---------------------------------------------------------------------------

def _geometry_from_path(path: str | None, feature_id: str | None) -> Polygon | None:
    """Load a single polygon from a GeoJSON file for containment / growth checks.
    Returns None on any error (file missing, invalid geometry, etc.)."""
    if path is None:
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path).to_crs(epsg=5070)
        idx = int(feature_id) if feature_id is not None else 0
        geom = gdf.geometry.iloc[idx]
        from shapely import make_valid
        return checks._largest_polygon(make_valid(geom))
    except Exception:
        return None


def _obs_time_of(entry) -> datetime | None:
    """obs_time of a cache["good_obs"] entry.

    steps.update stores these as model_dump(mode="json") dicts, so obs_time arrives
    as an ISO string, but a caller (or a test) may hold ObservationEvent objects.
    Accept either, and None when it cannot be read.
    """
    raw = entry.get("obs_time") if isinstance(entry, dict) else getattr(entry, "obs_time", None)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, datetime) else None


def compute_signals(state: TwinState) -> Signals:
    """
    Run all forecast-independent checks on the current observation, plus
    wind_shift and obs_age.

    The normalized_innovation field requires the forecast ensemble (computed after
    FARSITE runs). Here it defaults to 0.0 — the assimilation agent overwrites it
    with the real value once the ensemble is available.

    Args:
        state: current TwinState (clock, current, prev_obs_path, cache, drift_history)

    Returns:
        Signals (all fields populated)
    """
    current = state["current"]
    clock   = state["clock"]

    # --- geometry for containment / growth checks ---
    cur_poly  = _geometry_from_path(
        current.path if current else None,
        current.feature_id if current else None,
    )
    # Previous perimeter: take it from cache["good_obs"][-1], the last accepted
    # ObservationEvent, which carries BOTH path and feature_id.
    # state["prev_obs_path"] is only a path — the feature id is lost, so loading from
    # it silently reads feature 0 of a multi-feature file rather than the perimeter
    # that was actually accepted. On Border 2 that meant every cycle compared against
    # the 229 ha seed perimeter instead of the 1,714 / 2,533 ha ones, which makes
    # contains_previous nearly inert. prev_obs_path remains the "is there a previous
    # observation at all" flag for the guardrail.
    prev_entry = (cache_good[-1] if (cache_good := ((state.get("cache") or {})
                                                    .get("good_obs") or [])) else None)
    if prev_entry is not None:
        prev_path = (prev_entry.get("path") if isinstance(prev_entry, dict)
                     else getattr(prev_entry, "path", None))
        prev_fid = (prev_entry.get("feature_id") if isinstance(prev_entry, dict)
                    else getattr(prev_entry, "feature_id", None))
    else:
        prev_path, prev_fid = state.get("prev_obs_path"), None
    prev_poly = _geometry_from_path(prev_path, prev_fid)

    # --- obs age ---
    if current and clock:
        obs_age_min = (clock - current.obs_time).total_seconds() / 60.0
    else:
        obs_age_min = 0.0

    # --- dt for growth plausibility ---
    cache = state.get("cache") or {}
    good_obs: list = cache.get("good_obs", [])
    dt_minutes = 0.0
    if good_obs and current:
        last_time = _obs_time_of(good_obs[-1])
        if last_time is not None:
            dt_minutes = (current.obs_time - last_time).total_seconds() / 60.0

    # --- weather for growth plausibility and wind_shift ---
    # The observation node runs BEFORE the weather node, so the current window has not
    # been fetched yet. That is fine: this observation reports growth over the interval
    # that just ENDED, and the weather of that interval is cache["weather_last"] (fetched
    # by the previous cycle's weather agent). wind_shift is therefore one cycle lagged —
    # it answers "did the wind shift during the interval that produced this perimeter?",
    # which is exactly the question that justifies a large jump. The weather agent
    # overwrites signals.wind_shift with the current-window value for downstream agents.
    weather_last = cache.get("weather_last", None)   # window that produced this observation
    weather_prev = cache.get("weather_prev", None)   # the window before that
    wind_speed_mph = float(np.mean((weather_last or {}).get("wind_speeds", [0.0])))

    # --- individual checks ---
    _contains_prev  = checks.contains_previous(cur_poly, prev_poly)
    _growth_ok      = checks.growth_plausible(cur_poly, prev_poly, dt_minutes, wind_speed_mph)
    _wind_shift     = checks.wind_shift(weather_prev, weather_last or {})

    # FIRMS: best-effort; None if data not available
    hotspot_path = cache.get("hotspot_path", None)
    hotspots = None
    if hotspot_path and current:
        hotspots = checks.load_hotspots(hotspot_path, current.obs_time, clock)
    _agrees_firms = checks.agrees_with_firms(cur_poly, hotspots)

    # consecutive disagreements (from drift_history stored by assimilation agent)
    drift_history: list[float] = state.get("drift_history") or []
    # count the tail of the drift_history where values were > 1.5 (disagreement)
    consec = 0
    for v in reversed(drift_history):
        if v > 1.5:
            consec += 1
        else:
            break

    source: Source = current.source if current else "mapped_perimeter"

    return Signals(
        normalized_innovation=0.0,          # placeholder; overwritten by assimilation agent
        agrees_with_firms=_agrees_firms,
        contains_previous=_contains_prev,
        growth_plausible=_growth_ok,
        obs_age_min=obs_age_min,
        source=source,
        wind_shift=_wind_shift,
        consecutive_disagreements=consec,
    )


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class ObservationPolicy(ABC):
    """
    A policy takes the current TwinState + pre-computed Signals and returns an
    ObservationVerdict.

    It also optionally returns a ForecastPlan (the observation agent is a good place
    to pick a plan because the plan depends on the deadline and the event source, not
    on the forecast). Return None to let a later agent (or the weather agent) set the plan.
    """

    @abstractmethod
    def decide(
        self, state: TwinState, signals: Signals
    ) -> tuple[ObservationVerdict, ForecastPlan | None]:
        ...


# ---------------------------------------------------------------------------
# FixedPolicy — always accept, baseline settings (reproduces Border_2_Real)
# ---------------------------------------------------------------------------

class FixedPolicy(ObservationPolicy):
    """Always accept every observation with the baseline FARSITE settings.

    This is the control condition: it reproduces the Border_2_Real experiment
    with no filtering or adaptive planning. Use it to establish the baseline
    metrics before comparing to RulePolicy.
    """

    def decide(
        self, state: TwinState, signals: Signals
    ) -> tuple[ObservationVerdict, ForecastPlan | None]:
        verdict = ObservationVerdict(
            action="accept",
            reason="FixedPolicy: accept all observations unconditionally (baseline).",
        )
        plan = ForecastPlan(
            ensemble_size=BASELINE["ensemble_size"],
            resolution_m=BASELINE["resolution_m"],
            workers=BASELINE["workers"],
            reuse_cache=False,
            reason="FixedPolicy: baseline settings (N=300, 150 m, 12 workers).",
        )
        return verdict, plan


# ---------------------------------------------------------------------------
# RulePolicy — accept / reject based on signals
# ---------------------------------------------------------------------------

@dataclass
class RulePolicy(ObservationPolicy):
    """
    Accept or reject observations based on checks.py signals.

    Rejection rules (any one is sufficient):
      1. ANY forecast-independent check failed (bad containment, implausible growth,
         FIRMS disagrees with the perimeter) — unless a wind shift was detected.
         (A wind shift can legitimately cause a large jump; independent checks are
         relaxed in that case.)
      2. The observation is too old (obs_age_min > max_age_min).
      3. It was explicitly injected as a fault (event.injected is not None) and
         reject_injected=True (default for unit tests; set False in production replay).

    ForecastPlan: scales N and resolution based on the operator deadline and
    the observation source.
    """
    # Thresholds — all can be overridden at construction time
    max_age_min: float = 360.0           # 6 hours: older than this, reject
    reject_injected: bool = False        # True to auto-reject faulted events (test mode)

    def decide(
        self, state: TwinState, signals: Signals
    ) -> tuple[ObservationVerdict, ForecastPlan | None]:

        current = state.get("current")
        request = state.get("request")

        # ---- decide -----------------------------------------------------------
        action: str = "accept"
        reason_parts: list[str] = []

        # Rule 1: forecast-independent checks (relax if wind_shift is detected)
        if signals.independent_check_failed and not signals.wind_shift:
            action = "reject"
            if not signals.contains_previous:
                reason_parts.append("perimeter does not contain the previous one")
            if not signals.growth_plausible:
                reason_parts.append("implied spread rate is physically implausible")
            if signals.agrees_with_firms is False:
                reason_parts.append("FIRMS hotspots are outside the perimeter")

        # Rule 2: stale observation
        if signals.obs_age_min > self.max_age_min:
            action = "reject"
            reason_parts.append(
                f"observation is {signals.obs_age_min:.0f} min old "
                f"(> {self.max_age_min:.0f} min threshold)"
            )

        # Rule 3: explicitly injected fault (test mode)
        if self.reject_injected and current and current.injected is not None:
            action = "reject"
            reason_parts.append(f"injected fault: {current.injected}")

        if action == "accept":
            reason = "All checks passed."
            if signals.wind_shift and signals.independent_check_failed:
                reason = (
                    "Independent checks flagged issues but wind_shift detected — "
                    "accepting (wind shift can cause legitimate perimeter jumps)."
                )
        else:
            reason = "Reject: " + "; ".join(reason_parts) + "."

        verdict = ObservationVerdict(action=action, reason=reason)  # type: ignore[arg-type]

        # ---- plan -------------------------------------------------------------
        plan = self._pick_plan(state, signals)
        return verdict, plan

    def _pick_plan(self, state: TwinState, signals: Signals) -> ForecastPlan:
        """Choose FARSITE settings based on deadline and source quality."""
        request = state.get("request")
        deadline = request.deadline_min if request else 60.0

        # Scale N with deadline: fast run (<=15 min budget): N=50, medium: N=150, full: N=300
        if deadline <= 15:
            n = 50
        elif deadline <= 45:
            n = 150
        else:
            n = BASELINE["ensemble_size"]  # 300

        # Coarser grid when the budget is tight, or when the observation is a satellite
        # hotspot: VIIRS/MODIS pixels are hundreds of metres across, so a fine grid
        # resolves detail the observation cannot constrain.
        # 150 m is the floor rather than 75 m because 75 m is far slower per member and
        # the baseline run established 150 m as sufficient; the weather agent drops to
        # 75 m when high wind justifies the cost (see agents/weather.RulePolicy).
        if signals.source in ("firms_viirs", "firms_modis") or deadline <= 15:
            res = 300
        else:
            res = BASELINE["resolution_m"]   # 150

        reason = (
            f"RulePolicy: source={signals.source}, deadline={deadline:.0f} min → "
            f"N={n}, res={res} m."
        )
        return ForecastPlan(
            ensemble_size=n,
            resolution_m=res,                 # type: ignore[arg-type]
            workers=BASELINE["workers"],
            reuse_cache=False,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Node factory — wrap a policy into a LangGraph node function
# ---------------------------------------------------------------------------

def make_observation_agent(policy: ObservationPolicy, hotspot_path: str | None = None):
    """
    Return a LangGraph node function that applies `policy` to each observation.

    The returned function:
      1. Computes Signals (all forecast-independent checks)
      2. Runs policy.decide(state, signals)
      3. Wraps the verdict + plan in a DecisionRecord
      4. Returns {"obs_verdict", "signals", "plan" (if policy returned one), "log"}

    Args:
        policy: FixedPolicy() or RulePolicy()
        hotspot_path: FIRMS CSV that enables the agrees_with_firms check. Without one
            that signal stays None ("cannot judge") and only the geometry checks run.
            For Border 2: "data/border2_inputs/firms_hotspots.csv".

    Example
    -------
        from orchestrator.agents.observation import make_observation_agent, RulePolicy
        nodes["observation"] = make_observation_agent(
            RulePolicy(), hotspot_path="data/border2_inputs/firms_hotspots.csv")
    """
    policy_name = type(policy).__name__

    def _node(state: TwinState) -> dict:
        t0 = time.perf_counter()

        # 1. Compute signals (forecast-independent checks + obs age + wind shift).
        #    A path given to the factory wins over one left in the cache.
        if hotspot_path is not None:
            state = {**state, "cache": {**(state.get("cache") or {}),
                                        "hotspot_path": hotspot_path}}
        signals = compute_signals(state)

        # 2. Policy decides
        verdict, plan = policy.decide(state, signals)

        # 3. Guardrail: always accept if there is no previous observation.
        #    Without a reference perimeter, containment and growth checks are uninformative.
        overridden = False
        if verdict.action == "reject":
            prev_path = state.get("prev_obs_path")
            if prev_path is None:
                verdict = ObservationVerdict(
                    action="accept",
                    reason=(
                        "Guardrail: no previous observation to compare against; "
                        "must accept the first observation to initialise the state."
                    ),
                    override_guardrail=True,
                )
                overridden = True

        # 4. Log
        current = state.get("current")
        record = DecisionRecord(
            sim_time=state.get("clock") or datetime.utcnow(),
            node="observation",
            policy=policy_name,
            inputs={
                "source": signals.source,
                "obs_age_min": round(signals.obs_age_min, 1),
                "contains_previous": signals.contains_previous,
                "growth_plausible": signals.growth_plausible,
                "agrees_with_firms": signals.agrees_with_firms,
                "wind_shift": signals.wind_shift,
                "consecutive_disagreements": signals.consecutive_disagreements,
                "injected": current.injected if current else None,
            },
            decision=verdict.model_dump(),
            overridden_by_guardrail=overridden,
            agent_latency_s=round(time.perf_counter() - t0, 4),
        )

        out: dict = {
            "obs_verdict": verdict,
            "signals": signals,
            "log": [record],
        }
        if plan is not None:
            out["plan"] = plan
        return out

    # Give the node a readable name for LangGraph tracing
    _node.__name__ = f"observation_agent[{policy_name}]"
    return _node
