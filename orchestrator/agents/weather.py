"""
§6 weather_agent — fetches the weather for the forecast window and flags conditions
that change how the forecast or the assimilation should be run.

Two policies are provided:
  FixedPolicy  — always "normal" (reproduces the baseline Border 2 run)
  RulePolicy   — flags wind shifts and fallback weather; can refine the forecast plan

Both are callables taking TwinState + the fetched weather dict and returning a
WeatherVerdict.

The node is the only place that touches the weather service, so it owns the weather
cache. It maintains two keys:

    cache["weather_last"]  the window just fetched (this cycle's forecast interval)
    cache["weather_prev"]  the window before it (what "weather_last" held on entry)

`synthesize` reads `weather_last` to grow a synthetic perimeter downwind, and the next
cycle's observation agent compares `weather_prev` to `weather_last` for its (one cycle
lagged) wind_shift signal. See the note in agents/observation.compute_signals.

Usage
-----
    from orchestrator.agents.weather import make_weather_agent, RulePolicy

    nodes["weather"] = make_weather_agent(RulePolicy())

    # Offline / unit test: inject a fetch function instead of calling firemap
    nodes["weather"] = make_weather_agent(FixedPolicy(), fetch=lambda s, e: FAKE_WEATHER)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from orchestrator import checks
from orchestrator.state import (
    DecisionRecord,
    ForecastPlan,
    Signals,
    TwinState,
    WeatherVerdict,
)

# Border 2 fire centroid — the default location adapter.fetch_weather uses.
BORDER2_LAT = 32.59588662537945
BORDER2_LON = -116.85708741551528

# firemap's static fallback when no RAWS data is available (see firemap.py).
FALLBACK_SPEEDS = [10.0]
FALLBACK_DIRECTIONS = [225.0]

# Default forecast interval when there is no previous observation to measure from.
DEFAULT_WINDOW_HOURS = 1.0


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def forecast_window(state: TwinState) -> tuple[datetime, datetime]:
    """
    Work out the interval the forecast will cover: from the last analysis time to
    the time of the observation being assimilated this cycle.

    The analysis time is the obs_time of the most recently accepted observation
    (cache["good_obs"][-1]). On the first cycle there is none, so the window is
    DEFAULT_WINDOW_HOURS long ending at the current observation.

    Returns:
        (start, end) — both tz-aware if the events are
    """
    current = state.get("current")
    end = current.obs_time if current else (state.get("clock") or datetime.utcnow())

    cache = state.get("cache") or {}
    good_obs = cache.get("good_obs", [])
    start = None
    if good_obs:
        last = good_obs[-1]
        # good_obs holds model_dump(mode="json") dicts (see steps.update)
        last_time = last.get("obs_time") if isinstance(last, dict) else getattr(last, "obs_time", None)
        if isinstance(last_time, str):
            try:
                last_time = datetime.fromisoformat(last_time)
            except ValueError:
                last_time = None
        if last_time is not None and last_time < end:
            start = last_time

    if start is None:
        start = end - timedelta(hours=DEFAULT_WINDOW_HOURS)

    return start, end


def _is_fallback(weather: dict) -> bool:
    """True if firemap returned its static fallback instead of real RAWS data."""
    if weather.get("used_fallback") is not None:
        return bool(weather["used_fallback"])
    return (list(weather.get("wind_speeds", [])) == FALLBACK_SPEEDS
            and list(weather.get("wind_directions", [])) == FALLBACK_DIRECTIONS)


def _mean_speed_mph(weather: dict) -> float:
    """Mean usable wind speed (mph). Skips gaps: a real RAWS window can contain
    None, and np.mean over it raises."""
    return checks.mean_speed(weather.get("wind_speeds"))


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class WeatherPolicy(ABC):
    """
    A policy takes the state and the weather just fetched, and returns a WeatherVerdict.

    It may also return a revised ForecastPlan (or None to keep the one the observation
    agent chose). The weather agent is the right place to refine the plan, because wind
    speed determines how short the FARSITE sub-step needs to be.
    """

    @abstractmethod
    def decide(
        self, state: TwinState, weather: dict, wind_shift: bool
    ) -> tuple[WeatherVerdict, ForecastPlan | None]:
        ...


# ---------------------------------------------------------------------------
# FixedPolicy — always normal (reproduces Border_2_Real)
# ---------------------------------------------------------------------------

class FixedPolicy(WeatherPolicy):
    """Report the weather and nothing else: never flags a shift, never changes the plan.

    This is the control condition. The baseline Border 2 run used the weather as fetched
    with no conditional handling, so this reproduces it exactly.
    """

    def decide(
        self, state: TwinState, weather: dict, wind_shift: bool
    ) -> tuple[WeatherVerdict, ForecastPlan | None]:
        verdict = WeatherVerdict(
            action="normal",
            used_fallback=_is_fallback(weather),
            wind_shift=wind_shift,          # recorded, but not acted on
            reason=(f"FixedPolicy: mean wind {_mean_speed_mph(weather):.1f} mph; "
                    "no conditional handling (baseline)."),
        )
        return verdict, None


# ---------------------------------------------------------------------------
# RulePolicy — flag shifts and fallback weather, refine the plan
# ---------------------------------------------------------------------------

@dataclass
class RulePolicy(WeatherPolicy):
    """
    Flag conditions that downstream nodes should treat differently.

    Verdict actions, in priority order:
      1. "wind_shift_detected" — direction rotated > deg_thresh or speed changed by
         more than speed_thresh_mph since the previous window. The assimilation agent
         reads this and prefers 'inflate_and_assimilate', because a surprising
         observation after a wind shift is likely real rather than a bad reading.
      2. "fallback_weather"    — firemap returned its static fallback (10 mph / 225°).
         The forecast is running on a guess, so uncertainty is higher than usual.
      3. "normal"              — neither.

    Plan refinement: above `high_wind_mph` the fire front moves far within one FARSITE
    sub-step, so the plan is nudged to a finer resolution when the deadline allows it.
    (The sub-step length itself is `farsite_dt_minutes` in adapter.run_ensemble; the
    experiment notebook uses 15 min for high-wind steps and 30 otherwise.)
    """
    deg_thresh: float = 45.0
    speed_thresh_mph: float = 10.0
    high_wind_mph: float = 25.0
    refine_on_high_wind: bool = True

    def decide(
        self, state: TwinState, weather: dict, wind_shift: bool
    ) -> tuple[WeatherVerdict, ForecastPlan | None]:

        mean_mph = _mean_speed_mph(weather)
        fallback = _is_fallback(weather)

        if wind_shift:
            action = "wind_shift_detected"
            reason = (f"Wind shifted sharply since the last window "
                      f"(> {self.deg_thresh:.0f}° or > {self.speed_thresh_mph:.0f} mph); "
                      f"mean wind now {mean_mph:.1f} mph. A surprising observation this "
                      "cycle is likely real.")
        elif fallback:
            action = "fallback_weather"
            reason = ("No RAWS data for this window; firemap returned its static fallback "
                      f"({FALLBACK_SPEEDS[0]:.0f} mph from {FALLBACK_DIRECTIONS[0]:.0f}°). "
                      "Forecast uncertainty is higher than usual.")
        else:
            action = "normal"
            reason = f"Conditions steady; mean wind {mean_mph:.1f} mph."

        verdict = WeatherVerdict(
            action=action,                    # type: ignore[arg-type]
            used_fallback=fallback,
            wind_shift=wind_shift,
            reason=reason,
        )

        plan = self._refine_plan(state, mean_mph)
        return verdict, plan

    def _refine_plan(self, state: TwinState, mean_mph: float) -> ForecastPlan | None:
        """Nudge the observation agent's plan to a finer resolution in high wind.
        Returns None when no change is warranted (the existing plan stands)."""
        if not self.refine_on_high_wind or mean_mph < self.high_wind_mph:
            return None

        plan = state.get("plan")
        if plan is None:
            return None

        request = state.get("request")
        deadline = request.deadline_min if request else 60.0

        # Only refine when there is budget for it, and only one step finer.
        finer = {300: 150, 150: 75}.get(plan.resolution_m)
        if finer is None or deadline <= 30:
            return None

        return ForecastPlan(
            ensemble_size=plan.ensemble_size,
            resolution_m=finer,               # type: ignore[arg-type]
            workers=plan.workers,
            reuse_cache=plan.reuse_cache,
            reason=(f"RulePolicy(weather): mean wind {mean_mph:.1f} mph "
                    f"(>= {self.high_wind_mph:.0f}); refined resolution "
                    f"{plan.resolution_m} m -> {finer} m within the "
                    f"{deadline:.0f} min budget."),
        )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_weather_agent(
    policy: WeatherPolicy,
    fetch=None,
    lat: float = BORDER2_LAT,
    lon: float = BORDER2_LON,
):
    """
    Return a LangGraph node function that fetches weather and applies `policy`.

    The returned function:
      1. Works out the forecast window (forecast_window)
      2. Fetches the weather for it
      3. Computes wind_shift against cache["weather_last"]
      4. Runs policy.decide(...)
      5. Rotates the weather cache (weather_last -> weather_prev, new -> weather_last)
      6. Overwrites signals.wind_shift so the assimilation agent sees the current value
      7. Returns {"weather_verdict", "cache", "signals"?, "plan"?, "log"}

    Args:
        policy: FixedPolicy() or RulePolicy()
        fetch:  callable(start, end) -> weather dict. Defaults to adapter.fetch_weather,
                imported lazily so this module can be tested without FARSITE installed.
                Inject a stub in tests.
        lat, lon: location passed to the default fetch (Border 2 centroid).

    Example
    -------
        nodes["weather"] = make_weather_agent(RulePolicy())
    """
    policy_name = type(policy).__name__

    def _default_fetch(start: datetime, end: datetime) -> dict:
        # Imported here, not at module scope: adapter pulls in enkf/farsite/geopandas.
        from orchestrator import adapter
        return adapter.fetch_weather(start, end, lat=lat, lon=lon)

    do_fetch = fetch if fetch is not None else _default_fetch

    def _node(state: TwinState) -> dict:
        t0 = time.time()

        # 1-2. Window + fetch. A service failure must not kill the replay: fall back.
        start, end = forecast_window(state)
        fetch_error: str | None = None
        try:
            weather = do_fetch(start, end)
        except Exception as exc:                                  # noqa: BLE001
            fetch_error = f"{type(exc).__name__}: {exc}"
            weather = {
                "wind_speeds": list(FALLBACK_SPEEDS),
                "wind_directions": list(FALLBACK_DIRECTIONS),
                "used_fallback": True,
            }

        # 3. Wind shift against the window the previous cycle fetched
        cache = dict(state.get("cache") or {})
        prev_window = cache.get("weather_last")
        shift = checks.wind_shift(
            prev_window, weather,
            deg_thresh=getattr(policy, "deg_thresh", 45.0),
            speed_thresh_mph=getattr(policy, "speed_thresh_mph", 10.0),
        )

        # 4. Policy decides
        verdict, plan = policy.decide(state, weather, shift)

        # Guardrail: a fetch failure is always worth flagging, whatever the policy said.
        overridden = False
        if fetch_error is not None and verdict.action == "normal":
            verdict = WeatherVerdict(
                action="fallback_weather",
                used_fallback=True,
                wind_shift=shift,
                reason=(f"Guardrail: weather fetch failed ({fetch_error}); "
                        "using the static fallback. Forecast uncertainty is higher."),
            )
            overridden = True

        # 5. Rotate the cache so the next cycle can compare windows
        if prev_window is not None:
            cache["weather_prev"] = prev_window
        cache["weather_last"] = weather

        out: dict = {
            "weather_verdict": verdict,
            "cache": cache,
            "log": [],
        }

        # 6. Keep signals in step: the observation agent's wind_shift is one cycle
        #    lagged, so overwrite it with the value for the window being forecast.
        signals = state.get("signals")
        if signals is not None and signals.wind_shift != shift:
            out["signals"] = signals.model_copy(update={"wind_shift": shift})

        if plan is not None:
            out["plan"] = plan

        # 7. Log
        record = DecisionRecord(
            sim_time=state.get("clock") or datetime.utcnow(),
            node="weather",
            policy=policy_name,
            inputs={
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "n_readings": len(weather.get("wind_speeds") or []),
                "mean_wind_mph": round(_mean_speed_mph(weather), 2),
                "used_fallback": _is_fallback(weather),
                "wind_shift": shift,
                "fetch_error": fetch_error,
            },
            decision=verdict.model_dump(),
            overridden_by_guardrail=overridden,
            agent_latency_s=round(time.time() - t0, 4),
        )
        out["log"] = [record]
        return out

    _node.__name__ = f"weather_agent[{policy_name}]"
    return _node