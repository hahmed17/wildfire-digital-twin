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
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

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
# LLMPolicy — same verdict, made by an LLM from the actual wind numbers
# ---------------------------------------------------------------------------

class LLMWeatherChoice(BaseModel):
    """The part of the verdict the LLM makes. used_fallback is a fact, filled in by code."""
    action: Literal["normal", "wind_shift_detected", "fallback_weather"] = Field(
        description=WeatherVerdict.model_fields["action"].description)
    reason: str = Field(
        description="One or two sentences on the weather conditions, citing the numbers.")


LLM_SYSTEM_PROMPT = """\
You are the weather agent in a wildfire digital twin. Each cycle, wind observations from the \
nearest weather station (RAWS) are fetched for the forecast window, and FARSITE uses them to \
predict fire spread. You summarise the conditions for the rest of the pipeline with one label:

- normal: nothing that changes how the forecast or the data correction should be treated.
- wind_shift_detected: the wind changed enough since the previous window that the fire's \
behaviour likely changed (e.g. it turned or sped up). Downstream, this tells the assimilation \
agent that a surprising fire observation this cycle is probably real, so it should be trusted \
rather than dismissed. Do not raise it for small, ordinary fluctuations.
- fallback_weather: no real data for this window; the weather service returned a static guess \
(10 mph from 225°). The forecast is running on made-up wind, so its uncertainty is higher.

Notes:
- Speeds are in mph; directions are degrees the wind blows FROM (0 = north, 90 = east).
- Direction changes are circular: 350° to 10° is a 20° change, not 340°.
- Guidance from the rule-based policy: it flags a shift when the mean direction rotates by \
{deg_thresh:.0f}° or more, or the mean speed changes by {speed_thresh_mph:.0f} mph or more. These \
are guidelines, not hard limits — a smaller rotation together with a large speed increase, or \
very strong winds, can matter more than either number alone; light, variable winds matter less.
- If the data is the fallback, say fallback_weather unless there is a real reason not to.
"""

LLM_HUMAN_PROMPT = """\
Wind for this forecast window:
{now}

Wind for the previous window:
{prev}

Change between windows:
{change}

Choose the label and explain why in one or two sentences citing the numbers."""


def _summarise_window(weather: dict | None) -> str:
    if weather is None:
        return "- none (first cycle: no previous window to compare against)"
    speeds, directions = checks.finite_readings(weather.get("wind_speeds"),
                                                weather.get("wind_directions"))
    lines = [f"- readings: {len(speeds)}",
             f"- used fallback (made-up data): {_is_fallback(weather)}"]
    if speeds:
        lines += [f"- mean speed: {np.mean(speeds):.1f} mph (min {min(speeds):.1f}, "
                  f"max {max(speeds):.1f})"]
    if directions:
        lines += [f"- mean direction: {checks.mean_direction_deg(directions):.0f}°"]
    return "\n".join(lines)


def _describe_change(prev: dict | None, now: dict) -> str:
    if prev is None:
        return "- not available (first cycle)"
    d_speed = _mean_speed_mph(now) - _mean_speed_mph(prev)
    raw = abs(checks.mean_direction_deg(now.get("wind_directions"))
              - checks.mean_direction_deg(prev.get("wind_directions"))) % 360
    d_dir = min(raw, 360 - raw)
    return f"- mean speed change: {d_speed:+.1f} mph\n- mean direction change: {d_dir:.0f}°"


class LLMPolicy(WeatherPolicy):
    """
    The LLM labels the weather window (normal / wind_shift_detected / fallback_weather)
    from the actual wind numbers for this window and the previous one. used_fallback is
    filled in by code, and the plan refinement is RulePolicy's, so the LLM and rule arms
    use the same resources.

    The verdict's wind_shift is the LLM's judgment (action == "wind_shift_detected"); the
    threshold-based value is still logged by the node as inputs["wind_shift"].

    If the LLM call fails or returns something unparsable, the RulePolicy verdict is used
    and the reason says so. After each decide(): `last_tokens` and `last_fallback`.
    """

    def __init__(self, llm, deg_thresh: float = 45.0, speed_thresh_mph: float = 10.0,
                 high_wind_mph: float = 25.0):
        from langchain_core.prompts import ChatPromptTemplate
        from orchestrator.llm import STRUCTURED_OUTPUT_METHOD

        # The node reads these two to compute its threshold-based wind_shift.
        self.deg_thresh = deg_thresh
        self.speed_thresh_mph = speed_thresh_mph
        self.rule = RulePolicy(deg_thresh=deg_thresh, speed_thresh_mph=speed_thresh_mph,
                               high_wind_mph=high_wind_mph)
        prompt = ChatPromptTemplate.from_messages(
            [("system", LLM_SYSTEM_PROMPT), ("human", LLM_HUMAN_PROMPT)]
        ).partial(deg_thresh=deg_thresh, speed_thresh_mph=speed_thresh_mph)
        self.chain = prompt | llm.with_structured_output(
            LLMWeatherChoice, method=STRUCTURED_OUTPUT_METHOD, include_raw=True)
        self.last_tokens = 0
        self.last_fallback = False

    def decide(
        self, state: TwinState, weather: dict, wind_shift: bool
    ) -> tuple[WeatherVerdict, ForecastPlan | None]:
        from orchestrator.llm import total_tokens

        plan = self.rule._refine_plan(state, _mean_speed_mph(weather))
        prev = (state.get("cache") or {}).get("weather_last")   # not rotated yet here
        self.last_tokens, self.last_fallback = 0, False

        try:
            out = self.chain.invoke({"now": _summarise_window(weather),
                                     "prev": _summarise_window(prev),
                                     "change": _describe_change(prev, weather)})
            self.last_tokens = total_tokens(out.get("raw"))
            choice = out.get("parsed")
            if choice is None:
                raise ValueError(f"unparsable LLM output: {out.get('parsing_error')}")
        except Exception as e:
            self.last_fallback = True
            verdict, _ = self.rule.decide(state, weather, wind_shift)
            return verdict.model_copy(update={
                "reason": f"LLM unavailable ({type(e).__name__}); rule fallback: {verdict.reason}"
            }), plan

        verdict = WeatherVerdict(
            action=choice.action,
            used_fallback=_is_fallback(weather),
            wind_shift=choice.action == "wind_shift_detected",
            reason=choice.reason,
        )
        return verdict, plan


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
        t0 = time.perf_counter()

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
                # LLMPolicy only: True if the LLM call failed and the rule decided instead.
                "llm_fallback": getattr(policy, "last_fallback", None),
            },
            decision=verdict.model_dump(),
            overridden_by_guardrail=overridden,
            agent_latency_s=round(time.perf_counter() - t0, 4),
            tokens=getattr(policy, "last_tokens", 0) or 0,
        )
        out["log"] = [record]
        return out

    _node.__name__ = f"weather_agent[{policy_name}]"
    return _node