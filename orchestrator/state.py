"""
§1 state.py — what flows around the loop (State X, Observation O, Time window T).

Pure data types only: no FARSITE / geo imports, so policies, guardrails and tests
can import this module without the heavy simulation stack.
"""
from __future__ import annotations

from datetime import datetime
import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, computed_field, model_validator

Source = Literal["mapped_perimeter", "synthetic_perimeter", "firms_viirs", "firms_modis"]
Resolution = Literal[75, 150, 300]


# ---------------------------------------------------------------------------
# Inputs: observations (O) and operator requests (T)
# ---------------------------------------------------------------------------
class ObservationEvent(BaseModel):
    """One record per observation arrival in the replay."""
    obs_time: datetime                 # when the fire was observed
    arrival_time: datetime             # when it reached the system (latency matters)
    source: Source
    path: str                          # GeoJSON / CSV on disk
    feature_id: str | None = None      # row/feature inside `path`, if it holds many
    injected: str | None = None        # None | "shifted_2km" | "stale" | "mislabeled" ...

    @model_validator(mode="after")
    def _arrival_after_obs(self):
        # An observation can't reach the system before it was made.
        if self.arrival_time < self.obs_time:
            raise ValueError(
                f"arrival_time ({self.arrival_time}) is before obs_time ({self.obs_time})")
        return self

    @property
    def latency_min(self) -> float:
        return (self.arrival_time - self.obs_time).total_seconds() / 60.0


class OperatorRequest(BaseModel): # this is the "T" in the loop
    deadline_min: float = Field(gt=0)  # wall-clock budget per forecast
    priority: int = 1


# ---------------------------------------------------------------------------
# Decisions (structured outputs: every policy picks from a menu)
# ---------------------------------------------------------------------------
VSIZE_PRESETS: dict[str, float] = {    # metres; justified by source resolution
    "mapped_perimeter": 200,
    "synthetic_perimeter": 500,
    "firms_viirs": 750,                # ~375 m pixels
    "firms_modis": 1000,               # ~1 km pixels
}

# Baseline run (outputs/Border_2_Real/steps.csv, members.csv): N=300, 150 m, 12 workers,
# vsize=300, wsize=100. FixedPolicy reproduces exactly this.
BASELINE = dict(ensemble_size=300, resolution_m=150, workers=12, vsize_m=300.0, wsize_m=100.0)

MAX_WORKERS = 32                       # one machine; baseline already used 12


class ForecastPlan(BaseModel):
    """How to run the next FARSITE ensemble forecast. Choose settings that give the
    most accurate forecast that still finishes before the operator's deadline."""
    ensemble_size: int = Field(
        ge=4, le=300,
        description="Number of FARSITE ensemble members (N). More members give a better "
                    "estimate of uncertainty and a more accurate EnKF update, but runtime "
                    "grows roughly linearly with N.")
    resolution_m: Resolution = Field(
        description="FARSITE distance and perimeter resolution in metres (dist_res = "
                    "perim_res). Finer (75) is more detailed but much slower per member; "
                    "coarser (300) is fast but less precise. Baseline is 150.")
    workers: int = Field(
        ge=1, le=MAX_WORKERS,
        description="Number of ensemble members run in parallel on one machine. More "
                    "workers shorten wall-clock time until the machine's cores are saturated.")
    reuse_cache: bool = Field(
        default=False,
        description="Reuse cached inputs (landscape file, weather pulls, previous state) "
                    "instead of recomputing them, to save time.")
    reason: str = Field(
        default="",
        description="One or two sentences explaining why these settings were chosen.")


ObsAction = Literal["accept", "reject"]
WeatherAction = Literal["normal", "wind_shift_detected", "fallback_weather"]


class ObservationVerdict(BaseModel):
    """Decision made by the observation agent about whether to trust the new perimeter."""
    action: ObsAction = Field(
        description="'accept': use the observation as-is. "
                    "'reject': discard it and ask synthesize to build a synthetic one.")
    reason: str = Field(
        default="",
        description="One sentence explaining the decision, citing the check(s) that drove it.")
    override_guardrail: bool = Field(
        default=False,
        description="True if a guardrail overrode the agent's initial action.")


class WeatherVerdict(BaseModel):
    """Decision made by the weather agent about conditions for the forecast window."""
    action: WeatherAction = Field(
        description="'normal': no special treatment needed. "
                    "'wind_shift_detected': a large wind shift was flagged; the assimilation "
                    "agent should consider inflate_and_assimilate. "
                    "'fallback_weather': firemap returned the static fallback; forecast "
                    "uncertainty is higher than usual.")
    used_fallback: bool = Field(
        default=False,
        description="True if firemap returned the static fallback (10 mph, 225°) instead "
                    "of live RAWS data.")
    wind_shift: bool = Field(
        default=False,
        description="True if wind direction or speed changed sharply since the last cycle.")
    reason: str = Field(
        default="",
        description="One sentence noting any notable weather conditions.")


AssimAction = Literal["assimilate", "inflate_and_assimilate", "skip", "reinitialize_from_obs"]


class AssimilationDecision(BaseModel):
    """What to do with a new fire observation after comparing it with the forecast."""
    action: AssimAction = Field(
        description="'assimilate': normal EnKF update with the observation. "
                    "'inflate_and_assimilate': widen the ensemble spread first, for when the "
                    "observation is surprising but believable (e.g. after a wind shift). "
                    "'skip': ignore the observation; only appropriate when an independent "
                    "check says it is wrong. "
                    "'reinitialize_from_obs': restart the state from the observation, for "
                    "when the forecast has drifted away for several steps in a row.")
    vsize_m: float = Field(
        gt=0,
        description="Observation error standard deviation in metres: how far the observed "
                    "perimeter may be from the true one. Larger = trust the observation less. "
                    "Should match the source (mapped perimeter ~200, synthetic ~500, VIIRS "
                    "~750, MODIS ~1000).")
    wsize_m: float = Field(
        default=BASELINE["wsize_m"], gt=0,
        description="Model error standard deviation in metres: how far FARSITE's forecast "
                    "may be from the truth. Larger = trust the forecast less.")
    inflation: float = Field(
        default=1.0, ge=1.0, le=3.0,
        description="Factor that multiplies the ensemble spread before the update. 1.0 = no "
                    "inflation. Use more than 1.0 only with 'inflate_and_assimilate'.")
    reason: str = Field(
        default="",
        description="One or two sentences explaining the decision, citing the signals used.")


# ---------------------------------------------------------------------------
# Signals (§3 output) — computed before any policy decides, no LLM
# ---------------------------------------------------------------------------

class CheckDetail(BaseModel):
    """The number behind one True/False check, so an LLM can tell a near miss from a
    clear failure. Rule and Fixed policies only read the True/False fields."""
    passed: bool | None = Field(description="The check's True/False result (None = could not judge).")
    value: float | None = Field(default=None, description="The measured value.")
    threshold: float | None = Field(default=None, description="The cutoff the True/False uses.")
    unit: str = Field(default="", description="What the value and threshold measure.")


class Signals(BaseModel):
    """Checks run on a new observation before any decision is made. Some compare it
    with the forecast; others are independent of the forecast and can catch a bad
    observation even when the forecast itself is wrong."""
    normalized_innovation: float = Field(
        ge=0,
        description="How surprising the observation is given the forecast. About 1 means "
                    "consistent; much greater than 1 means the observation is far from "
                    "what the ensemble predicted.")
    agrees_with_firms: bool | None = Field(
        description="Whether satellite hotspots from about the same time fall inside the "
                    "observed perimeter. Independent of the forecast. None = no hotspots.")
    contains_previous: bool = Field(
        description="Whether the new perimeter contains the previous one. Fires rarely "
                    "shrink, so False suggests a bad observation. Independent of the forecast.")
    growth_plausible: bool = Field(
        description="Whether the implied spread rate is physically possible given wind and "
                    "fuel. Independent of the forecast.")
    obs_age_min: float = Field(
        ge=0, description="Minutes between when the fire was observed and now.")
    source: Source = Field(description="Type of observation.")
    wind_shift: bool = Field(
        description="Whether wind direction or speed changed sharply this interval. A "
                    "surprising observation after a wind shift is likely real.")
    consecutive_disagreements: int = Field(
        ge=0,
        description="How many steps in a row the forecast has missed in the same direction. "
                    "A long streak means the model, not the observation, is the problem.")
    details: dict[str, CheckDetail] = Field(
        default_factory=dict,
        description="Optional numbers behind the True/False checks, keyed by check name "
                    "(e.g. 'contains_previous'). Empty when not computed. Only the LLM "
                    "policies read it; the True/False fields above are unchanged.")

    @computed_field
    @property
    def independent_check_failed(self) -> bool:
        """True if any forecast-independent check says the observation is wrong."""
        return (not self.contains_previous
                or not self.growth_plausible
                or self.agrees_with_firms is False)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class DecisionRecord(BaseModel):
    """One row per policy decision. Written to the results JSON, so it's a BaseModel
    (model_dump_json / model_validate_json give a lossless round trip)."""
    sim_time: datetime                 # simulated clock when the decision was made
    node: str                          # "plan" | "assimilate"
    policy: str                        # "fixed" | "rules" | "llm"
    inputs: dict[str, Any]             # what the policy saw
    decision: dict[str, Any]           # what it chose (after guardrails)
    overridden_by_guardrail: bool
    agent_latency_s: float = Field(ge=0)   # LLM time counts against the deadline
    tokens: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Graph state — keep values serializable (LangGraph checkpointing)
# ---------------------------------------------------------------------------
class TwinState(TypedDict):  # The shared dictionary LangGraph passes from node to node during a run.
    request: OperatorRequest
    clock: datetime                    # simulated time in the replay
    step: int
    state_path: str                    # .npz holding analysis vector + covariance X
    prev_obs_path: str | None          # previous observed perimeter (containment check)
    pending: list[ObservationEvent]    # arrived, not yet handled
    current: ObservationEvent | None   # event being processed this cycle
    plan: ForecastPlan | None
    ensemble_path: str | None          # latest forecast ensemble on disk
    signals: Signals | None
    # One key per agent, so agents never overwrite each other and later agents can read
    # earlier verdicts. Routing only needs a `.action` attribute, so plain dicts also work.
    obs_verdict: ObservationVerdict | None
    weather_verdict: WeatherVerdict | None
    assim_verdict: Any | None
    drift_history: list[float]         # recent normalized innovations
    cache: dict[str, Any]              # LCP, weather pulls ("weather_last"), warm-start states,
                                       # accepted observations ("good_obs", for synthetic perimeters)
    # Annotated[..., operator.add] = reducer: nodes return only NEW items and LangGraph appends them
    log: Annotated[list[DecisionRecord], operator.add]
    metrics: Annotated[list[dict], operator.add]


def initial_state(events: list[ObservationEvent], request: OperatorRequest,
                  state_path: str) -> TwinState:
    if not events:
        raise ValueError("initial_state needs at least one observation event")
    events = sorted(events, key=lambda e: e.arrival_time)
    return TwinState(
        request=request,
        clock=events[0].arrival_time,
        step=0,
        state_path=state_path,
        prev_obs_path=None,
        pending=events,
        current=None,
        plan=None,
        ensemble_path=None,
        signals=None,
        obs_verdict=None,
        weather_verdict=None,
        assim_verdict=None,
        drift_history=[],
        cache={},
        log=[],
        metrics=[],
    )