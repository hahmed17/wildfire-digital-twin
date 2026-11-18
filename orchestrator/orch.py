"""
Agentic orchestration for the wildfire digital twin.

Suggested layout inside Hena's repo (her files untouched):
  orchestrator/
    state.py        # §1 TwinState + event/decision types
    adapter.py      # §2 calls into src/ (switch once her refactor lands)
    checks.py       # §3 consistency + independent checks (pure functions)
    costmodel.py    # §4 fitted from Hena's members.csv / steps.csv
    executors.py    # §5 LiveExecutor, TraceExecutor
    policies/       # §6 fixed.py, rules.py, llm.py   (same interface)
    guardrails.py   # §7 clamps every decision, whatever the policy says
    graph.py        # §8 LangGraph wiring
    replay.py       # §9 event-driven replay + scenario injection
    metrics.py      # §10 results table for the paper

Build order: §1 → §2 (stub) → §5 TraceExecutor on fake data → §6 FixedPolicy → §8 → §10
             → RulePolicy → LLMPolicy → real traces from Hena.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol
from pydantic import BaseModel, Field
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_<provider> import Chat<Model>          # same provider/version as AoI


# ===========================================================================
# §1  state.py — what flows around the loop (X, O, T in the diagram)
# ===========================================================================
Source = Literal["mapped_perimeter", "synthetic_perimeter", "firms_viirs", "firms_modis"]


class ObservationEvent(BaseModel):
    """One record per observation arrival in the replay (agree this with Hena)."""
    obs_time: datetime            # when the fire was observed
    arrival_time: datetime        # when it reached the system (latency matters)
    source: Source
    path: str                     # GeoJSON / CSV on disk
    injected: str | None = None   # None | "shifted_2km" | "stale" | "mislabeled" ...


class OperatorRequest(BaseModel):
    horizon_h: float              # how far ahead to forecast
    deadline_min: float           # wall-clock budget per forecast
    priority: int = 1


class TwinState(dict):
    """Passed between LangGraph nodes. Keep values serializable (checkpointing)."""
    # request: OperatorRequest
    # clock: datetime                         # simulated time in the replay
    # state_vec, cov_X: arrays / .npz path    # X — current analysis + covariance
    # prev_obs_geom: polygon                  # for the containment check
    # pending: list[ObservationEvent]         # O — arrived, not yet handled
    # ensemble: EnsembleResult | None         # latest forecast ensemble
    # signals: Signals | None                 # §3 outputs for the current obs
    # drift_history: list[float]              # recent normalized innovations
    # cache: dict                             # LCPs, weather pulls, warm-start states
    # log: list[DecisionRecord]


# ---- Decisions (structured outputs: the agent picks from a menu) ----------
VSIZE_PRESETS = {                 # metres; justified by source resolution, not new UQ
    "mapped_perimeter": 200,
    "synthetic_perimeter": 500,
    "firms_viirs": 750,           # ~375 m pixels
    "firms_modis": 1000,          # ~1 km pixels
}


class ForecastPlan(BaseModel):
    ensemble_size: int = Field(ge=4, le=300)
    resolution_m: Literal[75, 150, 300]           # dist_res = perim_res
    workers: int = Field(ge=1, le=8)              # parallel members on ONE machine
    reuse_cache: bool
    reason: str
    # placement: Literal["edge","local","hpc"]    # future work — same interface


class AssimilationDecision(BaseModel):
    action: Literal["assimilate", "inflate_and_assimilate", "skip", "reinitialize_from_obs"]
    vsize_m: float                                # chosen from VSIZE_PRESETS (optionally scaled)
    inflation: float = Field(default=1.0, ge=1.0, le=3.0)
    reason: str


@dataclass
class DecisionRecord:
    sim_time: str
    node: str
    policy: str
    inputs: dict              # the signals the policy saw (for the paper + debugging)
    decision: dict
    overridden_by_guardrail: bool
    agent_latency_s: float    # LLM time counts against the deadline
    tokens: int = 0


# ===========================================================================
# §2  adapter.py — the ONLY file that imports Hena's code
# ===========================================================================
# Today her code fuses forecast + assimilation in enkf.adjusted_state_EnKF_farsite.
# Ask her to split it; until then run_ensemble/enkf_update can wrap the fused call.

@dataclass
class EnsembleResult:
    forecast: "np.ndarray"        # zkphat (n_states, N)   — FARSITE output per member
    predicted_obs: "np.ndarray"   # ykhat  (n_output, N)
    valid_mask: "np.ndarray"      # which members succeeded
    member_wall_s: list[float]    # timing per member (from her logs or measured)


def load_observation(event: ObservationEvent) -> list: ...
    # perimeter -> geometry.geom_to_state ; FIRMS -> hotspot points (for checks only)

def fetch_weather(lat, lon, start, end): ...
    # firemap.query_weather_for_timestep, with the same fallback her notebook uses

def run_ensemble(state_vec, cov_X, plan: ForecastPlan, weather, dt, lcp) -> EnsembleResult: ...
    # after refactor: her run_ensemble(...)
    # members call farsite.forward_pass_farsite(poly, {windspeed, winddirection, dt},
    #                                           start_time, lcppath, dist_res, perim_res)

def enkf_update(ens: EnsembleResult, obs_vec, vsize_m, wsize_m, inflation): ...
    # after refactor: her enkf_update(...) -> (analysis_vec, new_cov_X)

def save_state(path, **arrays): ...    # enkf.save_enkf_state
def load_state(path): ...         # enkf.load_enkf_state


# ===========================================================================
# §3  checks.py — pure functions, no LLM. Computed BEFORE any policy decides.
# ===========================================================================
@dataclass
class Signals:
    normalized_innovation: float  # ~1 consistent, >>1 surprising (uses the forecast)
    # --- independent of the forecast (breaks the circularity) ---
    agrees_with_firms: bool | None    # perimeter vs. hotspots from ~same time
    contains_previous: bool           # fires rarely shrink
    growth_plausible: bool            # implied spread rate vs. wind/fuel bound
    # --- context ---
    obs_age_min: float
    source: Source
    wind_shift: bool                  # sharp change in wind dir/speed this interval
    consecutive_disagreements: int    # same-direction drift streak


def normalized_innovation(ens: EnsembleResult, obs_vec, vsize_m) -> float: ...
    # gap = mean |obs - mean(predicted_obs)| per vertex
    # expected = sqrt(ensemble_spread^2 + vsize^2)
    # return gap / expected        (formal chi-square version later)

def agrees_with_firms(perimeter, hotspots, buffer_m=500) -> bool | None: ...
def contains_previous(perimeter, prev_perimeter, tol=0.9) -> bool: ...
def growth_plausible(perimeter, prev_perimeter, dt, weather) -> bool: ...
def wind_shift(weather, deg_thresh=45, speed_thresh=10) -> bool: ...


# ===========================================================================
# §4  costmodel.py — fitted from Hena's logs (members.csv, steps.csv)
# ===========================================================================
class CostModel:
    """runtime ≈ overhead(res) + ceil(N / workers) * t_member(res) / (1 - fail_rate)
       error   ≈ f(N, res)  — fitted curve from the sweep"""
    def fit(self, members_csv, steps_csv): ...
    def estimate(self, plan: ForecastPlan) -> dict: ...   # {"runtime_s", "error_m", "fail_rate"}
    def best_plan_within(self, deadline_s) -> ForecastPlan: ...   # used by RulePolicy + guardrails


# ===========================================================================
# §5  executors.py — same interface, live or replayed
# ===========================================================================
class Executor(Protocol):
    def run_ensemble(self, state, plan: ForecastPlan) -> EnsembleResult: ...


class LiveExecutor:
    """Calls adapter.run_ensemble (real FARSITE). Use for the final 3–4-step comparison."""


class TraceExecutor:
    """Answers from Hena's saved runs: take the first N saved members for this
    step/resolution, time them from members.csv. EnKF update still runs for real.
    Caveat: later steps depend on earlier analyses -> use for one-step decisions
    from a shared start state, or for fast iteration before live runs."""
    def __init__(self, trace_dir): ...


# ===========================================================================
# §6  policies/ — one interface, three implementations
# ===========================================================================
class Policy(Protocol):
    name: str
    def plan_forecast(self, state, cost_model: CostModel) -> ForecastPlan: ...
    def decide_assimilation(self, state, signals: Signals) -> AssimilationDecision: ...


class FixedPolicy:
    """Baseline = today's notebook: N=300 (config), 150 m, 1 worker, no reuse,
    always assimilate, one fixed vsize (pin 300 vs 500 with Hena)."""
    name = "fixed"


class RulePolicy:
    """Thresholds: N = cost_model.best_plan_within(deadline); vsize from presets;
    ratio > k -> inflate; skip only if an independent check fails; streak >= 3 -> reinit."""
    name = "rules"


class LLMPolicy:
    """Same decisions, made by an LLM that sees Signals + context in a prompt.
        chain = prompt | llm.with_structured_output(AssimilationDecision)
        d = chain.invoke({...signals, recent drift, source, wind_shift, deadline...})
    Then ALWAYS pass through guardrails (§7). Reuse AoI's LLM setup.
    Value over rules: weighing ambiguous combinations (e.g. high ratio + wind shift +
    FIRMS agrees -> real behavior change, accept) and explaining the choice."""
    name = "llm"


# ===========================================================================
# §7  guardrails.py — applied to every policy's output (rules and LLM alike)
# ===========================================================================
def clamp_plan(plan: ForecastPlan, cost_model: CostModel, deadline_s) -> tuple[ForecastPlan, bool]: ...
    # shrink N / coarsen resolution until estimate(plan).runtime_s <= deadline_s

def clamp_assimilation(d: AssimilationDecision, s: Signals) -> tuple[AssimilationDecision, bool]: ...
    # "skip" allowed ONLY if an independent check failed:
    #     not s.contains_previous or not s.growth_plausible or s.agrees_with_firms is False
    #   otherwise -> downgrade to "inflate_and_assimilate"
    # s.consecutive_disagreements >= 3 -> force "reinitialize_from_obs" (model is the problem)
    # vsize must come from VSIZE_PRESETS[s.source] (± a bounded factor)


# ===========================================================================
# §8  graph.py — LangGraph wiring. Node = ask policy -> guardrail -> run task -> log
# ===========================================================================
def node_receive(state): ...        # pop next ObservationEvent, advance simulated clock
def node_plan(state): ...           # policy.plan_forecast -> clamp_plan -> log
def node_forecast(state): ...       # executor.run_ensemble (cache: LCP, weather, warm start)
def node_check(state): ...          # checks.* -> state["signals"]   (no LLM)
def node_assimilate(state): ...     # policy.decide_assimilation -> clamp_assimilation -> adapter.enkf_update / skip / reinit
def node_update(state): ...         # save_state, update drift_history + cache, log metrics


def route(state) -> str: ...
    # "receive" if more events, else END


def build_graph(policy: Policy, executor: Executor, cost_model: CostModel):
    """
    g = StateGraph(TwinState)
    for name, fn in [("receive", node_receive), ("plan", node_plan),
                     ("forecast", node_forecast), ("check", node_check),
                     ("assimilate", node_assimilate), ("update", node_update)]:
        g.add_node(name, partial(fn, policy=policy, executor=executor, cost_model=cost_model))
    g.set_entry_point("receive")
    g.add_edge("receive", "plan"); g.add_edge("plan", "forecast")
    g.add_edge("forecast", "check"); g.add_edge("check", "assimilate")
    g.add_edge("assimilate", "update")
    g.add_conditional_edges("update", route, {"receive": "receive", END: END})
    return g.compile(checkpointer=MemorySaver())
    """


# ===========================================================================
# §9  replay.py — Border 2 on a simulated clock + injected scenarios
# ===========================================================================
SCENARIOS = {
    "clean":       "real perimeters only, real timestamps",
    "bad_obs":     "1–2 perimeters shifted 2 km / stale / mislabeled source",
    "wind_shift":  "real surprising obs after a wind change (should be ACCEPTED)",
    "failures":    "randomly drop 20% of members",
    "deadlines":   "sweep deadline_min from tight to loose",
}

def build_events(fire="border_2", scenario="clean") -> list[ObservationEvent]: ...
    # read data/fire_perimeters/*, use REAL perimeters for evaluation (synthetic ones are
    # interpolated from real ones), apply scenario edits, sort by arrival_time

def replay(policy_name, scenario, executor="trace", seed=0): ...
    # graph = build_graph(POLICIES[policy_name], EXECUTORS[executor], CostModel().fit(...))
    # graph.invoke(initial_state(events, request))
    # write results/<scenario>/<policy>_<seed>.json  (DecisionRecords + per-step metrics)


# ===========================================================================
# §10 metrics.py — the paper's results table
# ===========================================================================
# per (policy, scenario): time-to-forecast, deadline hit rate, area error / IoU vs observed,
# core-seconds used, false rejections (good obs skipped), false acceptances (bad obs used),
# guardrail override count, agent latency + tokens.

# if __name__ == "__main__":
#     for scenario in SCENARIOS:
#         for p in ["fixed", "rules", "llm"]:
#             replay(p, scenario)