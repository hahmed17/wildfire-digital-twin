"""
§6 assimilation_agent — decides how to fold the observation into the fire state.

This is the agent with real consequences: it picks one of four EnKF actions and sets
the noise parameters that control how much the observation moves the state.

Two policies are provided:
  FixedPolicy — always 'assimilate' with the baseline vsize/wsize (reproduces
                Border_2_Real: vsize=300, wsize=100 from outputs/Border_2_Real/steps.csv)
  RulePolicy  — chooses among the four actions using the signals, the weather verdict
                and the drift history

Unlike the other two agents, this one needs the forecast ensemble: normalized_innovation
compares the observation against the spread of FARSITE members. The ensemble lives on
disk (TwinState holds `ensemble_path`, not the arrays), so the node loads it. The loader
is injectable, so the policy logic can be tested without FARSITE or a real .npz.

Usage
-----
    from orchestrator.agents.assimilation import make_assimilation_agent, RulePolicy

    nodes["assimilation"] = make_assimilation_agent(RulePolicy())

    # Offline: feed a known innovation instead of loading an ensemble
    nodes["assimilation"] = make_assimilation_agent(
        RulePolicy(), load_ensemble=lambda state: (None, None))
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from orchestrator import checks
from orchestrator.state import (
    BASELINE,
    VSIZE_PRESETS,
    AssimAction,
    AssimilationDecision,
    DecisionRecord,
    Signals,
    TwinState,
)

# How many recent innovations to keep in drift_history.
DRIFT_HISTORY_LEN = 6

# An innovation above this is "surprising": the observation sits well outside the
# ensemble spread. 1.0 means "about as far out as a typical member".
SURPRISE_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Ensemble loading (injectable)
# ---------------------------------------------------------------------------

def default_load_ensemble(state: TwinState) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load the forecast ensemble and the observation vector for this cycle.

    Returns (forecast_ensemble, obs_vec), either of which may be None when the
    ensemble is not available (e.g. the placeholder forecast node, or a missing file).
    A None forecast makes normalized_innovation unavailable, and the policies fall
    back to the forecast-independent signals.

    Returns:
        forecast_ensemble: (n_states, N) zkphat, or None
        obs_vec:           (n_states, 1) observed state, or None
    """
    path = state.get("ensemble_path")
    if not path:
        return None, None
    try:
        data = np.load(path, allow_pickle=True)
        forecast = data["zkphat_ensemble"]
    except Exception:
        return None, None

    # The observation vector: load it from the current event.
    obs_vec = None
    current = state.get("current")
    if current is not None:
        try:
            from orchestrator import adapter
            obs_vec = adapter.load_observation(current)
        except Exception:
            obs_vec = None
    return forecast, obs_vec


def compute_innovation(
    state: TwinState,
    forecast: np.ndarray | None,
    obs_vec: np.ndarray | None,
    vsize_m: float,
) -> float | None:
    """Normalized innovation for this cycle, or None when the ensemble is unavailable."""
    if forecast is None or obs_vec is None:
        return None
    try:
        return checks.normalized_innovation(forecast, obs_vec, vsize_m)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class AssimilationPolicy(ABC):
    """Takes the state, the signals (with normalized_innovation filled in if available)
    and returns an AssimilationDecision."""

    @abstractmethod
    def decide(self, state: TwinState, signals: Signals) -> AssimilationDecision:
        ...


# ---------------------------------------------------------------------------
# FixedPolicy — always assimilate with the baseline noise (reproduces Border_2_Real)
# ---------------------------------------------------------------------------

class FixedPolicy(AssimilationPolicy):
    """Always a plain EnKF update with the baseline noise parameters.

    This is the control condition. outputs/Border_2_Real/steps.csv shows every step of
    the baseline run used vsize=300, wsize=100 and a straight update, so this reproduces
    it exactly — including for synthetic perimeters, which the baseline did not
    down-weight.
    """

    def decide(self, state: TwinState, signals: Signals) -> AssimilationDecision:
        return AssimilationDecision(
            action="assimilate",
            vsize_m=BASELINE["vsize_m"],
            wsize_m=BASELINE["wsize_m"],
            inflation=1.0,
            reason=("FixedPolicy: plain EnKF update with baseline noise "
                    f"(vsize={BASELINE['vsize_m']:.0f}, wsize={BASELINE['wsize_m']:.0f})."),
        )


# ---------------------------------------------------------------------------
# RulePolicy — pick among the four actions
# ---------------------------------------------------------------------------

@dataclass
class RulePolicy(AssimilationPolicy):
    """
    Choose the EnKF action from the signals, in priority order:

      1. 'skip' — an independent check says the observation is wrong AND no wind shift
         explains it. Skipping discards data, so it is reserved for the one case where
         something other than the forecast says the observation is bad.

      2. 'reinitialize_from_obs' — the forecast has disagreed with the observations
         `drift_steps` times in a row. A long streak means the model, not the
         observation, is the problem, so the state is restarted from the observation.

      3. 'inflate_and_assimilate' — the observation is surprising
         (normalized_innovation > surprise) but believable: either a wind shift was
         detected, or the independent checks all passed. Widening the prior spread
         first lets the update move further without the filter dismissing the
         observation as an outlier.

      4. 'assimilate' — everything else.

    vsize_m comes from VSIZE_PRESETS for the observation's source, so a synthetic
    perimeter (500 m) is trusted less than a mapped one (200 m). Set
    `use_source_vsize=False` to pin it to the baseline instead.
    """
    surprise: float = SURPRISE_THRESHOLD
    drift_steps: int = 3
    inflation: float = 1.4
    use_source_vsize: bool = True
    wsize_m: float = BASELINE["wsize_m"]

    def decide(self, state: TwinState, signals: Signals) -> AssimilationDecision:
        vsize = (VSIZE_PRESETS.get(signals.source, BASELINE["vsize_m"])
                 if self.use_source_vsize else BASELINE["vsize_m"])

        # The weather agent's verdict is the authority on wind shift for this window.
        wx = state.get("weather_verdict")
        wind_shift = bool(getattr(wx, "wind_shift", None)) or signals.wind_shift

        # The node sets cache["innovation_available"] to False when there was no
        # ensemble to measure against, so a missing innovation is never mistaken
        # for a low one.
        cache = state.get("cache") or {}
        have_innovation = cache.get("innovation_available", True)
        innovation = signals.normalized_innovation
        surprising = have_innovation and innovation > self.surprise

        # --- 1. an independent check failed and no wind shift explains it -------
        if signals.independent_check_failed and not wind_shift:
            failed = []
            if not signals.contains_previous:
                failed.append("does not contain the previous perimeter")
            if not signals.growth_plausible:
                failed.append("implied spread rate is implausible")
            if signals.agrees_with_firms is False:
                failed.append("FIRMS hotspots fall outside it")
            return AssimilationDecision(
                action="skip", vsize_m=vsize, wsize_m=self.wsize_m, inflation=1.0,
                reason=("Skip: an independent check rejects this observation "
                        f"({'; '.join(failed)}) and no wind shift explains it."),
            )

        # --- 2. the forecast has drifted for several steps ----------------------
        if signals.consecutive_disagreements >= self.drift_steps:
            return AssimilationDecision(
                action="reinitialize_from_obs", vsize_m=vsize, wsize_m=self.wsize_m,
                inflation=1.0,
                reason=(f"Reinitialize: the forecast has disagreed "
                        f"{signals.consecutive_disagreements} steps in a row "
                        f"(>= {self.drift_steps}), so the model has drifted, not the "
                        "observation."),
            )

        # --- 3. surprising but believable ---------------------------------------
        if surprising:
            why = ("a wind shift was detected" if wind_shift
                   else "every independent check passed")
            return AssimilationDecision(
                action="inflate_and_assimilate", vsize_m=vsize, wsize_m=self.wsize_m,
                inflation=self.inflation,
                reason=(f"Inflate and assimilate: innovation {innovation:.2f} "
                        f"(> {self.surprise:.1f}) is surprising but believable because "
                        f"{why}; widening the prior spread by {self.inflation:.1f}x."),
            )

        # --- 4. normal ----------------------------------------------------------
        detail = (f"innovation {innovation:.2f}" if have_innovation
                  else "innovation unavailable (no ensemble)")
        return AssimilationDecision(
            action="assimilate", vsize_m=vsize, wsize_m=self.wsize_m, inflation=1.0,
            reason=f"Plain update: {detail}; all checks consistent.",
        )


# ---------------------------------------------------------------------------
# LLMPolicy — same decision, made by an LLM from the same signals
# ---------------------------------------------------------------------------

class LLMAssimilationChoice(BaseModel):
    """The part of the decision the LLM makes. vsize and wsize are facts about the sensor
    and the model, not judgment calls, so code fills them in (same values as RulePolicy)."""
    action: AssimAction = Field(description=AssimilationDecision.model_fields["action"].description)
    inflation: float = Field(
        default=1.0, ge=1.0, le=3.0,
        description="Factor that widens the forecast ensemble spread before the update. "
                    "1.0 = no inflation. Only used with 'inflate_and_assimilate'; larger "
                    "values move the analysis further toward the observation.")
    reason: str = Field(
        description="One or two sentences explaining the decision, citing the signals used.")


LLM_SYSTEM_PROMPT = """\
You are the assimilation agent in a wildfire digital twin. Each cycle, a new fire-perimeter \
observation arrives and an Ensemble Kalman Filter (EnKF) combines it with a FARSITE fire-spread \
forecast ensemble. You decide HOW the observation is used in that update.

Key idea: a large disagreement between forecast and observation means one of them is wrong, \
but the disagreement alone cannot tell you which. The forecast-independent checks \
(contains_previous, growth_plausible, agrees_with_firms) are the evidence about the \
observation itself; the innovation and the drift streak are evidence about the forecast.

Actions:
- assimilate: normal EnKF update.
- inflate_and_assimilate: widen the forecast spread by `inflation` (1.0-3.0) first, so the update \
moves further toward the observation. For an observation that is surprising but believable.
- skip: ignore this observation and keep the forecast. Only when the independent evidence says \
the observation is wrong and nothing (such as a wind shift) explains it.
- reinitialize_from_obs: discard the forecast and restart from the observation. For when the \
forecast has kept missing over several cycles, so the model has drifted.

Notes:
- FIRMS = NASA satellite active-fire hotspot detections (not companies).
- Guidance from the rule-based policy: normalized_innovation above about 2 is surprising; \
3 or more consecutive misses suggests the model has drifted. These are guidelines, not hard \
limits — weigh the signals together.
- A synthetic_perimeter is not a real observation: it was generated because the real one was \
rejected, and it is less reliable than a mapped_perimeter.
"""

LLM_HUMAN_PROMPT = """\
Signals for this cycle:
{signals}

Context:
{context}

Choose the action (and the inflation factor if you inflate) and explain why in one or two \
sentences citing the signals."""


def _describe_signals(signals: Signals, innovation_available: bool) -> str:
    """Each signal as 'name: value — description', so the model gets the meaning, not just
    the number."""
    lines = []
    for name, field in Signals.model_fields.items():
        value = getattr(signals, name)
        if name == "normalized_innovation" and not innovation_available:
            value = "unavailable (no forecast ensemble this cycle)"
        lines.append(f"- {name}: {value} — {field.description}")
    lines.append(f"- independent_check_failed: {signals.independent_check_failed} — True if any "
                 "forecast-independent check says the observation is wrong.")
    return "\n".join(lines)


def _describe_context(state: TwinState) -> str:
    wx = state.get("weather_verdict")
    prev = state.get("assim_verdict")          # still last cycle's verdict at this point
    history = [round(x, 2) for x in (state.get("drift_history") or [])]
    return "\n".join([
        f"- weather agent: {getattr(wx, 'action', 'n/a')} — {getattr(wx, 'reason', '')}",
        f"- recent normalized innovations (oldest first): {history or 'none yet'}",
        f"- previous assimilation decision: {getattr(prev, 'action', 'none (first cycle)')}",
        f"- cycle: {state.get('step', '?')}",
    ])


class LLMPolicy(AssimilationPolicy):
    """
    The LLM picks the action and the inflation factor from the same signals the RulePolicy
    sees; code fills in vsize (source preset) and wsize (baseline), exactly as RulePolicy
    does, so the LLM-vs-rule comparison differs only in the judgment.

    If the LLM call fails or returns something unparsable, the RulePolicy decision is used
    and the reason says so — a run never stops because the LLM endpoint was busy.

    After each decide(): `last_tokens` (for the DecisionRecord) and `last_fallback`.
    """

    def __init__(self, llm, use_source_vsize: bool = True,
                 wsize_m: float = BASELINE["wsize_m"]):
        from langchain_core.prompts import ChatPromptTemplate
        from orchestrator.llm import STRUCTURED_OUTPUT_METHOD

        self.use_source_vsize = use_source_vsize
        self.wsize_m = wsize_m
        self.fallback = RulePolicy(use_source_vsize=use_source_vsize, wsize_m=wsize_m)
        prompt = ChatPromptTemplate.from_messages(
            [("system", LLM_SYSTEM_PROMPT), ("human", LLM_HUMAN_PROMPT)])
        self.chain = prompt | llm.with_structured_output(
            LLMAssimilationChoice, method=STRUCTURED_OUTPUT_METHOD, include_raw=True)
        self.last_tokens = 0
        self.last_fallback = False

    def decide(self, state: TwinState, signals: Signals) -> AssimilationDecision:
        from orchestrator.llm import total_tokens

        vsize = (VSIZE_PRESETS.get(signals.source, BASELINE["vsize_m"])
                 if self.use_source_vsize else BASELINE["vsize_m"])
        innovation_available = (state.get("cache") or {}).get("innovation_available", True)
        self.last_tokens, self.last_fallback = 0, False

        try:
            out = self.chain.invoke({
                "signals": _describe_signals(signals, innovation_available),
                "context": _describe_context(state),
            })
            self.last_tokens = total_tokens(out.get("raw"))
            choice = out.get("parsed")
            if choice is None:
                raise ValueError(f"unparsable LLM output: {out.get('parsing_error')}")
        except Exception as e:
            self.last_fallback = True
            d = self.fallback.decide(state, signals)
            return d.model_copy(update={
                "reason": f"LLM unavailable ({type(e).__name__}); rule fallback: {d.reason}"})

        return AssimilationDecision(
            action=choice.action,
            vsize_m=vsize,
            wsize_m=self.wsize_m,
            # Inflation only means something with inflate_and_assimilate.
            inflation=choice.inflation if choice.action == "inflate_and_assimilate" else 1.0,
            reason=choice.reason,
        )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_assimilation_agent(policy: AssimilationPolicy, load_ensemble=None):
    """
    Return a LangGraph node function that applies `policy`.

    The returned function:
      1. Loads the forecast ensemble (TwinState holds a path, not arrays)
      2. Computes normalized_innovation and fills it into signals
      3. Runs policy.decide(state, signals)
      4. Applies the guardrail on 'skip'
      5. Appends the innovation to drift_history
      6. Returns {"assim_verdict", "signals", "drift_history", "log"}

    Args:
        policy: FixedPolicy() or RulePolicy()
        load_ensemble: callable(state) -> (forecast_ensemble, obs_vec). Defaults to
            default_load_ensemble, which reads state["ensemble_path"]. Inject a stub
            to test policy logic without FARSITE.

    Example
    -------
        nodes["assimilation"] = make_assimilation_agent(RulePolicy())
    """
    policy_name = type(policy).__name__
    do_load = load_ensemble if load_ensemble is not None else default_load_ensemble

    def _node(state: TwinState) -> dict:
        t0 = time.perf_counter()

        signals = state.get("signals")
        if signals is None:
            # The observation agent normally produces these. Build them here if a
            # placeholder observation node ran instead, so this agent still works.
            from orchestrator.agents.observation import compute_signals
            signals = compute_signals(state)

        # 1-2. Innovation needs the ensemble. vsize for the normalisation comes from
        #      the source preset, the same value the decision will use.
        vsize_for_norm = VSIZE_PRESETS.get(signals.source, BASELINE["vsize_m"])
        forecast, obs_vec = do_load(state)
        innovation = compute_innovation(state, forecast, obs_vec, vsize_for_norm)
        if innovation is not None:
            signals = signals.model_copy(update={"normalized_innovation": innovation})

        # Tell the policy whether the innovation is real, so a missing one is never
        # read as a low one. Written to the cache so the policy signature stays
        # (state, signals) for every policy, LLM ones included.
        cache = dict(state.get("cache") or {})
        cache["innovation_available"] = innovation is not None

        # 3. Policy decides
        verdict = policy.decide({**state, "cache": cache}, signals)

        # 4. Guardrail: never skip when there is nothing else to fall back on.
        #    Skipping the only observation of a cycle leaves the state un-updated;
        #    on the first cycle that means it is never initialised at all.
        overridden = False
        if verdict.action == "skip" and state.get("prev_obs_path") is None:
            verdict = AssimilationDecision(
                action="assimilate",
                vsize_m=max(verdict.vsize_m, VSIZE_PRESETS["synthetic_perimeter"]),
                wsize_m=verdict.wsize_m,
                inflation=1.0,
                reason=("Guardrail: skipping was requested but there is no established "
                        "state to keep, so the observation is assimilated with a widened "
                        "observation error instead."),
            )
            overridden = True

        # 5. Drift history: a disagreement is an innovation above the surprise
        #    threshold. compute_signals counts the tail of this list.
        history = list(state.get("drift_history") or [])
        if innovation is not None:
            history = (history + [innovation])[-DRIFT_HISTORY_LEN:]

        # 6. Log
        record = DecisionRecord(
            sim_time=state.get("clock") or datetime.utcnow(),
            node="assimilation",
            policy=policy_name,
            inputs={
                "source": signals.source,
                "normalized_innovation": (round(innovation, 4)
                                          if innovation is not None else None),
                "ensemble_available": forecast is not None,
                "independent_check_failed": signals.independent_check_failed,
                "wind_shift": signals.wind_shift,
                "weather_action": getattr(state.get("weather_verdict"), "action", None),
                "consecutive_disagreements": signals.consecutive_disagreements,
                # LLMPolicy only: True if the LLM call failed and the rule decided instead.
                "llm_fallback": getattr(policy, "last_fallback", None),
            },
            decision=verdict.model_dump(),
            overridden_by_guardrail=overridden,
            agent_latency_s=round(time.perf_counter() - t0, 4),
            tokens=getattr(policy, "last_tokens", 0) or 0,
        )

        return {
            "assim_verdict": verdict,
            "signals": signals,
            "drift_history": history,
            "cache": cache,
            "log": [record],
        }

    _node.__name__ = f"assimilation_agent[{policy_name}]"
    return _node
