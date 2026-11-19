"""
Placeholder agents and steps, for testing the pipeline wiring before the real agents exist.

Each placeholder follows the same rule as the real nodes: it takes the state and returns
only the keys it changed, including a DecisionRecord in "log". Swap them for the real
agents in build_pipeline(nodes) without changing graph.py or routing.py.
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from orchestrator.state import DecisionRecord, TwinState


class PlaceholderVerdict(BaseModel):
    """Stands in for ObservationVerdict / WeatherVerdict / AssimilationDecision.
    Routing only reads `.action`."""
    action: str
    reason: str = "placeholder"


def placeholder_agent(node: str, key: str, action: str | Callable[[TwinState], str]):
    """An agent that always returns the same verdict (or one computed by `action(state)`).

    Args:
        node: step name, logged in DecisionRecord.node ("observation", "weather", ...)
        key: TwinState key for the verdict ("obs_verdict", "weather_verdict", "assim_verdict")
        action: fixed action, or a function of the state (e.g. reject injected observations)
    """
    def agent(state: TwinState) -> dict:
        chosen = action(state) if callable(action) else action
        verdict = PlaceholderVerdict(action=chosen)
        record = DecisionRecord(
            sim_time=state["clock"], node=node, policy="placeholder",
            inputs={"event": state["current"].model_dump(mode="json") if state.get("current") else None},
            decision=verdict.model_dump(), overridden_by_guardrail=False, agent_latency_s=0.0)
        return {key: verdict, "log": [record]}
    agent.__name__ = f"{node}_placeholder"
    return agent


def placeholder_weather(weather: dict | None = None):
    """Weather placeholder: verdict 'use', and stores `weather` as the cached last window
    (the real weather agent fetches it via adapter.fetch_weather and caches it the same way)."""
    base = placeholder_agent("weather", "weather_verdict", "use")
    def agent(state: TwinState) -> dict:
        out = base(state)
        if weather is not None:
            out["cache"] = {**state["cache"], "weather_last": weather}
        return out
    agent.__name__ = "weather_placeholder"
    return agent


def placeholder_forecast(state: TwinState) -> dict:
    """Pretends to run the ensemble: records the fixed baseline plan, no FARSITE."""
    from orchestrator.state import BASELINE, ForecastPlan
    plan = ForecastPlan(ensemble_size=BASELINE["ensemble_size"], resolution_m=BASELINE["resolution_m"],
                        workers=BASELINE["workers"], reason="baseline (placeholder forecast)")
    return {"plan": plan, "ensemble_path": f"placeholder_ensemble_step{state['step']}.npz"}


def placeholder_nodes(obs_action="use", weather: dict | None = None, assim_action="assimilate") -> dict:
    """A complete `nodes` dict for build_pipeline using placeholders plus the real plain steps
    (receive, synthesize, update)."""
    from orchestrator import steps
    return {
        "receive": steps.receive,
        "observation": placeholder_agent("observation", "obs_verdict", obs_action),
        "synthesize": steps.synthesize,
        "weather": placeholder_weather(weather),
        "forecast": placeholder_forecast,
        "assimilation": placeholder_agent("assimilation", "assim_verdict", assim_action),
        "update": steps.update,
    }
