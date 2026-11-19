"""
§8 plain steps — the non-agent nodes of the pipeline (no decisions about data).

Each takes the TwinState and returns only the keys it changed.
  receive     pop the next observation, advance the simulated clock, clear last cycle's verdicts
  synthesize  replace a rejected observation with a grown copy of the last accepted perimeter
  forecast    run the FARSITE ensemble through adapter.py           (contract only, see below)
  update      remember accepted observations, append this cycle's metrics
"""
from __future__ import annotations

import time
from pathlib import Path

from orchestrator.state import DecisionRecord, ObservationEvent, TwinState
from orchestrator.synthetic import make_synthetic_event


def _action(verdict) -> str | None:
    if verdict is None:
        return None
    return verdict.get("action") if isinstance(verdict, dict) else getattr(verdict, "action", None)


def receive(state: TwinState) -> dict:
    """Take the next observation by arrival time and start a new cycle."""
    pending = sorted(state["pending"], key=lambda e: e.arrival_time)
    event, rest = pending[0], pending[1:]
    return {
        "current": event,
        "pending": rest,
        "clock": event.arrival_time,
        "step": state["step"] + 1,
        # verdicts belong to one cycle: clear them so routing never reads a stale one
        "obs_verdict": None, "weather_verdict": None, "assim_verdict": None, "signals": None,
    }


def synthesize(state: TwinState) -> dict:
    """Replace the rejected observation with a synthetic perimeter grown from the last
    accepted one (see synthetic.py), so the forecast and assimilation still run.
    The synthetic event has source="synthetic_perimeter", so the assimilation step trusts
    it less (VSIZE_PRESETS: 500 m vs 200 m for a mapped perimeter)."""
    t0 = time.time()
    rejected = state["current"]
    good = [ObservationEvent.model_validate(e) for e in state["cache"].get("good_obs", [])]
    out_dir = Path(state["cache"].get("run_dir", "runs")) / "synthetic"
    event, details = make_synthetic_event(rejected, good, state["cache"].get("weather_last"), out_dir)
    record = DecisionRecord(
        sim_time=state["clock"], node="synthesize", policy="rule",
        inputs={"rejected": rejected.model_dump(mode="json"),
                "obs_verdict": _verdict_dump(state.get("obs_verdict"))},
        decision={"synthetic": event.model_dump(mode="json") if event else None, **details},
        overridden_by_guardrail=False, agent_latency_s=time.time() - t0)
    out = {"log": [record]}
    if event is not None:
        out["current"] = event
    return out


def forecast(state: TwinState) -> dict:
    """Contract: run the FARSITE ensemble for this cycle and return
        {"plan": ForecastPlan(**BASELINE), "ensemble_path": "<.npz with the ensemble>"}

    Steps, once adapter.load_state / save_state exist ([T] in TODO.md):
        state_vec, cov_X = adapter.load_state(state["state_path"])
        ens = adapter.run_ensemble(state_vec, cov_X, plan, state["cache"]["weather_last"],
                                   start_time=<analysis time>, dt=<obs_time - analysis time>, rng=...)
        save ens.raw to an .npz and return its path (TwinState keeps paths, not arrays)
    Tests use a placeholder in its place.
    """
    raise NotImplementedError("forecast: needs adapter.load_state/save_state (see docstring)")


def update(state: TwinState) -> dict:
    """End of a cycle: remember accepted observations and log this cycle's outcome.
    (Saving the EnKF analysis belongs here too once adapter.save_state exists.)"""
    current = state.get("current")
    obs_action = _action(state.get("obs_verdict"))
    accepted = (current is not None and current.source != "synthetic_perimeter"
                and obs_action != "reject")
    cache = dict(state["cache"])
    if accepted:   # last two accepted perimeters: what synthetic.py grows from
        cache["good_obs"] = (cache.get("good_obs", []) + [current.model_dump(mode="json")])[-2:]
    metrics = {
        "step": state["step"],
        "clock": state["clock"].isoformat(),
        "obs_time": current.obs_time.isoformat() if current else None,
        "source": current.source if current else None,
        "synthetic": bool(current and current.source == "synthetic_perimeter"),
        "obs_action": obs_action,
        "weather_action": _action(state.get("weather_verdict")),
        "assim_action": _action(state.get("assim_verdict")),
    }
    return {"cache": cache,
            "prev_obs_path": current.path if accepted else state.get("prev_obs_path"),
            "metrics": [metrics]}


def _verdict_dump(v):
    if v is None or isinstance(v, dict):
        return v
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else str(v)
