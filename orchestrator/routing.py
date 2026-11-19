"""
§8 routing.py — the organizer's exceptions to the fixed order.

The pipeline order lives in graph.PIPELINE. These functions are the only places where the
flow branches. Each takes the state and returns the name of the next step. They are plain
rules today; a smarter organizer (LLM, resource-aware) only needs to replace these functions.
"""
from __future__ import annotations

from typing import Literal

from orchestrator.state import TwinState


def _action(verdict) -> str | None:
    """`.action` of a verdict (pydantic model or dict); None if there is no verdict yet."""
    if verdict is None:
        return None
    return verdict.get("action") if isinstance(verdict, dict) else getattr(verdict, "action", None)


def after_observation(state: TwinState) -> Literal["synthesize", "weather"]:
    """A rejected observation is replaced by a synthetic one, so the forecast still runs."""
    return "synthesize" if _action(state.get("obs_verdict")) == "reject" else "weather"


def after_synthesize(state: TwinState) -> Literal["weather", "update"]:
    """Continue with the synthetic observation. If none could be made (no earlier accepted
    perimeter to grow), there is nothing to forecast toward: go straight to update."""
    current = state.get("current")
    made = current is not None and current.source == "synthetic_perimeter"
    return "weather" if made else "update"


def after_update(state: TwinState) -> Literal["receive", "end"]:
    """Loop while observations are waiting; stop when none are left."""
    return "receive" if state.get("pending") else "end"
