"""
§8 graph.py — the pipeline organizer (LangGraph wiring, no LLM).

One cycle per observation:

    receive → observation ─┬─→ weather → forecast → assimilation → update ─┬─→ receive (more events)
                           └─→ synthesize ─┘ (rejected obs replaced)       └─→ end

The fixed order is PIPELINE; the branches are the functions in routing.py.
Every step follows one rule: it takes the state and returns only the keys it changed,
e.g. `def observation_agent(state) -> dict: return {"obs_verdict": v, "log": [record]}`.
"""
from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from orchestrator import routing
from orchestrator.state import TwinState

# Main order of one cycle. "synthesize" sits off the main path (only after a reject).
PIPELINE = ["receive", "observation", "weather", "forecast", "assimilation", "update"]
OFF_PATH = ["synthesize"]
ALL_STEPS = PIPELINE + OFF_PATH

# Branch points: step -> (routing function, {routing result: next step})
BRANCHES: dict[str, tuple[Callable, dict]] = {
    "observation": (routing.after_observation, {"synthesize": "synthesize", "weather": "weather"}),
    "synthesize":  (routing.after_synthesize,  {"weather": "weather", "update": "update"}),
    "update":      (routing.after_update,      {"receive": "receive", "end": END}),
}


def build_pipeline(nodes: dict[str, Callable[[TwinState], dict]], checkpointer=None):
    """Compile the organizer graph.

    Args:
        nodes: one function per step name in ALL_STEPS. Each takes the TwinState and returns
               a dict with only the keys it changed. Agents are passed already bound to their
               policy, e.g. `functools.partial(observation_agent, policy=RulePolicy())`.
        checkpointer: optional LangGraph checkpointer to keep every intermediate state,
               e.g. memory_checkpointer() (registers our types for loading).

    Returns:
        A compiled LangGraph graph. Run it with run_pipeline(graph, state).
    """
    missing = [s for s in ALL_STEPS if s not in nodes]
    extra = [s for s in nodes if s not in ALL_STEPS]
    if missing or extra:
        raise ValueError(f"build_pipeline: missing steps {missing}, unknown steps {extra}")

    g = StateGraph(TwinState)
    for name in ALL_STEPS:
        g.add_node(name, nodes[name])

    g.add_edge(START, PIPELINE[0])
    for a, b in zip(PIPELINE, PIPELINE[1:]):      # fixed order, except at branch points
        if a not in BRANCHES:
            g.add_edge(a, b)
    for step, (fn, targets) in BRANCHES.items():
        g.add_conditional_edges(step, fn, targets)

    return g.compile(checkpointer=checkpointer)


# Our pydantic types stored in TwinState. LangGraph checkpointers only load registered types
# (unregistered ones warn now and will be blocked in a future version). Add new verdict types here.
CHECKPOINT_TYPES = [
    ("orchestrator.state", "ObservationEvent"), ("orchestrator.state", "OperatorRequest"),
    ("orchestrator.state", "ForecastPlan"), ("orchestrator.state", "AssimilationDecision"),
    ("orchestrator.state", "Signals"), ("orchestrator.state", "DecisionRecord"),
    ("orchestrator.placeholders", "PlaceholderVerdict"),
]


def memory_checkpointer():
    """In-memory checkpointer (keeps every intermediate state) that can load our types."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def run_pipeline(graph, state: TwinState, config: dict | None = None) -> TwinState:
    """Run until no observations are left.

    LangGraph stops any run after `recursion_limit` steps (default 25), which a few
    observations already exceed (7 steps per cycle). The limit is set from the number of
    pending observations so long replays aren't cut off.
    """
    n_events = len(state.get("pending") or [])
    cfg = {"recursion_limit": len(ALL_STEPS) * (n_events + 1) + 10}
    if config:
        cfg.update(config)
    return graph.invoke(state, cfg)
