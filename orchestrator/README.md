# Orchestrator

A [LangGraph](https://github.com/langchain-ai/langgraph) pipeline that wraps the EnKF fire-spread assimilation loop with agent-based decision making. Each assimilation cycle, three agent nodes decide (1) whether to trust a new perimeter observation, (2) whether weather conditions warrant special treatment, and (3) how to update the fire state — then a set of plain nodes do the actual work (pop the next event, run FARSITE, save results).

---

## Prerequisites

Run the one-time setup from the **repo root** (not from inside `orchestrator/`):

```bash
bash setup_env.sh
```

This creates `~/envs/wildfire-twin`, installs the repo with `pip install -e .`, and registers the **Python (wildfire-twin)** Jupyter kernel. See the top-level README for details.

After setup, verify the import works:

```bash
python -c "from orchestrator import graph, steps, synthetic; print('ok')"
```

---

## Quick start

```python
from datetime import datetime, timezone
from orchestrator.graph import build_pipeline, run_pipeline, memory_checkpointer
from orchestrator.placeholders import make_placeholder_nodes
from orchestrator.state import ObservationEvent, OperatorRequest, initial_state

# 1. Describe the observations to replay (sorted by arrival time internally)
events = [
    ObservationEvent(
        obs_time=datetime(2025, 1, 23, 12, 0, tzinfo=timezone.utc),
        arrival_time=datetime(2025, 1, 23, 12, 5, tzinfo=timezone.utc),
        source="mapped_perimeter",
        path="data/border2_inputs/border_2_perimeters.geojson",
        feature_id="1",
    ),
    # ... more events
]

# 2. Build the graph (placeholder agents = accept everything, for testing)
nodes = make_placeholder_nodes()
graph = build_pipeline(nodes, checkpointer=memory_checkpointer())

# 3. Run
state = initial_state(events, OperatorRequest(deadline_min=30.0), state_path="outputs/state.npz")
final = run_pipeline(graph, state)

# 4. Inspect the per-cycle log
for m in final["metrics"]:
    print(m)
```

Or open `test_orchestrator.ipynb` for a walkthrough with the Border 2 perimeters.

---

## Pipeline overview

One cycle runs per observation event. The fixed spine is:

```
receive → observation ──┬──→ weather → forecast → assimilation → update ──┬──→ receive (more events)
                        └──→ synthesize ──┘ (rejected obs replaced)       └──→ end
```

| Node | Kind | What it does |
|------|------|--------------|
| `receive` | plain | Pops the next `ObservationEvent` from `pending`, advances the clock, clears last cycle's verdicts |
| `observation` | **agent** | Decides whether to trust the new perimeter (`accept` / `reject`) |
| `synthesize` | plain | If the observation was rejected, grows a synthetic perimeter from the last accepted one |
| `weather` | **agent** | Checks conditions for the forecast interval; can flag a wind shift |
| `forecast` | plain | Runs the FARSITE ensemble via `adapter.py` *(stub — see below)* |
| `assimilation` | **agent** | Chooses the EnKF update action (`assimilate`, `inflate_and_assimilate`, `skip`, `reinitialize_from_obs`) |
| `update` | plain | Records accepted observations in `cache["good_obs"]`, appends cycle metrics to `state["metrics"]` |

**Branches** (defined in `routing.py`):

- After `observation`: if `obs_verdict.action == "reject"` → go to `synthesize`; otherwise → `weather`
- After `synthesize`: if a synthetic perimeter was produced → `weather`; if not (no prior accepted perimeter) → `update`
- After `update`: if `pending` is non-empty → `receive`; otherwise → `end`

---

## State schema (`TwinState`)

`TwinState` is a `TypedDict` — LangGraph passes it from node to node. Each node receives the full state and **returns only the keys it changed**:

```python
def my_node(state: TwinState) -> dict:
    return {"obs_verdict": verdict, "log": [record]}  # only what changed
```

| Key | Type | Written by | Purpose |
|-----|------|-----------|---------|
| `request` | `OperatorRequest` | caller | Wall-clock budget (`deadline_min`) and priority |
| `clock` | `datetime` | `receive` | Simulated time — current observation's arrival time |
| `step` | `int` | `receive` | Cycle counter (1-indexed) |
| `state_path` | `str` | caller / `update` | Path to `.npz` holding the analysis vector and covariance `X` |
| `prev_obs_path` | `str \| None` | `update` | Path to previous accepted perimeter (containment check) |
| `pending` | `list[ObservationEvent]` | caller → `receive` | Observations not yet processed |
| `current` | `ObservationEvent \| None` | `receive`, `synthesize` | The event being processed this cycle |
| `plan` | `ForecastPlan \| None` | `observation` or `weather` agent | FARSITE settings chosen for this cycle |
| `ensemble_path` | `str \| None` | `forecast` | Path to `.npz` with the forecast ensemble |
| `signals` | `Signals \| None` | `observation` agent | Pre-computed checks (innovation, FIRMS agreement, containment, …) |
| `obs_verdict` | `Any \| None` | `observation` agent | Has `.action`: `"accept"` or `"reject"` |
| `weather_verdict` | `Any \| None` | `weather` agent | Has `.action` and signals wind shift etc. |
| `assim_verdict` | `AssimilationDecision \| None` | `assimilation` agent | Chooses update action, `vsize_m`, `wsize_m` |
| `drift_history` | `list[float]` | `assimilation` agent | Recent normalized innovations (tracks systematic drift) |
| `cache` | `dict` | any node | Shared scratch: `"good_obs"`, `"weather_last"`, `"run_dir"` |
| `log` | `list[DecisionRecord]` | any node | **Append-only** (LangGraph reducer). Every agent logs here |
| `metrics` | `list[dict]` | `update` | **Append-only**. One dict per cycle (see `update` in `steps.py`) |

The `Annotated[list, operator.add]` on `log` and `metrics` means LangGraph **appends** new items rather than replacing the whole list — nodes return `{"log": [new_record]}`, not the full log.

---

## Module reference

### `state.py`
Pure data types. No FARSITE or geo imports — policies, tests, and the routing logic all import this without touching the simulation stack.

Key classes: `ObservationEvent`, `OperatorRequest`, `ForecastPlan`, `AssimilationDecision`, `Signals`, `DecisionRecord`, `TwinState`. The `initial_state()` helper builds a valid starting state.

### `graph.py`
Wires the LangGraph `StateGraph`. The two functions you call directly:

- **`build_pipeline(nodes, checkpointer=None)`** — compiles the graph. `nodes` is a `dict[str, Callable]` mapping each step name to its implementation. The step names are `PIPELINE + OFF_PATH = ["receive", "observation", "weather", "forecast", "assimilation", "update", "synthesize"]`. Raises `ValueError` if any name is missing or unknown.
- **`run_pipeline(graph, state, config=None)`** — invokes the graph, setting `recursion_limit` automatically based on the number of pending events so long replays are not cut off.
- **`memory_checkpointer()`** — an in-memory checkpointer pre-registered with the project's Pydantic types. Pass it to `build_pipeline` to keep every intermediate state for debugging.

### `routing.py`
Three branch functions that return the name of the next node. They read verdicts from the state but make no decisions themselves. If you want smarter routing (e.g. skip weather when the agent already flagged a wind shift), change these functions.

| Function | Reads | Returns |
|----------|-------|---------|
| `after_observation(state)` | `obs_verdict.action` | `"synthesize"` or `"weather"` |
| `after_synthesize(state)` | `current.source` | `"weather"` or `"update"` |
| `after_update(state)` | `pending` | `"receive"` or `"end"` |

### `steps.py`
Implementations of the plain (non-agent) nodes: `receive`, `synthesize`, `forecast`, `update`. Each returns only the keys it changed.

> **`forecast` is a stub.** It raises `NotImplementedError`. Its docstring describes what a full implementation does once `adapter.load_state` / `adapter.save_state` exist. For now, replace it with a placeholder (see `placeholders.py`).

### `synthetic.py`
`make_synthetic_event(rejected, good_obs, weather, out_dir)` — grows the last accepted perimeter using a spread rate derived from the wind speed and the Anderson (1983) length-to-breadth ratio. Returns an `ObservationEvent` with `source="synthetic_perimeter"` so the assimilation step trusts it less (`vsize_m = 500` vs `200` for a mapped perimeter).

### `adapter.py`
Bridges `src/enkf.py` and `src/farsite.py` to the LangGraph state dict. Converts between `TwinState` paths and the numpy arrays the EnKF expects.

### `placeholders.py`
`make_placeholder_nodes()` returns a nodes dict with stub agents that always return a fixed "accept" verdict. Use for testing the graph routing without running FARSITE.

---

## Running the tests

```bash
# from the repo root:
python -m pytest orchestrator/test_pipeline.py -v
```

The test suite uses `make_placeholder_nodes()` so it runs without FARSITE. It checks:
- the graph compiles without errors
- `receive` correctly pops the next event and clears verdicts
- the routing branches (`after_observation`, `after_synthesize`, `after_update`) return the right next step
- `update` appends to `cache["good_obs"]` only for accepted observations
- a full replay of two synthetic events produces the expected `metrics` list

---

## How to add a new agent node

Agent nodes are ordinary Python functions that return a dict. Here is a minimal example — an observation agent that checks the innovation and rejects stale observations:

**1. Write the function** (e.g. in a new file `orchestrator/agents/observation.py`):

```python
from orchestrator.state import TwinState, DecisionRecord
import time

def observation_agent(state: TwinState) -> dict:
    t0 = time.time()
    current = state["current"]
    signals = state.get("signals")  # may be None on first cycle

    # Simple rule: reject if observation is > 6 hours old
    if current and current.obs_age_min > 360:
        action = "reject"
        reason = f"Observation is {current.obs_age_min:.0f} min old (> 360 min threshold)"
    else:
        action = "accept"
        reason = "Observation passes age check"

    verdict = {"action": action, "reason": reason}
    record = DecisionRecord(
        sim_time=state["clock"], node="observation", policy="rules",
        inputs={"source": current.source if current else None,
                "obs_age_min": current.obs_age_min if current else None},
        decision=verdict,
        overridden_by_guardrail=False,
        agent_latency_s=time.time() - t0,
    )
    return {"obs_verdict": verdict, "log": [record]}
```

**2. Add it to the nodes dict** when calling `build_pipeline`:

```python
from orchestrator.graph import build_pipeline
from orchestrator.placeholders import make_placeholder_nodes
from orchestrator.agents.observation import observation_agent

nodes = make_placeholder_nodes()
nodes["observation"] = observation_agent     # replace the placeholder
graph = build_pipeline(nodes)
```

**3. Test the routing** by checking that `after_observation` returns the right branch:

```python
from orchestrator.routing import after_observation

# rejected obs → synthesize
assert after_observation({"obs_verdict": {"action": "reject"}}) == "synthesize"
# accepted obs → weather
assert after_observation({"obs_verdict": {"action": "accept"}}) == "weather"
```

The same pattern applies to `weather` (key `weather_verdict`) and `assimilation` (key `assim_verdict`). The routing functions in `routing.py` only read `.action`, so any dict or Pydantic model with an `action` field works.

---

## Known stubs

| Item | Status | Notes |
|------|--------|-------|
| `forecast` node | `NotImplementedError` | Needs `adapter.load_state` / `adapter.save_state`; replace with a placeholder for testing |
| LLM agents | Not yet wired | Pydantic schemas (`ForecastPlan`, `AssimilationDecision`, `Signals`) are designed for structured LLM output |
| `adapter.py` | Partial | Bridges defined; `load_state` / `save_state` not yet implemented |
| Parameter-adaptive policy | Not yet implemented | Will use the sweep CSVs from `border2_experiment.ipynb` to pick `(N, resolution, workers)` given `deadline_min` |
