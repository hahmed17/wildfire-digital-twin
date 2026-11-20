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

# 2. Build the graph. Start from placeholders, then swap in the agents you want.
from orchestrator.agents.observation  import make_observation_agent,  RulePolicy as ObsRule
from orchestrator.agents.weather      import make_weather_agent,      RulePolicy as WxRule
from orchestrator.agents.assimilation import make_assimilation_agent, RulePolicy as AsRule

nodes = make_placeholder_nodes()                 # supplies receive/synthesize/forecast/update
nodes["observation"]  = make_observation_agent(ObsRule())
nodes["weather"]      = make_weather_agent(WxRule())
nodes["assimilation"] = make_assimilation_agent(AsRule())
graph = build_pipeline(nodes, checkpointer=memory_checkpointer())

# 3. Run
state = initial_state(events, OperatorRequest(deadline_min=30.0), state_path="outputs/state.npz")
final = run_pipeline(graph, state)

# 4. Inspect the per-cycle log
for m in final["metrics"]:
    print(m)
```

Swap `RulePolicy` for `FixedPolicy` in all three to get the baseline configuration: every observation accepted, baseline FARSITE settings, a plain EnKF update with `vsize=300, wsize=100`. That reproduces `outputs/Border_2_Real` and is the control the agentic runs are compared against.

Or open `test_orchestrator.ipynb` for a walkthrough with the Border 2 perimeters, and `demo_placeholders.ipynb` for a per-node walkthrough with placeholders.

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
| `observation` | **agent** | Decides whether to trust the new perimeter (`accept` / `reject`) and picks the `ForecastPlan` |
| `synthesize` | plain | If the observation was rejected, grows a synthetic perimeter from the last accepted one |
| `weather` | **agent** | Fetches the window's weather, owns the weather cache, flags wind shifts and fallback weather |
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

> **`forecast` is a stub.** It raises `NotImplementedError`. Its docstring describes what a full implementation does; `adapter.load_state` / `adapter.save_state` now exist, so it is implementable. For now, replace it with a placeholder (see `placeholders.py`).

### `synthetic.py`
`make_synthetic_event(rejected, good_obs, weather, out_dir)` — grows the last accepted perimeter using a spread rate derived from the wind speed and the Anderson (1983) length-to-breadth ratio. Returns an `ObservationEvent` with `source="synthetic_perimeter"` so the assimilation step trusts it less (`vsize_m = 500` vs `200` for a mapped perimeter).

### `adapter.py`
The only module that imports `src/`. Bridges `src/enkf.py` and `src/farsite.py` to the LangGraph state dict, converting between `TwinState` paths and the numpy arrays the EnKF expects: `load_observation`, `fetch_weather`, `sample_winds`, `run_ensemble`, `enkf_update`, `save_state`, `load_state`.

Two things worth knowing. Wind speeds are **mph** end to end — `firemap.query_weather_for_timestep` returns mph because `config.RAWS_UNITS == 'English'`, and FARSITE expects the same; `mph_to_ms` exists only because `members.csv` logs m/s. And `enkf_update` applies covariance inflation itself, by scaling the `zkphat` anomalies around their mean before calling `enkf.enkf_update`, so `src/enkf.py` needs no inflation parameter.

### `checks.py`
Pure signal functions — no LLM, no FARSITE, no `src/` imports. Each one fills a field of `Signals`: `normalized_innovation` (the only forecast-dependent one), `contains_previous`, `growth_plausible`, `agrees_with_firms` / `load_hotspots`, `wind_shift`. All geometry is EPSG:5070 metres.

### `agents/`
One module per data decision, all three built to the same shape:

| Module | Verdict | Actions |
|--------|---------|---------|
| `agents/observation.py` | `ObservationVerdict` | `accept`, `reject` |
| `agents/weather.py` | `WeatherVerdict` | `normal`, `wind_shift_detected`, `fallback_weather` |
| `agents/assimilation.py` | `AssimilationDecision` | `assimilate`, `inflate_and_assimilate`, `skip`, `reinitialize_from_obs` |

Each module provides a `*Policy` ABC, a `FixedPolicy` that reproduces the baseline Border 2 run, a `RulePolicy` that uses the signals, and a `make_*_agent(policy, ...)` factory that wraps a policy into a LangGraph node. An LLM policy is a fourth subclass of the same ABC — the node does not change.

Every external call a node makes is injectable, so policy logic can be tested without FARSITE:

```python
nodes["weather"]      = make_weather_agent(RulePolicy(), fetch=lambda s, e: FAKE_WEATHER)
nodes["assimilation"] = make_assimilation_agent(RulePolicy(), load_ensemble=lambda st: (Z, obs))
```

**Guardrails** sit in the node, after the policy, and are recorded in `DecisionRecord.overridden_by_guardrail`:

| Agent | Guardrail |
|-------|-----------|
| observation | Always accept the first observation — with no previous perimeter, containment and growth checks are uninformative |
| weather | A fetch failure is always reported as `fallback_weather`, whatever the policy said |
| assimilation | Never `skip` when there is no established state to keep; assimilate with a widened `vsize_m` instead |

**How `Signals` gets filled.** Not all at once — the pipeline order forces it:

1. `observation` writes every forecast-independent check, plus `wind_shift` compared between `cache["weather_prev"]` and `cache["weather_last"]`. That is one cycle lagged, which is the right question: *did the wind shift during the interval that produced this perimeter?*
2. `weather` overwrites `wind_shift` with the value for the window about to be forecast, and rotates the weather cache.
3. `assimilation` fills `normalized_innovation` once the ensemble exists, and sets `cache["innovation_available"]` so a missing innovation is never mistaken for a low one.

### `placeholders.py`
`make_placeholder_nodes()` returns a nodes dict with stub agents that always return a fixed "accept" verdict. Use for testing the graph routing without running FARSITE.

---

## Running the tests

```bash
# from the repo root:
python -m pytest orchestrator/ -v          # 61 tests, ~1 s
```

Nothing here needs FARSITE, the network or any data files.

There are three notebooks and two test modules, covering the same ground at different
levels of detail:

| File | What it is |
|------|------------|
| `test_pipeline.py` | graph wiring, with placeholder agents |
| `test_agents.py` | the three real agents, with `fetch` / `load_ensemble` injected |
| `test_agents.ipynb` | the readable version of `test_agents.py` — every decision with the state shown before and after, then a Fixed-vs-Rule replay |
| `demo_placeholders.ipynb` | the graph itself, with placeholder agents: routing, branches, state handoff |
| `test_orchestrator.ipynb` | a walkthrough against the real Border 2 perimeters |

Start with `test_agents.ipynb` if you want to understand what the agents decide and why;
start with `demo_placeholders.ipynb` if you want to understand how the graph moves state around.

**`test_pipeline.py` (14 tests)** — the graph wiring, using `make_placeholder_nodes()`:
- the graph compiles without errors
- `receive` correctly pops the next event and clears verdicts
- the routing branches (`after_observation`, `after_synthesize`, `after_update`) return the right next step
- `update` appends to `cache["good_obs"]` only for accepted observations
- a full replay of two synthetic events produces the expected `metrics` list

**`test_agents.py` (43 tests)** — the three agents, with `fetch` and `load_ensemble` injected:
- each policy's decisions, including every branch of the assimilation rule set
- all three guardrails fire, and are recorded in `DecisionRecord.overridden_by_guardrail`
- the weather cache rotates so the next cycle can compare windows
- `vsize_m` follows `VSIZE_PRESETS[source]`
- a missing innovation is never read as a low one
- a 4-cycle replay where Fixed and Rule policies diverge on identical inputs — the comparison the experiment rests on

---

## How to add a new agent node

All three agents follow one pattern, so copy the nearest one rather than starting from scratch. The pattern is: a **policy** decides, a **node** does the I/O and enforces the guardrail.

**1. Define the menu** — a pydantic model in `state.py` with an `action` field. Write the field descriptions for a reader who has no context, because an LLM policy will be given exactly those strings:

```python
class ResourceVerdict(BaseModel):
    action: Literal["keep", "downscale"] = Field(
        description="'keep': run the plan as chosen. 'downscale': halve the ensemble "
                    "because the deadline is at risk.")
    reason: str = ""
```

**2. Write the policy ABC and at least two policies.** `FixedPolicy` must reproduce the baseline, so there is always a control to compare against:

```python
class ResourcePolicy(ABC):
    @abstractmethod
    def decide(self, state: TwinState, signals: Signals) -> ResourceVerdict: ...

class FixedPolicy(ResourcePolicy):
    def decide(self, state, signals):
        return ResourceVerdict(action="keep", reason="FixedPolicy: baseline.")

@dataclass
class RulePolicy(ResourcePolicy):
    margin: float = 0.8
    def decide(self, state, signals):
        ...
```

**3. Write the node factory.** The node owns everything the policy should not: loading data, the guardrail, the `DecisionRecord`. Keep external calls injectable so the policy stays testable without FARSITE:

```python
def make_resource_agent(policy: ResourcePolicy, estimate_cost=None):
    policy_name = type(policy).__name__
    do_estimate = estimate_cost or default_estimate_cost

    def _node(state: TwinState) -> dict:
        t0 = time.time()
        verdict = policy.decide(state, state["signals"])

        overridden = False
        if verdict.action == "downscale" and state["plan"].ensemble_size <= 8:
            verdict = ResourceVerdict(action="keep",
                reason="Guardrail: N is already at the floor.")
            overridden = True

        record = DecisionRecord(
            sim_time=state["clock"], node="resource", policy=policy_name,
            inputs={...}, decision=verdict.model_dump(),
            overridden_by_guardrail=overridden,
            agent_latency_s=round(time.time() - t0, 4))
        return {"resource_verdict": verdict, "log": [record]}

    _node.__name__ = f"resource_agent[{policy_name}]"
    return _node
```

**4. Wire it in.** A node on the existing spine needs no graph change:

```python
nodes = make_placeholder_nodes()
nodes["observation"] = make_resource_agent(RulePolicy())
graph = build_pipeline(nodes)
```

A *new* step needs three edits: add its name to `PIPELINE` (or `OFF_PATH`) in `graph.py`, add its verdict key to `TwinState` in `state.py`, and add the verdict class to `CHECKPOINT_TYPES` in `graph.py` so checkpointers can load it. Add a branch function to `routing.py` only if the step can change where the cycle goes next.

**5. Test the routing**, which reads nothing but `.action`:

```python
from orchestrator.routing import after_observation
assert after_observation({"obs_verdict": {"action": "reject"}}) == "synthesize"
assert after_observation({"obs_verdict": {"action": "accept"}}) == "weather"
