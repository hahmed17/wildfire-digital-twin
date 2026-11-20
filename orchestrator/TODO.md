# Orchestrator checklist

**Pivot (2026-11-18):** multi-agent pipeline. Simple agents make decisions about the **data**
(observation, weather, assimilation). The LangGraph graph is the orchestrator: routing is plain
conditional edges, no LLM organizer. Resource decisions (N, resolution, workers) stay fixed at
`BASELINE` for now. Design reference: `orch.py` (resource parts are now "later").

Owners: **[T]** Taina · **[H]** Hena. Contracts both sides code against: `state.py` (types,
`Signals`) and the function signatures listed below.

## 0. Setup
- [x] [T] Conda env `orchestrator`, `pip install -e .` (pyproject.toml)
- [ ] [H] Same setup in her env: `pip install -e .` once at the repo root

## §1 state.py — data types  (done)
- [x] [T] `ObservationEvent`, `OperatorRequest`, `ForecastPlan`, `AssimilationDecision`,
      `VSIZE_PRESETS`, `BASELINE`, `Signals`, `DecisionRecord`, `TwinState`
- [x] [T] Add a verdict type per agent (`ObservationVerdict`, `WeatherVerdict`)
- [x] [H] `TwinState` keys `obs_verdict`, `weather_verdict`, `assim_verdict` (now typed
      `ObservationVerdict` / `WeatherVerdict` / `AssimilationDecision`; routing still only reads
      `.action`, so plain dicts and placeholders keep working). Cache keys: `weather_last` (last weather window, for
      "reuse last window" and synthetic perimeters), `good_obs` (last two accepted observations)

## §2 adapter.py — the only file that calls `src/`
- [x] [T] `load_observation`
- [x] [T] `fetch_weather`
- [x] [T] `sample_winds` + `run_ensemble` wrapper (tested with fake FARSITE)
- [x] [T] `enkf_update` wrapper (inflation applied by scaling `zkphat` anomalies around their
      mean, so `src/enkf.py` needs no change)
- [x] [T] `save_state` / `load_state` (+ `mph_to_ms` / `MPH_TO_MS`: firemap returns mph,
      `members.csv` logs m/s)
- [ ] [H] Live smoke test: `adapter.run_ensemble` with N=4, workers=4 on her machine (has `landscape.lcp`)

## Hena: EnKF / FARSITE side (in `src/`, her files)
- [ ] [H] Answer: is the predicted observation `ykhat` (from the prior `xkhat`) intended, or should it be the forecast `zkphat`?
- [x] [H] Answer: wind speed units — **mph** end to end. `query_weather_for_timestep` returns mph
      (`RAWS_UNITS='English'`), FARSITE expects mph, and `members.csv` converts to m/s for logging only
- [~] [H] `enkf_update`: an `inflation` parameter. Not needed for now — `adapter.enkf_update` does it
      by pre-scaling the `zkphat` anomalies. Still worth doing natively if the update changes
- [ ] [H] Define reinitialize: new state = observation, and what covariance `X` to restart with
- [ ] [H] Save each step's forecast ensemble (`zkphat`, `xkhat`, `zero_samples`) to `.npz` from the
      baseline run, so agents can be developed without rerunning FARSITE
- [ ] [H] Optional: `query_weather_for_timestep` reports when it used the fallback (flag or exception)

## §3 checks.py — signals, pure functions, no LLM  (done)
Each returns one field of `Signals`. Inputs are state vectors / shapely polygons in EPSG:5070.
- [x] [T] `normalized_innovation(ensemble, obs_vec, vsize_m, valid_mask=None) -> float`
- [x] [T] `contains_previous(perimeter, prev_perimeter, tol=0.9) -> bool`
- [x] [T] `growth_plausible(perimeter, prev_perimeter, dt_minutes, wind_speed_mph) -> bool`
- [x] [T] `load_hotspots(path, start, end)` + `agrees_with_firms(perimeter, hotspots, buffer_m=500) -> bool | None`
      Verified against `data/border2_inputs/firms_hotspots.csv` (334 detections): reads the raw FIRMS
      `acq_date` + `acq_time` (HHMM int) shape as well as a single datetime column. Enable per run with
      `make_observation_agent(policy, hotspot_path=...)`. This is the only check that brings in data
      from outside the perimeter series, so it can catch a perimeter that is self-consistent but wrong
- [x] [T] `wind_shift(weather_prev, weather_now, deg_thresh=45, speed_thresh_mph=10) -> bool`

## §6 Agents — one per data decision, each with Fixed + Rule policies  (done)
Each agent = menu (pydantic) + policy interface + guardrail + `DecisionRecord`.
All three live in `orchestrator/agents/` and share one shape: a `*Policy` ABC with
`decide(...)`, a `FixedPolicy` (reproduces the baseline) and a `RulePolicy`, wrapped by
`make_*_agent(policy, ...)` into a LangGraph node. Every external call the node makes is
injectable (`fetch`, `load_ensemble`), so policy logic is testable without FARSITE.
- [x] [T] Observation agent (`agents/observation.py`): accept / reject + `ForecastPlan`.
      Guardrail: always accept the first observation (no previous perimeter to check against).
      `compute_signals()` runs every forecast-independent check
- [x] [T] Weather agent (`agents/weather.py`): normal / wind_shift_detected / fallback_weather.
      Owns the weather cache (`weather_last` + `weather_prev`), refines the plan in high wind.
      Guardrail: a fetch failure is always flagged as `fallback_weather`
- [x] [T] Assimilation agent (`agents/assimilation.py`): assimilate / inflate_and_assimilate /
      skip / reinitialize_from_obs; streak >= 3 -> reinit; vsize from `VSIZE_PRESETS[source]`.
      Guardrail: never skip when there is no established state to keep

## §8 Pipeline organizer — `graph.py` + `routing.py`  [H]
No LLM: the graph is the organizer. Order lives in one list; exceptions live in `routing.py`.

**Rule every step follows (agents and plain steps alike):** one function that takes the state
and returns only the keys it changed, e.g.
`def observation_agent(state: TwinState) -> dict: return {"obs_verdict": v, "log": [record]}`

**Decision (2026-11-18):** a rejected observation does **not** skip the forecast. It is replaced by a
synthetic perimeter (`synthesize` step): the last accepted perimeter grown by the fire's observed spread
rate, stretched downwind (Anderson 1983 length-to-breadth). It is tagged `source="synthetic_perimeter"`,
so assimilation trusts it less (vsize 500 m). If there is no accepted perimeter yet, the cycle goes
straight to `update`.

- [x] [H] Verdict keys in `TwinState` (see §1)
- [x] [H] `graph.py`: `PIPELINE = ["receive", "observation", "weather", "forecast", "assimilation", "update"]`
      + off-path `synthesize`; `build_pipeline(nodes)`; `run_pipeline()` sets LangGraph's recursion limit
      from the number of events (default 25 stops a replay after ~3 observations);
      `memory_checkpointer()` registers our pydantic types
- [x] [H] `routing.py`: `after_observation` → `"synthesize"` if reject else `"weather"`;
      `after_synthesize` → `"weather"` (or `"update"` if nothing to grow); `after_update` → `"receive"` / `"end"`
- [x] [H] `synthetic.py`: `grow_perimeter`, `make_synthetic_event` (writes GeoJSON, returns an `ObservationEvent`)
- [x] [H] Plain steps (`steps.py`): `receive` (pop next event, advance clock, clear verdicts),
      `synthesize`, `update` (remember accepted obs, append metrics)
- [ ] [H] `forecast` step: contract written in `steps.forecast`; needs `adapter.load_state` / `save_state` [T]
- [ ] [H] `update`: also save the EnKF analysis once `adapter.save_state` exists
- [x] [H] Tests with **placeholder agents** (`placeholders.py`, `tests/test_pipeline.py`, 14 tests): order,
      reject → synthesize → forecast still runs, first-obs reject, loop ends, 10-event replay, checkpointing
- [~] [T] Swap in the real agents: all three plug into `build_pipeline(nodes)` and a 4-cycle replay
      runs with Fixed and with Rule policies (verified: identical inputs, Rule inflates on the
      wind-shift cycle where Fixed does a plain update). Still to do: run it against real FARSITE
      instead of `placeholder_forecast` — blocked on the `forecast` step below
- [x] [T] Resolved without splitting `Signals`: it is filled progressively. The observation agent
      writes every forecast-independent check (and a one-cycle-lagged `wind_shift`, which is the right
      question anyway: did the wind shift during the interval that produced this perimeter). The weather
      agent overwrites `wind_shift` for the window being forecast. The assimilation agent fills
      `normalized_innovation` once the ensemble exists, and sets `cache["innovation_available"]` so a
      missing innovation is never read as a low one
- Later: a smarter organizer (LLM / resource decisions) only replaces the functions in `routing.py`

## §9 replay.py — Border 2 scenarios  [H]
- [ ] [H] `build_events(fire, scenario) -> list[ObservationEvent]` from `data/border2_inputs/`
      (her selected rows + exact timestamps from the notebook)
- [ ] [H] Scenario injections: `bad_obs` (shift 2 km / stale / mislabeled), `wind_shift`, `failures`
- [ ] [T] `replay(policies, scenario)` runs the graph and writes results JSON

## §10 metrics.py
- [ ] [H] Accuracy: area error and IoU of the analysis vs the observed perimeter
- [ ] [T] Per run: decisions per agent, guardrail overrides, agent latency / tokens

## Then
- [ ] [T] LLM policy per agent (structured output, always through guardrails)
- [ ] [T+H] Run all policies × scenarios; results table for the paper

## Later (resource decisions)
- [ ] `executors.py` (TraceExecutor / LiveExecutor)
- [ ] `costmodel.py` from `members.csv` / `steps.csv` (needs the parameter sweep)
- [ ] Resource agent + `clamp_plan`; LLM organizer only if routing becomes ambiguous