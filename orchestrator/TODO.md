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
- [x] Answer: it must be the forecast `zkphat`. Built from the prior, the innovation is
      (obs - prior) rather than (obs - forecast), so FARSITE's prediction never enters it, and the
      state/observation cross-covariance is estimated from prior-space anomalies while the gain is
      applied to forecast-space ones. Measured on a forecast that under-predicts (prior 1499 m,
      forecast 1799 m, obs 2598 m, spread 400 m): analysis error 781 m before, 477 m after. With a
      perfect forecast the fixed version correctly leaves it alone. Fixed in `src_fixes.patch`;
      **[H] please confirm before the rerun**, since it changes every analysis
- [x] [H] Answer: wind speed units — **mph** end to end, with no conversion anywhere.
      `query_weather_for_timestep` returns mph (`RAWS_UNITS='English'`), FARSITE is given the same
      numbers, and `members.csv` logged them unchanged. The column was named `wind_speed_ms` by
      mistake — it always held mph (baseline step 0 mean 8.78 is mph, not m/s). Renamed to
      `wind_speed_mph`; any existing CSV's `wind_speed_ms` column is already mph, so read it as such
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

A rejected observation does **not** skip the forecast. It is replaced by a
synthetic perimeter (`synthesize` step): the last accepted perimeter grown by the fire's observed spread
rate, stretched downwind (Anderson 1983 length-to-breadth). It is tagged `source="synthetic_perimeter"`,
so assimilation trusts it less (vsize 500 m). If there is no accepted perimeter yet, the cycle goes
straight to `update`.