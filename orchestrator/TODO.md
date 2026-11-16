# Orchestrator checklist

Design reference: `orch.py`. Work top to bottom; check items off as we understand and finish them.

## 0. Setup
- [x] Pick a conda env and install `requirements.txt` + `langgraph` + `pydantic`
- [x] Confirm `src/` imports work from the orchestrator folder (`import enkf, geometry, farsite`)

## §1 state.py — data types
- [x] Decide: keep the drafted `state.py` and review it, or restart from the stub
- [x] `ObservationEvent` (obs_time vs arrival_time, source, path, injected)
- [x] `OperatorRequest` (horizon, deadline)
- [x] `ForecastPlan` (N, resolution, workers) + allowed ranges
- [x] `AssimilationDecision` (assimilate / inflate / skip / reinit, vsize, inflation)
- [x] `VSIZE_PRESETS` and `BASELINE` values (baseline: N=300, 150 m, 12 workers, vsize 300, wsize 100)
- [x] `Signals` (what the checks report)
- [x] `DecisionRecord` (what gets logged per decision)
- [x] `TwinState` (what flows between graph nodes)

## §2 adapter.py — the only file that calls Hena's `src/`
- [ ] Read `enkf.run_ensemble`, `enkf.enkf_update`, `enkf.step` and understand inputs/outputs
- [ ] `load_observation` (perimeter → state vector via `geometry.geom_to_state`)
- [ ] `fetch_weather` (wrap `firemap`, same fallback as the notebook)
- [ ] `run_ensemble` wrapper (takes a `ForecastPlan`)
- [ ] `enkf_update` wrapper (takes vsize, wsize, inflation)
- [ ] `save_state` / `load_state`

## §5 executors.py — run live or replay saved runs
- [ ] `Executor` interface
- [ ] `TraceExecutor` reading `outputs/Border_2_Real/members.csv` (fake data first)
- [ ] `LiveExecutor` calling the adapter (real FARSITE)

## §6 policies — FixedPolicy first
- [ ] `Policy` interface (`plan_forecast`, `decide_assimilation`)
- [ ] `FixedPolicy` = today's notebook baseline

## §8 graph.py — LangGraph loop
- [ ] Nodes: receive → plan → forecast → check → assimilate → update
- [ ] Route: loop while events remain, else END
- [ ] Run end to end with FixedPolicy + TraceExecutor

## §10 metrics.py — results table
- [ ] Per step: time-to-forecast, deadline hit, area error / IoU vs observed
- [ ] Per run: core-seconds, guardrail overrides, agent latency/tokens

## §3 checks.py — consistency + independent checks (no LLM)
- [ ] `normalized_innovation`
- [ ] `contains_previous`
- [ ] `growth_plausible`
- [ ] `agrees_with_firms`
- [ ] `wind_shift`

## §4 costmodel.py — fitted from Hena's logs
- [ ] Fit runtime model from `members.csv` / `steps.csv`
- [ ] `estimate(plan)` and `best_plan_within(deadline)`
- [ ] Note: only N=300 @ 150 m so far — needs the parameter sweep for more

## §7 guardrails.py
- [ ] `clamp_plan` (fit within deadline)
- [ ] `clamp_assimilation` (skip only if an independent check failed; streak ≥ 3 → reinit)

## §6 policies — the smarter ones
- [ ] `RulePolicy` (thresholds)
- [ ] `LLMPolicy` (structured output + always through guardrails)

## §9 replay.py — Border 2 scenarios
- [ ] `build_events` from `data/border2_inputs/`
- [ ] Scenarios: clean, bad_obs, wind_shift, failures, deadlines
- [ ] `replay(policy, scenario)` writes results JSON

## Final
- [ ] Run all policies × scenarios
- [ ] Live FARSITE comparison on 3–4 steps
- [ ] Results table for the paper
