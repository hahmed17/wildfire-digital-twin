# Orchestrator checklist

**Pivot (2026-11-18):** multi-agent pipeline. Simple agents make decisions about the **data**
(observation, weather, assimilation). The LangGraph graph is the orchestrator: routing is plain
conditional edges, no LLM organizer. Resource decisions (N, resolution, workers) stay fixed at
`BASELINE` for now. Design reference: `orch.py` (resource parts are now "later").

**Status (2026-11-23):** the whole pipeline runs end to end with three interchangeable policy
families, chosen with one flag: `replay(policies="fixed" | "rule" | "llm", scenario=...)`.
135 tests pass. What is missing is a **real FARSITE run** — with the stub FARSITE every cycle
looks surprising, so the three families cannot be compared meaningfully yet.

Owners: **[T]** Taina · **[H]** Hena.

## 0. Setup  (done)
- [x] [T] Conda env `orchestrator`, `pip install -e .` (pyproject.toml)
- [x] [H] `environment.yml` + `setup_env.sh`
- [x] [T] `langchain-openai` added to `environment.yml`; `NRP_LLM_API_KEY=` added to
      `user_info_template.env` — **everyone puts their own NRP key in `.env`** (gitignored)

## §1 state.py — data types  (done)
- [x] [T] `ObservationEvent`, `OperatorRequest`, `ForecastPlan`, `AssimilationDecision`,
      `VSIZE_PRESETS`, `BASELINE`, `Signals`, `DecisionRecord`, `TwinState`
- [x] [H] Verdict types `ObservationVerdict`, `WeatherVerdict` + `TwinState` keys `obs_verdict`,
      `weather_verdict`, `assim_verdict`; cache keys `weather_last`, `weather_prev`, `good_obs`

## §2 adapter.py — the only file that calls `src/`  (done, except the live test)
- [x] [T] `load_observation`, `fetch_weather`, `sample_winds`, `run_ensemble` wrapper
- [x] [H] `enkf_update` wrapper (inflation by scaling `zkphat` anomalies), `save_state` / `load_state`
- [ ] [H] **Live smoke test**: `adapter.run_ensemble` with N=4, workers=4 on her machine (has `landscape.lcp`)

## Hena: EnKF / FARSITE side (`src/`)
- [x] [H] Predicted observation now built from the forecast `zkphat` (was the prior `xkhat`);
      model noise added to members; covariance from valid members only. Applied in `src/enkf.py`
      (the old note about `src_fixes.patch` is obsolete). **Changes every analysis → rerun the baseline**
- [x] [H] Wind units: **mph** end to end; `members.csv` column renamed `wind_speed_mph`
- [~] [H] Native `inflation` in `enkf_update`: not needed, the adapter does it
- [ ] [H] Define reinitialize: which covariance `X` to restart with (open question in `steps.py`,
      `reinit_cov`; affects published numbers)
- [ ] [H] Save each step's forecast ensemble to `.npz` from the baseline run
- [ ] [H] Optional: `query_weather_for_timestep` reports when it used the fallback

## §3 checks.py — signals, pure functions, no LLM  (done)
- [x] [H] `normalized_innovation`, `contains_previous`, `growth_plausible`,
      `load_hotspots` + `agrees_with_firms` (FIRMS), `wind_shift`

## §6 Agents — Fixed + Rule + LLM policies for each  (done)
Each agent = menu (pydantic) + `*Policy` ABC with `decide(...)` + guardrail + `DecisionRecord`,
wrapped by `make_*_agent(policy, ...)` into a LangGraph node.
- [x] [H] Observation agent (`agents/observation.py`): accept / reject. Guardrail: always accept
      the first observation. Fixed + Rule policies
- [x] [H] Weather agent (`agents/weather.py`): normal / wind_shift_detected / fallback_weather.
      Guardrail: a fetch failure is always `fallback_weather`. Fixed + Rule policies
- [x] [H] Assimilation agent (`agents/assimilation.py`): assimilate / inflate_and_assimilate /
      skip / reinitialize_from_obs. Guardrail: never skip without an established state. Fixed + Rule
- [x] [T] **LLM policies for all three** (`LLMPolicy` in each file). Same pattern everywhere:
      - the LLM makes only the judgment (accept/reject; the weather label; action + inflation);
        code fills in the facts (vsize from source preset, wsize, `used_fallback`) and the
        `ForecastPlan` is RulePolicy's, so LLM and rule arms differ only in the judgment
      - the prompt explains every signal (from the `Signals` field descriptions); rule thresholds
        are given as guidance, not hard limits; the injected-fault label is never shown
      - the weather LLM sees the actual wind numbers (this window, previous window, change)
      - if the LLM call fails, the RulePolicy decision is used and logged (`llm_fallback`)
      - tokens are logged in every `DecisionRecord` (were always 0)

## §7 LLM setup — `orchestrator/llm.py`  (done)
- [x] [T] One fixed model for every experiment: NRP `gpt-oss` (OpenAI-compatible endpoint,
      `ChatOpenAI`), temperature 0, structured output `json_schema`. Chosen 2026-11-23 over
      `gemma` and `qwen3-small` (fastest, fewest tokens, consistent answers)
- [x] [T] Model + date recorded in each LLM run's `config.json` (NRP can repoint the alias:
      run all LLM experiments within a short window)

## §8 Pipeline organizer — `graph.py` + `routing.py` + `steps.py`  (done)
- [x] [H] `receive → observation → [synthesize] → weather → forecast → assimilation → update`;
      a rejected observation is replaced by a synthetic perimeter (`synthetic.py`)
- [x] [T] Fix: `synthesize` now recomputes the signals for the synthetic perimeter. Before, the
      assimilation agent judged it by the *rejected* observation's signals (vsize 200 instead of
      500). Regression test `test_synthetic_replacement_is_judged_as_synthetic`.
      **Any run with a rejected observation made before this fix must be redone**
- [x] [T] Fix: durations use `time.perf_counter()` (wall-clock `time.time()` went backwards on
      WSL → negative latencies → random test failures)

## §9 replay.py + §10 metrics.py  (done)
- [x] [H] Scenarios: `clean`, `shifted_obs`, `stale_obs`, `mislabeled_obs`, `wind_shift`,
      `weather_failure`; scoring against the true perimeter (IoU, area error); per-run outputs
- [x] [T] `policies="llm"` option (all three LLM policies sharing one model)

## To discuss with Hena
- [ ] Give the LLMs the **numbers behind the checks** (containment fraction, implied spread rate,
      fraction of hotspots inside) instead of only True/False — needs `checks.py` / `Signals` changes
- [ ] `normalized_innovation`: `spread` is a norm over all 40 coordinates but `vsize` is a
      per-coordinate std, so the denominator is too small and the score too large → "surprising"
      too often. Check on a real FARSITE run
- [ ] The observation agent (and weather in high wind) also changes the `ForecastPlan` — a
      resource decision, which the plan said to keep fixed for now. Intended?
- [ ] Reinitialize covariance (see her list above)

## Next: experiments
- [ ] [H] Live FARSITE smoke test, then rerun the baseline with the EnKF fixes
- [ ] [T+H] Run `fixed` / `rule` / `llm` × the 6 scenarios with real FARSITE
      (LLM runs within a short window; same seed across arms)
- [ ] [T+H] Results table for the paper: accuracy (IoU, area error), decisions per agent,
      guardrail overrides, LLM fallbacks, latency and tokens

## Later (resource decisions)
- [ ] Cost model from `members.csv` / `steps.csv` (needs the parameter sweep)
- [ ] Resource agent (N, resolution, workers) + plan guardrail
- [ ] An LLM organizer only if routing becomes ambiguous (it would replace `routing.py` only)
