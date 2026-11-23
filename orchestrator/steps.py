"""
§8 plain steps — the non-agent nodes of the pipeline (no decisions about data).

Each takes the TwinState and returns only the keys it changed.
  receive     pop the next observation, advance the simulated clock, clear last cycle's verdicts
  synthesize  replace a rejected observation with a grown copy of the last accepted perimeter
  forecast    run the FARSITE ensemble through adapter.py
  update      execute the assimilation verdict, save the analysis, append cycle metrics

`forecast` and `update` reach FARSITE through adapter.py, which they import lazily so
this module stays importable (and testable with placeholders) without the simulation
stack. `seed_state` writes the initial state file the first forecast reads.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from orchestrator.state import DecisionRecord, ForecastPlan, ObservationEvent, TwinState
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
    t0 = time.perf_counter()
    rejected = state["current"]
    good = [ObservationEvent.model_validate(e) for e in state["cache"].get("good_obs", [])]
    out_dir = Path(state["cache"].get("run_dir", "runs")) / "synthetic"
    event, details = make_synthetic_event(rejected, good, state["cache"].get("weather_last"), out_dir)
    record = DecisionRecord(
        sim_time=state["clock"], node="synthesize", policy="rule",
        inputs={"rejected": rejected.model_dump(mode="json"),
                "obs_verdict": _verdict_dump(state.get("obs_verdict"))},
        decision={"synthetic": event.model_dump(mode="json") if event else None, **details},
        overridden_by_guardrail=False, agent_latency_s=time.perf_counter() - t0)
    out = {"log": [record]}
    if event is not None:
        out["current"] = event
        # The signals in the state describe the REJECTED observation (e.g. source
        # "mapped_perimeter", contains_previous=False). Recompute them for the synthetic
        # one, or the assimilation agent judges the replacement by the rejected
        # observation's signals and never applies the synthetic vsize (500 m).
        from orchestrator.agents.observation import compute_signals
        out["signals"] = compute_signals({**state, "current": event})
    return out


# FARSITE sub-step length. The experiment notebook uses 15 min on high-wind steps and
# 30 otherwise, because a fast front covers too much ground in one 30 min sub-step.
FARSITE_DT_NORMAL = 30
FARSITE_DT_HIGH_WIND = 15
HIGH_WIND_MPH = 25.0


def make_forecast(lcp_path: str | Path | None = None,
                  n_vertex: int = 20,
                  params: dict | None = None,
                  members_csv: str | Path | None = None,
                  fire_name: str = "Border_2",
                  run_id: str | None = None):
    """Build the `forecast` node: run the FARSITE ensemble for this cycle.

    The node loads the carried-forward state, runs the ensemble over the interval the
    weather agent already fetched, writes the ensemble to an .npz and returns its path.
    TwinState holds paths, not arrays, so the big matrices never enter the graph state.

    The returned node reads `state["plan"]` (from the observation agent, possibly refined
    by the weather agent) and `state["cache"]["weather_last"]` (from the weather agent),
    so both must have run first — which is the pipeline order.

    Args:
        lcp_path: FARSITE landscape file; defaults to adapter.LCP_PATH
        n_vertex: vertices per perimeter; must match the seeded state
        params: FARSITE temperature/humidity; defaults to adapter.FARSITE_PARAMS
        members_csv: per-member log, the same file enkf.run_ensemble writes in the notebook
        fire_name, run_id: carried into that log

    Returns:
        A node function returning {"ensemble_path", "cache", "log"}.
    """
    def _node(state: TwinState) -> dict:
        # Imported here, not at module scope: adapter pulls in enkf/farsite/geopandas,
        # so importing it eagerly would make steps.py unimportable without FARSITE.
        from orchestrator import adapter
        from orchestrator.agents.weather import forecast_window

        t0 = time.perf_counter()
        plan = state.get("plan")
        if plan is None:
            raise ValueError("forecast: state['plan'] is unset — the observation agent "
                             "must run first")
        weather = (state.get("cache") or {}).get("weather_last")
        if weather is None:
            raise ValueError("forecast: cache['weather_last'] is unset — the weather "
                             "agent must run first")

        state_path = Path(state["state_path"])
        if not state_path.exists():
            raise FileNotFoundError(
                f"forecast: no state at {state_path}. Seed it from the first perimeter "
                "with steps.seed_state(path, event) before the replay — that perimeter is "
                "the initial condition, not an observation, exactly as the notebook uses "
                "perimeters_gdf.iloc[0].")
        prior_state, prior_cov_X, rng = adapter.load_state(state_path)

        # Put the prior and the observation on a common vertex ordering BEFORE the
        # ensemble runs, which is what the notebook does:
        #     observation_state, current_state = align_states(
        #         [observation_state, current_state], vertex_count=N_VERTEX)
        # align_states matches vertex count *and* optimal rotation. Without it the EnKF
        # compares vertex i of the state against vertex i of the observation at a
        # different angular position, so the innovation measures vertex-ordering
        # mismatch rather than shape difference and the update is meaningless.
        # Both aligned vectors are saved with the ensemble so `update` uses exactly
        # the ones the forecast was driven from.
        from geometry import align_states

        current = state.get("current")
        if current is None:
            raise ValueError("forecast: state['current'] is unset — receive must run first")
        obs_vec = adapter.load_observation(current, n_vertex=n_vertex)
        obs_vec, prior_state = align_states([obs_vec, prior_state],
                                            vertex_count=n_vertex)

        # Same interval the weather agent fetched, so the winds match the forecast window.
        start, end = forecast_window(state)
        dt = end - start

        # checks.mean_speed skips missing readings; np.mean over a window with
        # gaps raises TypeError (float + NoneType).
        from orchestrator import checks
        mean_wind = checks.mean_speed(weather.get("wind_speeds"))
        farsite_dt = (FARSITE_DT_HIGH_WIND if mean_wind >= HIGH_WIND_MPH
                      else FARSITE_DT_NORMAL)

        ens = adapter.run_ensemble(
            state_vec=prior_state, cov_X=prior_cov_X, plan=plan, weather=weather,
            start_time=start, dt=dt, rng=rng, n_vertex=n_vertex,
            farsite_dt_minutes=farsite_dt,
            lcp_path=lcp_path if lcp_path is not None else adapter.LCP_PATH,
            params=params, members_csv=members_csv, fire_name=fire_name,
            step_idx=state["step"] - 1,       # notebook's t is 0-indexed
            run_id=run_id or (state.get("cache") or {}).get("run_id"),
        )

        # Persist the ensemble; the assimilation agent and update both read it back.
        run_dir = Path((state.get("cache") or {}).get("run_dir", "runs"))
        out = run_dir / "ensembles" / f"step{state['step']:02d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            zkphat_ensemble=ens.raw["zkphat_ensemble"],
            xkhat_ensemble=ens.raw["xkhat_ensemble"],
            ykhat_ensemble=ens.raw["ykhat_ensemble"],
            zero_samples=np.asarray(ens.raw["zero_samples"], dtype=int),
            wind_speeds=ens.wind_speeds,
            wind_directions=ens.wind_directions,
            ensemble_size=ens.plan.ensemble_size,
            resolution_m=ens.plan.resolution_m,
            workers=ens.plan.workers,
            # The aligned pair the ensemble was driven from. `update` reads these
            # rather than recomputing, so the analysis cannot drift out of step with
            # the forecast's vertex ordering.
            aligned_obs=np.asarray(obs_vec).reshape(-1, 1),
            aligned_prior=np.asarray(prior_state).reshape(-1, 1),
        )

        # The RNG advanced inside run_ensemble; keep it so the run stays reproducible.
        cache = {**(state.get("cache") or {}), "rng_state": rng.bit_generator.state}

        record = DecisionRecord(
            sim_time=state["clock"], node="forecast", policy="rule",
            inputs={"window_start": start.isoformat(), "window_end": end.isoformat(),
                    "dt_min": dt.total_seconds() / 60.0,
                    "N": plan.ensemble_size, "resolution_m": plan.resolution_m,
                    "workers": plan.workers, "mean_wind_mph": round(mean_wind, 2),
                    "farsite_dt_minutes": farsite_dt},
            decision={"ensemble_path": str(out), "n_valid": ens.n_valid,
                      "ensemble_wall_s": round(ens.wall_s, 1)},
            overridden_by_guardrail=False, agent_latency_s=round(time.perf_counter() - t0, 3))

        return {"ensemble_path": str(out), "cache": cache, "log": [record]}

    _node.__name__ = "forecast"
    return _node


# Default-configured node, so `nodes["forecast"] = steps.forecast` keeps working.
# Calling make_forecast(...) directly is how a run overrides the LCP path or the logs.
forecast = make_forecast()


def seed_state(path: str | Path, event: ObservationEvent, n_vertex: int = 20,
               cov_scale: float = 1000.0, seed: int | None = None) -> None:
    """Write the initial state file the first forecast reads.

    The first perimeter is the initial condition, not an observation: the notebook sets
    `current_state = geom_to_state(perimeters_gdf.geometry.iloc[0])` and `X = 1000 * I`,
    then assimilates the remaining perimeters. A replay should seed this file from that
    first perimeter and pass only the later perimeters as events.

    Args:
        path: .npz to write, matching the TwinState `state_path`
        event: ObservationEvent pointing at the initial perimeter
        n_vertex: vertices to resample it to; must match the forecast node
        cov_scale: initial covariance is cov_scale * I (notebook uses 1000.0)
        seed: RNG seed, saved with the state so a replay is reproducible
    """
    from orchestrator import adapter

    state_vec = adapter.load_observation(event, n_vertex=n_vertex)
    cov_X = cov_scale * np.eye(2 * n_vertex)
    adapter.save_state(path, state_vec, cov_X, np.random.default_rng(seed))



REINIT_COV_INITIAL = 1000.0


def apply_assimilation(state: TwinState, n_vertex: int = 20,
                       reinit_cov: str = "observation") -> dict:
    """Execute the assimilation agent's verdict and save the new analysis.

    The agent decides, this does the work — the same split as everywhere else in the
    pipeline. Handles all four actions:

      assimilate              EnKF update with the verdict's vsize/wsize
      inflate_and_assimilate  the same, with the ensemble spread widened first
      skip                    keep the forecast, apply no correction
      reinitialize_from_obs   discard the forecast, restart from the observation

    Returns a dict of details for the cycle's metrics, or {} when there is nothing to
    apply (no verdict, or no ensemble because the forecast was a placeholder or skipped).
    """
    verdict = state.get("assim_verdict")
    action = _action(verdict)
    if action is None:
        return {}

    ens_path = state.get("ensemble_path")
    current = state.get("current")
    if not ens_path or not Path(ens_path).exists() or current is None:
        return {"assim_applied": False,
                "assim_note": "no ensemble on disk; nothing to apply"}

    # Only now does FARSITE become necessary, so the import stays below the early
    # returns: a placeholder forecast leaves no ensemble, and those runs must keep
    # working on a machine without the simulation stack (see test_pipeline.py).
    from orchestrator import adapter

    _, prior_cov_X, rng = adapter.load_state(state["state_path"])
    data = np.load(ens_path, allow_pickle=True)
    n_states = 2 * n_vertex

    # The forecast aligned the observation and the prior to a common vertex ordering
    # and saved both; reuse them so the update matches the ensemble exactly. Fall back
    # to recomputing only for an ensemble written before that was saved.
    if "aligned_obs" in data and "aligned_prior" in data:
        obs_vec = data["aligned_obs"]
        prior_state = data["aligned_prior"]
    else:
        from geometry import align_states
        obs_vec = adapter.load_observation(current, n_vertex=n_vertex)
        prior_state, _, _ = adapter.load_state(state["state_path"])
        obs_vec, prior_state = align_states([obs_vec, prior_state],
                                            vertex_count=n_vertex)

    if action == "skip":
        # No correction: the analysis is the forecast mean over the valid members.
        Z = data["zkphat_ensemble"]
        zero = set(np.asarray(data["zero_samples"], dtype=int).tolist())
        keep = [i for i in range(Z.shape[1]) if i not in zero]
        Z_valid = Z[:, keep] if keep else Z
        analysis = Z_valid.mean(axis=1, keepdims=True)
        new_cov_X = np.cov(Z_valid) if Z_valid.shape[1] > 1 else prior_cov_X

    elif action == "reinitialize_from_obs":
        analysis = np.asarray(obs_vec).reshape(n_states, 1)
        scale = (verdict.vsize_m ** 2 if reinit_cov == "observation"
                 else REINIT_COV_INITIAL)
        new_cov_X = scale * np.eye(n_states)

    else:   # assimilate / inflate_and_assimilate
        plan = state.get("plan")
        ens = adapter.EnsembleResult(
            raw={"zkphat_ensemble": data["zkphat_ensemble"],
                 "xkhat_ensemble": data["xkhat_ensemble"],
                 "ykhat_ensemble": data["ykhat_ensemble"],
                 "zero_samples": np.asarray(data["zero_samples"], dtype=int).tolist(),
                 "member_results": []},
            plan=plan if plan is not None else _plan_from_npz(data),
            wind_speeds=data["wind_speeds"], wind_directions=data["wind_directions"],
            wall_s=0.0)
        analysis, new_cov_X = adapter.enkf_update(
            ens=ens, obs_vec=obs_vec, prior_state=prior_state,
            prior_cov_X=prior_cov_X, vsize_m=verdict.vsize_m,
            wsize_m=verdict.wsize_m, inflation=verdict.inflation,
            n_vertex=n_vertex, rng=rng)

    adapter.save_state(state["state_path"], analysis, new_cov_X, rng)

    return {"assim_applied": True,
            "vsize_m": verdict.vsize_m, "wsize_m": verdict.wsize_m,
            "inflation": verdict.inflation,
            "analysis_area_m2": _area_of(analysis),
            "observed_area_m2": _area_of(obs_vec)}


def _plan_from_npz(data) -> ForecastPlan:
    return ForecastPlan(ensemble_size=int(data["ensemble_size"]),
                        resolution_m=int(data["resolution_m"]),
                        workers=int(data["workers"]),
                        reason="reconstructed from the saved ensemble")


def _area_of(state_vec) -> float | None:
    """Polygon area (m^2) of a state vector, or None if it is not a usable polygon."""
    try:
        from shapely.geometry import Polygon
        from shapely import make_valid
        v = np.asarray(state_vec).ravel()
        poly = make_valid(Polygon(np.c_[v[0::2], v[1::2]]))
        return float(poly.area)
    except Exception:
        return None


def update(state: TwinState) -> dict:
    """End of a cycle: execute the assimilation verdict, save the analysis, remember
    accepted observations and log this cycle's outcome."""
    current = state.get("current")
    obs_action = _action(state.get("obs_verdict"))
    assim_details = apply_assimilation(state)
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
        **assim_details,
    }
    return {"cache": cache,
            "prev_obs_path": current.path if accepted else state.get("prev_obs_path"),
            "metrics": [metrics]}


def _verdict_dump(v):
    if v is None or isinstance(v, dict):
        return v
    return v.model_dump(mode="json") if hasattr(v, "model_dump") else str(v)