"""
Ensemble Kalman Filter (EnKF) module for fire data assimilation.
Implements EnKF with FARSITE forward model for wildfire state estimation.

Architecture
------------
run_ensemble(t, obs, ...)   – runs N FARSITE members in parallel, returns forecast ensemble
enkf_update(ensemble, obs, ...)  – pure EnKF math, returns analysis state + updated covariance
step(t, obs, ...)           – convenience wrapper: run_ensemble → enkf_update → checkpoint

The step() interface is what the agentic orchestrator calls. It can override N, resolution,
and n_workers at runtime based on its empirical cost/accuracy models.
"""
import os
import csv
import time
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from farsite import forward_pass_farsite_multistep, cleanup_farsite_outputs
from geometry import (
    state_to_geom, geom_to_state, align_states,
    resample_state_to_vertex_count, validate_geom
)
from config import FARSITE_TMP_DIR


# ============================================================================
# TOP-LEVEL WORKER FUNCTION (must be module-level for ProcessPoolExecutor)
# ============================================================================

def _run_one_member(args):
    """
    Run a single FARSITE ensemble member. Must be a top-level function so it
    is picklable by ProcessPoolExecutor.

    Args:
        args: dict with keys:
            s, seed, xkhat_col, ws, wd, observation_time, lcppath,
            dist_res, perim_res, params, n_vertex, initial_state_arr,
            run_id, fire_name, step_idx

    Returns:
        dict with keys: s, zkphat_col (or None), status, wall_s, cpu_s
    """
    import numpy as _np
    from farsite import forward_pass_farsite_multistep
    from geometry import state_to_geom, geom_to_state, align_states

    s           = args['s']
    xkhat_col   = _np.array(args['xkhat_col'])   # (n_states, 1)
    ws          = args['ws']
    wd          = args['wd']
    obs_time    = args['observation_time']
    lcppath     = args['lcppath']
    dist_res    = args['dist_res']
    perim_res   = args['perim_res']
    params      = args['params'] or {}
    n_vertex    = args['n_vertex']
    initial_state_arr = _np.array(args['initial_state_arr'])

    t0_wall = time.time()
    t0_cpu  = time.process_time()

    farsite_params = {
        'windspeed':     int(ws),
        'winddirection': int(wd),
        'temperature':   int(params.get('temperature', 66)),
        'humidity':      float(params.get('humidity', 0.08)),
        'dt':            args['dt'],
        'farsite_dt_minutes': args.get('farsite_dt_minutes', 30)
    }
    input_poly = state_to_geom(xkhat_col)
    start_time = obs_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        forward_geom = forward_pass_farsite_multistep(
            poly=input_poly,
            params=farsite_params,
            start_time=start_time,
            lcppath=lcppath,
            dist_res=dist_res,
            perim_res=perim_res,
        )
    except Exception as e:
        wall_s = time.time() - t0_wall
        cpu_s  = time.process_time() - t0_cpu
        return dict(s=s, zkphat_col=None, status=f'error:{e}',
                    wall_s=wall_s, cpu_s=cpu_s)

    wall_s = time.time() - t0_wall
    cpu_s  = time.process_time() - t0_cpu

    if forward_geom is None:
        return dict(s=s, zkphat_col=None, status='farsite_failed',
                    wall_s=wall_s, cpu_s=cpu_s)

    forward_state = geom_to_state(forward_geom)
    aligned = align_states([initial_state_arr, forward_state], vertex_count=n_vertex)
    return dict(s=s, zkphat_col=aligned[1], status='ok',
                wall_s=wall_s, cpu_s=cpu_s)


# ============================================================================
# RUN ENSEMBLE
# ============================================================================

def run_ensemble(
    initial_state, observation_time,
    n_states, n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    dist_res, perim_res, lcppath,
    X=None,           # (n_states, n_states) prior covariance — used for Cholesky perturbations
    params=None,
    n_workers=1,
    farsite_dt_minutes=30,
    # logging
    members_csv=None,
    fire_name='unknown',
    step_idx=0,
    run_id=None,
):
    """
    Run N FARSITE ensemble members in parallel and return the forecast ensemble.

    Args:
        initial_state:    Initial fire perimeter state vector (2*n_vertex, 1)
        observation_time: Datetime of observation (start of forecast window)
        n_states:         State dimension (2 * n_vertex)
        n_vertex:         Number of perimeter vertices
        n_samples:        Number of ensemble members (N)
        rng:              Numpy random number generator (used for perturbations)
        sampled_wslst:    Wind speed samples [n_samples]
        sampled_wdlst:    Wind direction samples [n_samples]
        dt:               Forecast time step (pandas Timedelta)
        dist_res:         FARSITE distance resolution (m)
        perim_res:        FARSITE perimeter resolution (m)
        lcppath:          Path to landscape (.lcp) file
        params:           Extra FARSITE params dict (temperature, humidity, …)
        n_workers:        ProcessPoolExecutor max_workers (1 = serial)
        members_csv:      Path to members.csv log (None = no logging)
        fire_name:        Fire identifier for logging
        step_idx:         Assimilation step index for logging
        run_id:           Sweep run identifier for logging

    Returns:
        dict with keys:
            xkhat_ensemble   (n_states, n_samples) – prior ensemble (with noise)
            zkphat_ensemble  (n_states, n_samples) – FARSITE forecast ensemble
            ykhat_ensemble   (n_states, n_samples) – predicted observations (xkhat + obs noise)
            zero_samples     list[int]             – member indices that failed
            member_results   list[dict]            – per-member timing + status
    """
    initial_state = np.asarray(initial_state).reshape(-1, 1)
    initial_state = resample_state_to_vertex_count(initial_state, n_vertex)
    n_states = 2 * n_vertex

    xkhat_ensemble  = np.zeros((n_states, n_samples))
    zkphat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble  = np.zeros((n_states, n_samples))

    # Resolve prior covariance — use caller's X if provided, else identity * 1000
    if X is None:
        X_prior = 1000.0 * np.eye(n_states)
    else:
        X_prior = np.asarray(X)
        if X_prior.shape != (n_states, n_states):
            X_prior = 1000.0 * np.eye(n_states)

    # Pre-generate all perturbations in the main process (reproducible)
    xkhat_perturbations = rng.normal(size=(n_states, n_samples))   # (n_states, N)

    # Build prior ensemble using Cholesky of X (main process, cheap)
    Xs = np.linalg.cholesky(X_prior)
    for s in range(n_samples):
        xkhat_ensemble[:, s:s+1] = initial_state + Xs @ xkhat_perturbations[:, s:s+1]

    # Build job list for workers
    jobs = []
    for s in range(n_samples):
        jobs.append(dict(
            s=s,
            seed=int(rng.integers(0, 2**31)),
            xkhat_col=xkhat_ensemble[:, s:s+1].tolist(),
            ws=float(sampled_wslst[s]),
            wd=float(sampled_wdlst[s]),
            observation_time=observation_time,
            lcppath=str(lcppath),
            dist_res=dist_res,
            perim_res=perim_res,
            params=params,
            n_vertex=n_vertex,
            initial_state_arr=initial_state.tolist(),
            dt=dt,
            farsite_dt_minutes=farsite_dt_minutes,
            run_id=run_id or str(uuid.uuid4())[:8],
            fire_name=fire_name,
            step_idx=step_idx,
        ))

    zero_samples   = []
    member_results = []
    result_map     = {}

    t_ensemble_start = time.time()

    if n_workers <= 1:
        # Serial path (keeps tqdm progress bar)
        for job in tqdm(jobs, desc="Running ensemble"):
            res = _run_one_member(job)
            result_map[res['s']] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one_member, job): job['s'] for job in jobs}
            for fut in tqdm(as_completed(futures), total=n_samples, desc="Running ensemble"):
                res = fut.result()
                result_map[res['s']] = res

    # Collect results in order
    for s in range(n_samples):
        res = result_map[s]
        member_results.append(res)
        if res['zkphat_col'] is None:
            zero_samples.append(s)
        else:
            zkphat_ensemble[:, s:s+1] = np.array(res['zkphat_col'])

    # Build predicted observation ensemble (obs noise on prior)
    # vsize handled in enkf_update; here we just store xkhat + unit noise placeholder.
    # enkf_update will re-add obs noise with the correct vsize.
    ykhat_ensemble = xkhat_ensemble.copy()   # obs model is identity; enkf_update adds noise

    # Logging
    if members_csv is not None:
        _append_members_csv(members_csv, member_results, jobs, n_samples,
                            dist_res, perim_res, n_workers, fire_name, step_idx, run_id)

    print(f"Ensemble done: {n_samples - len(zero_samples)}/{n_samples} valid "
          f"({time.time() - t_ensemble_start:.1f}s, {n_workers} workers)")

    return dict(
        xkhat_ensemble=xkhat_ensemble,
        zkphat_ensemble=zkphat_ensemble,
        ykhat_ensemble=ykhat_ensemble,
        zero_samples=zero_samples,
        member_results=member_results,
    )


# ============================================================================
# ENKF UPDATE (pure math, no FARSITE calls)
# ============================================================================

def enkf_update(
    ensemble,         # dict returned by run_ensemble
    initial_state,    # (2*n_vertex, 1) — used for obs alignment
    observation_state,
    X,                # (n_states, n_states) prior covariance
    n_states,
    n_output,
    n_vertex,
    n_samples,
    rng,
    vsize,            # observation noise std (m)
    wsize,            # model noise std (m)
):
    """
    Pure EnKF analysis step. No FARSITE calls.

    Args:
        ensemble:          Output dict from run_ensemble
        initial_state:     (2*n_vertex, 1) prior mean
        observation_state: (2*n_vertex, 1) observed perimeter
        X:                 (n_states, n_states) prior covariance matrix
        n_states:          State dimension (2 * n_vertex)
        n_output:          Output dimension (equals n_states)
        n_vertex:          Number of perimeter vertices
        n_samples:         Number of ensemble members
        rng:               Numpy RNG (for obs noise)
        vsize:             Observation noise std (m)
        wsize:             Model noise std (m)

    Returns:
        dict with keys:
            adjusted_state   (n_states, 1) – analysis mean
            X                (n_states, n_states) – updated covariance
            xkphat_ensemble  (n_states, n_samples) – analysis ensemble
            zkphat_ensemble  (n_states, n_samples) – forecast ensemble (passthrough)
            xkhat_ensemble   (n_states, n_samples) – prior ensemble (passthrough)
            ykhat_ensemble   (n_output, n_samples) – predicted obs ensemble
    """
    initial_state     = np.asarray(initial_state).reshape(-1, 1)
    observation_state = np.asarray(observation_state).reshape(-1, 1)
    initial_state     = resample_state_to_vertex_count(initial_state, n_vertex)
    observation_state = resample_state_to_vertex_count(observation_state, n_vertex)
    n_states = 2 * n_vertex
    n_output = n_states

    if isinstance(X, list):
        X = X[0]
    X = np.asarray(X)
    if X.shape != (n_states, n_states):
        X = 1000.0 * np.eye(n_states)

    xkhat_ensemble  = ensemble['xkhat_ensemble']
    zkphat_ensemble = ensemble['zkphat_ensemble']
    zero_samples    = ensemble['zero_samples']

    # Re-build ykhat with correct vsize obs noise
    ykhat_ensemble = xkhat_ensemble + rng.normal(0, scale=vsize, size=(n_output, n_samples))

    valid_mask = np.ones(n_samples, dtype=bool)
    valid_mask[zero_samples] = False
    Ne = int(valid_mask.sum())

    print(f"Valid samples: {Ne}, Failed samples: {len(zero_samples)}")
    if Ne <= 1:
        raise RuntimeError(f"Too few valid FARSITE samples ({Ne}/{n_samples}). EnKF cannot proceed.")

    # Slice valid ensembles
    Z = zkphat_ensemble[:, valid_mask]   # (n_states, Ne)
    Y = ykhat_ensemble[:, valid_mask]    # (n_output, Ne)

    zkphat_mean = Z.mean(axis=1, keepdims=True)
    ykhat_mean  = Y.mean(axis=1, keepdims=True)

    Z_anom = Z - zkphat_mean
    if float(wsize) > 0:
        Z_anom = Z_anom + rng.normal(0.0, float(wsize), size=Z_anom.shape)
    Y_anom = Y - ykhat_mean

    # Align observation
    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk = aligned_states[1]                           # (n_output, 1)
    innovation = yk - ykhat_mean                     # (n_output, 1)

    # Ensemble-space Kalman gain
    r2   = float(vsize) ** 2
    if r2 <= 0:
        raise ValueError("vsize must be > 0")
    rinv = 1.0 / r2

    S     = rinv * (Y_anom.T @ Y_anom) + (Ne - 1) * np.eye(Ne)
    eps   = 1e-10 * (np.trace(S) / S.shape[0])
    S_reg = S + eps * np.eye(Ne)

    rhs           = rinv * (Y_anom.T @ innovation)
    w             = np.linalg.solve(S_reg, rhs)
    adjusted_state = zkphat_mean + (Z_anom @ w)

    # Analysis ensemble
    xkphat_ensemble = np.zeros((n_states, n_samples))
    innovation_all  = yk - Y                                      # (n_output, Ne)
    rhs_all         = rinv * (Y_anom.T @ innovation_all)         # (Ne, Ne)
    W_all           = np.linalg.solve(S_reg, rhs_all)            # (Ne, Ne)
    xkphat_valid    = Z + (Z_anom @ W_all)                       # (n_states, Ne)
    xkphat_ensemble[:, valid_mask]  = xkphat_valid
    xkphat_ensemble[:, ~valid_mask] = adjusted_state

    # Update covariance
    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    ex = xkphat_ensemble - xkphat_mean
    X  = (ex @ ex.T) / (n_samples - 1) + 1e-10 * np.eye(n_states)

    # Validate / align analysis state geometry
    adjusted_geom  = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom)
    adjusted_state = align_states([initial_state, adjusted_state], vertex_count=n_vertex)[1]

    print("Calculating adjusted state (ensemble-space)")

    return dict(
        adjusted_state=adjusted_state,
        X=X,
        xkphat_ensemble=xkphat_ensemble,
        zkphat_ensemble=zkphat_ensemble,
        xkhat_ensemble=xkhat_ensemble,
        ykhat_ensemble=ykhat_ensemble,
    )


# ============================================================================
# STEP — orchestrator interface
# ============================================================================

def step(
    t,                   # int – assimilation step index
    initial_state,       # (2*n_vertex, 1)
    observation_state,   # (2*n_vertex, 1)
    observation_time,    # datetime
    X,                   # (n_states, n_states) prior covariance
    n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    vsize, wsize,
    dist_res, perim_res, lcppath,
    params=None,
    n_workers=1,
    farsite_dt_minutes=30,
    # logging
    members_csv=None,
    steps_csv=None,
    fire_name='unknown',
    run_id=None,
):
    """
    One full assimilation cycle: run_ensemble → enkf_update → optional checkpointing.

    This is the interface the agentic orchestrator calls. The orchestrator can override
    N (n_samples), resolution (dist_res / perim_res), and n_workers at each step based
    on its runtime cost and accuracy models.

    Returns:
        dict with all keys from enkf_update, plus:
            t_ensemble_s   wall time for FARSITE ensemble (s)
            t_enkf_s       wall time for EnKF update (s)
            n_valid        number of valid ensemble members
    """
    n_states = 2 * n_vertex
    run_id   = run_id or str(uuid.uuid4())[:8]

    # --- Ensemble forecast ---
    t0_ens = time.time()
    ens = run_ensemble(
        initial_state=initial_state,
        observation_time=observation_time,
        n_states=n_states,
        n_vertex=n_vertex,
        n_samples=n_samples,
        rng=rng,
        sampled_wslst=sampled_wslst,
        sampled_wdlst=sampled_wdlst,
        dt=dt,
        dist_res=dist_res,
        perim_res=perim_res,
        lcppath=lcppath,
        X=X,          # pass updated covariance for prior perturbations
        params=params,
        n_workers=n_workers,
        farsite_dt_minutes=farsite_dt_minutes,
        members_csv=members_csv,
        fire_name=fire_name,
        step_idx=t,
        run_id=run_id,
    )
    t_ensemble_s = time.time() - t0_ens

    # --- EnKF update ---
    t0_enkf = time.time()
    result = enkf_update(
        ensemble=ens,
        initial_state=initial_state,
        observation_state=observation_state,
        X=X,
        n_states=n_states,
        n_output=n_states,
        n_vertex=n_vertex,
        n_samples=n_samples,
        rng=rng,
        vsize=vsize,
        wsize=wsize,
    )
    t_enkf_s = time.time() - t0_enkf

    n_valid = n_samples - len(ens['zero_samples'])
    result['t_ensemble_s'] = t_ensemble_s
    result['t_enkf_s']     = t_enkf_s
    result['n_valid']      = n_valid

    # --- Log step ---
    if steps_csv is not None:
        _append_steps_csv(steps_csv, t, observation_time, n_samples, n_valid,
                          t_ensemble_s, t_enkf_s, vsize, wsize, run_id)

    return result


# ============================================================================
# LEGACY WRAPPER — backward compatible with existing experiment notebook
# ============================================================================

def adjusted_state_EnKF_farsite(
    initial_state, observation_state, observation_time,
    X, n_states, n_output, n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    vsize, wsize,
    dist_res, perim_res, lcppath,
    params=None,
    n_workers=1,
    members_csv=None,
    steps_csv=None,
    fire_name='unknown',
    step_idx=0,
    run_id=None,
):
    """
    Legacy interface — calls step() internally.

    New code should call step() directly. The orchestrator uses step().

    Returns:
        (adjusted_state, X, zkphat_ensemble, xkhat_ensemble,
         ykhat_ensemble, xkphat_ensemble)
    """
    # Unwrap X if accidentally wrapped in list
    if isinstance(X, list):
        X = X[0]
    X = np.asarray(X)

    # Resample states to target vertex count
    initial_state     = np.asarray(initial_state).reshape(-1, 1)
    observation_state = np.asarray(observation_state).reshape(-1, 1)
    initial_state     = resample_state_to_vertex_count(initial_state, n_vertex)
    observation_state = resample_state_to_vertex_count(observation_state, n_vertex)
    n_states = 2 * n_vertex
    n_output = n_states

    if X.shape != (n_states, n_states):
        X = 1000.0 * np.eye(n_states)

    # Pre-generate perturbations so rng sequence matches the old function exactly
    # (xkhat perturbations come from Xs @ rng.normal)
    Xs = np.linalg.cholesky(X)

    # Build prior ensemble in main process (same as old code)
    xkhat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble = np.zeros((n_states, n_samples))
    for s in range(n_samples):
        xkhat_ensemble[:, s:s+1] = initial_state + Xs @ rng.normal(size=(n_states, 1))
        ykhat_ensemble[:, s:s+1] = xkhat_ensemble[:, s:s+1] + rng.normal(0, scale=vsize, size=(n_states, 1))

    # Debug bounds
    init_geom = state_to_geom(initial_state)
    print("Initial bounds:", init_geom.bounds)
    obs_geom = state_to_geom(observation_state)
    print("Observation bounds:", obs_geom.bounds)

    # Build job list
    jobs = []
    _run_id = run_id or str(uuid.uuid4())[:8]
    for s in range(n_samples):
        jobs.append(dict(
            s=s,
            seed=int(rng.integers(0, 2**31)),
            xkhat_col=xkhat_ensemble[:, s:s+1].tolist(),
            ws=float(sampled_wslst[s]),
            wd=float(sampled_wdlst[s]),
            observation_time=observation_time,
            lcppath=str(lcppath),
            dist_res=dist_res,
            perim_res=perim_res,
            params=params,
            n_vertex=n_vertex,
            initial_state_arr=initial_state.tolist(),
            dt=dt,
            run_id=_run_id,
            fire_name=fire_name,
            step_idx=step_idx,
        ))

    zero_samples = []
    result_map   = {}
    zkphat_ensemble = np.zeros((n_states, n_samples))

    if n_workers <= 1:
        for job in tqdm(jobs, desc="Running ensemble"):
            res = _run_one_member(job)
            result_map[res['s']] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one_member, job): job['s'] for job in jobs}
            for fut in tqdm(as_completed(futures), total=n_samples, desc="Running ensemble"):
                res = fut.result()
                result_map[res['s']] = res

    member_results = []
    for s in range(n_samples):
        res = result_map[s]
        member_results.append(res)
        if res['zkphat_col'] is None:
            zero_samples.append(s)
        else:
            zkphat_ensemble[:, s:s+1] = np.array(res['zkphat_col'])

    if members_csv is not None:
        _append_members_csv(members_csv, member_results, jobs, n_samples,
                            dist_res, perim_res, n_workers, fire_name, step_idx, _run_id)

    # EnKF update
    valid_mask = np.ones(n_samples, dtype=bool)
    valid_mask[zero_samples] = False
    Ne = int(valid_mask.sum())

    print(f"Valid samples: {Ne}, Failed samples: {len(zero_samples)}")
    if Ne <= 1:
        raise RuntimeError(f"Too few valid FARSITE samples ({Ne}/{n_samples}). EnKF cannot proceed.")

    Z = zkphat_ensemble[:, valid_mask]
    Y = ykhat_ensemble[:, valid_mask]

    zkphat_mean = Z.mean(axis=1, keepdims=True)
    ykhat_mean  = Y.mean(axis=1, keepdims=True)

    Z_anom = Z - zkphat_mean
    if float(wsize) > 0:
        Z_anom = Z_anom + rng.normal(0.0, float(wsize), size=Z_anom.shape)
    Y_anom = Y - ykhat_mean

    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk             = aligned_states[1]
    innovation     = yk - ykhat_mean

    r2   = float(vsize) ** 2
    if r2 <= 0:
        raise ValueError("vsize must be > 0")
    rinv = 1.0 / r2

    S     = rinv * (Y_anom.T @ Y_anom) + (Ne - 1) * np.eye(Ne)
    eps   = 1e-10 * (np.trace(S) / S.shape[0])
    S_reg = S + eps * np.eye(Ne)

    rhs            = rinv * (Y_anom.T @ innovation)
    w              = np.linalg.solve(S_reg, rhs)
    adjusted_state = zkphat_mean + (Z_anom @ w)

    xkphat_ensemble  = np.zeros((n_states, n_samples))
    innovation_all   = yk - Y
    rhs_all          = rinv * (Y_anom.T @ innovation_all)
    W_all            = np.linalg.solve(S_reg, rhs_all)
    xkphat_valid     = Z + (Z_anom @ W_all)
    xkphat_ensemble[:, valid_mask]  = xkphat_valid
    xkphat_ensemble[:, ~valid_mask] = adjusted_state

    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    ex = xkphat_ensemble - xkphat_mean
    X  = (ex @ ex.T) / (n_samples - 1) + 1e-10 * np.eye(n_states)

    adjusted_geom  = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom)
    adjusted_state = align_states([initial_state, adjusted_state], vertex_count=n_vertex)[1]

    if steps_csv is not None:
        _append_steps_csv(steps_csv, step_idx, observation_time, n_samples, Ne,
                          0.0, 0.0, vsize, wsize, _run_id)

    print("REACHED RETURN")
    return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble


# ============================================================================
# CSV LOGGING HELPERS
# ============================================================================

_MEMBERS_HEADER = [
    'run_id', 'fire', 'step', 'member', 'N',
    'dist_res', 'perim_res', 'ncores', 'n_parallel_workers',
    'wall_s', 'cpu_s', 'peak_mem_mb', 'status',
    'forecast_interval_min', 'perimeter_vertex_count',
    'perimeter_area_m2', 'wind_speed_ms', 'wind_dir_deg',
]

_STEPS_HEADER = [
    'run_id', 'step', 't_obs',
    'N', 'n_valid',
    't_ensemble_s', 't_enkf_s',
    'vsize', 'wsize',
]


def _append_members_csv(csv_path, member_results, jobs, n_samples,
                        dist_res, perim_res, n_workers, fire_name, step_idx, run_id):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_MEMBERS_HEADER)
        if write_header:
            writer.writeheader()
        for res, job in zip(member_results, jobs):
            writer.writerow({
                'run_id':               run_id,
                'fire':                 fire_name,
                'step':                 step_idx,
                'member':               res['s'],
                'N':                    n_samples,
                'dist_res':             dist_res,
                'perim_res':            perim_res,
                'ncores':               1,   # FARSITE uses 1 core per member
                'n_parallel_workers':   n_workers,
                'wall_s':               round(res.get('wall_s', 0), 3),
                'cpu_s':                round(res.get('cpu_s', 0), 3),
                'peak_mem_mb':          '',
                'status':               res.get('status', ''),
                'forecast_interval_min':'',
                'perimeter_vertex_count':'',
                'perimeter_area_m2':    '',
                'wind_speed_ms':        job.get('ws', ''),
                'wind_dir_deg':         job.get('wd', ''),
            })


def _append_steps_csv(csv_path, step_idx, observation_time, n_samples, n_valid,
                      t_ensemble_s, t_enkf_s, vsize, wsize, run_id):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_STEPS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'run_id':       run_id,
            'step':         step_idx,
            't_obs':        str(observation_time),
            'N':            n_samples,
            'n_valid':      n_valid,
            't_ensemble_s': round(t_ensemble_s, 3),
            't_enkf_s':     round(t_enkf_s, 3),
            'vsize':        vsize,
            'wsize':        wsize,
        })


# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def sample_geometry(geom, rng, sigma=1):
    """Sample a perturbed geometry by adding Gaussian noise to vertices."""
    sampled_vertices = []
    for (x, y) in geom.exterior.coords[:-1]:
        randx = rng.normal(0, sigma)
        randy = rng.normal(0, sigma)
        sampled_vertices.append((x + randx, y + randy))
    from shapely.geometry import Polygon
    return Polygon(sampled_vertices)


def sample_windspeed(loc, sigma, rng):
    """Sample wind speed from Gaussian distribution (non-negative)."""
    ws = rng.normal(loc, sigma)
    return max(ws, 0)


def sample_winddirection(loc, sigma, rng):
    """Sample wind direction from Gaussian distribution (wrapped to 0-360)."""
    return np.fmod(rng.normal(loc, sigma) + 360, 360)


# ============================================================================
# I/O UTILITIES
# ============================================================================

def save_enkf_state(filepath, initial_state, adjusted_state, X,
                    zkphat_ensemble, xkhat_ensemble, ykhat_ensemble,
                    xkphat_ensemble, rng):
    """Save EnKF state to disk for checkpointing or analysis."""
    np.savez(
        filepath,
        initial_state=initial_state,
        adjusted_state=adjusted_state,
        X=X,
        zkphat_ensemble=zkphat_ensemble,
        xkhat_ensemble=xkhat_ensemble,
        ykhat_ensemble=ykhat_ensemble,
        xkphat_ensemble=xkphat_ensemble,
        rng_state=rng.bit_generator.state
    )
    print(f"EnKF state saved to {filepath}")


def load_enkf_state(filepath):
    """Load EnKF state from disk."""
    data = np.load(filepath, allow_pickle=True)
    rng = np.random.default_rng()
    if "rng_state" in data:
        rng.bit_generator.state = data["rng_state"].item()
    return data, rng