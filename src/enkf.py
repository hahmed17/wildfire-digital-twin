"""
Ensemble Kalman Filter (EnKF) module for fire data assimilation.
Implements EnKF with FARSITE forward model for wildfire state estimation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from farsite import forward_pass_farsite, cleanup_farsite_outputs
from geometry import (
    state_to_geom, geom_to_state, align_states, 
    resample_state_to_vertex_count, validate_geom
)
from config import FARSITE_TMP_DIR


# ============================================================================
# ENSEMBLE KALMAN FILTER — MODULAR FUNCTIONS
# ============================================================================

def run_ensemble(
    initial_state, observation_time,
    X, n_states, n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    vsize, wsize,
    dist_res, perim_res, lcppath
):
    """
    Run FARSITE ensemble forecast for all ensemble members.

    Generates perturbed prior ensemble members, runs FARSITE forward model
    on each, and returns the full ensemble for use in the EnKF update step.

    Args:
        initial_state: Initial fire perimeter state vector (2*n_vertex, 1)
        observation_time: Datetime of current observation (used as FARSITE start time)
        X: State covariance matrix (n_states, n_states)
        n_states: State dimension (2 * n_vertex)
        n_vertex: Number of vertices in fire perimeter
        n_samples: Number of ensemble members
        rng: Numpy random number generator
        sampled_wslst: Wind speed samples for each ensemble member
        sampled_wdlst: Wind direction samples for each ensemble member
        dt: Forecast time step (pandas Timedelta)
        vsize: Observation noise standard deviation (meters)
        wsize: Model noise standard deviation (meters)
        dist_res: FARSITE distance resolution (meters)
        perim_res: FARSITE perimeter resolution (meters)
        lcppath: Path to landscape (.lcp) file

    Returns:
        Tuple of (zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, zero_samples)
        - zkphat_ensemble: Forecast ensemble after FARSITE (n_states, n_samples)
        - xkhat_ensemble: Prior ensemble before FARSITE (n_states, n_samples)
        - ykhat_ensemble: Predicted observation ensemble (n_states, n_samples)
        - zero_samples: List of indices where FARSITE failed
    """
    xkhat_ensemble = np.zeros((n_states, n_samples))
    zkphat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble = np.zeros((n_states, n_samples))

    Xs = np.linalg.cholesky(X)
    zero_samples = []

    print("Initial bounds:", state_to_geom(initial_state).bounds)

    for s in tqdm(range(n_samples), desc="Running ensemble"):

        # Generate prior ensemble member
        xkhat_ensemble[:, s:(s+1)] = initial_state + Xs @ rng.normal(size=(n_states, 1))

        ws = sampled_wslst[s]
        wd = sampled_wdlst[s]

        # Generate predicted observation with observation noise
        ykhat_ensemble[:, s:(s+1)] = xkhat_ensemble[:, s:(s+1)] + rng.normal(0, scale=vsize, size=(n_states, 1))

        # Run FARSITE (multi-step if dt > 30 min)
        total_dt = dt
        farsite_dt = pd.Timedelta(minutes=30)
        n_steps = int(total_dt / farsite_dt)

        farsite_params = {
            'windspeed': int(ws),
            'winddirection': int(wd),
            'dt': farsite_dt
        }
        input_poly = state_to_geom(xkhat_ensemble[:, s:(s+1)])
        start_time = observation_time.strftime("%Y-%m-%d %H:%M:%S")
        forward_geom = None

        for i in range(n_steps):
            forward_geom = forward_pass_farsite(
                poly=input_poly,
                params=farsite_params,
                start_time=start_time,
                lcppath=lcppath,
                dist_res=dist_res,
                perim_res=perim_res
            )
            if forward_geom is None:
                break
            input_poly = forward_geom

        if forward_geom is None:
            zero_samples.append(s)
            continue

        forward_state = geom_to_state(forward_geom)
        aligned_states = align_states([initial_state, forward_state], vertex_count=n_vertex)
        zkphat_ensemble[:, s:(s+1)] = aligned_states[1]

    print(f"Valid samples: {n_samples - len(zero_samples)}, Failed samples: {len(zero_samples)}")
    return zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, zero_samples


def enkf_update(
    initial_state, observation_state,
    zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, zero_samples,
    n_states, n_vertex, n_samples,
    vsize, wsize, rng
):
    """
    EnKF analysis update step.

    Given the forecast ensemble from run_ensemble and the actual observation,
    computes the analysis mean state and updated ensemble using the ensemble-space
    Kalman gain (numerically stable for high-dimensional states).

    Args:
        initial_state: Initial fire perimeter state vector (2*n_vertex, 1)
        observation_state: Observed fire perimeter state vector (2*n_vertex, 1)
        zkphat_ensemble: Forecast ensemble from run_ensemble (n_states, n_samples)
        xkhat_ensemble: Prior ensemble from run_ensemble (n_states, n_samples)
        ykhat_ensemble: Predicted observation ensemble from run_ensemble (n_states, n_samples)
        zero_samples: List of failed FARSITE indices from run_ensemble
        n_states: State dimension (2 * n_vertex)
        n_vertex: Number of vertices in fire perimeter
        n_samples: Number of ensemble members
        vsize: Observation noise standard deviation (meters)
        wsize: Model noise standard deviation (meters)
        rng: Numpy random number generator

    Returns:
        Tuple of (adjusted_state, X, xkphat_ensemble)
        - adjusted_state: Analysis mean state (2*n_vertex, 1)
        - X: Updated covariance matrix (n_states, n_states)
        - xkphat_ensemble: Analysis ensemble (n_states, n_samples)
    """
    valid_mask = np.ones(n_samples, dtype=bool)
    valid_mask[zero_samples] = False
    Ne = int(valid_mask.sum())

    if Ne <= 1:
        raise RuntimeError(f"Too few valid FARSITE samples ({Ne}/{n_samples}). EnKF cannot proceed.")

    Z = zkphat_ensemble[:, valid_mask]
    Y = ykhat_ensemble[:, valid_mask]

    zkphat_mean = Z.mean(axis=1, keepdims=True)
    ykhat_mean = Y.mean(axis=1, keepdims=True)

    Z_anom = Z - zkphat_mean
    if float(wsize) > 0:
        Z_anom = Z_anom + rng.normal(0.0, float(wsize), size=Z_anom.shape)
    Y_anom = Y - ykhat_mean

    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk = aligned_states[1]

    innovation = yk - ykhat_mean

    r2 = float(vsize) ** 2
    if r2 <= 0:
        raise ValueError("vsize must be > 0 (observation noise std).")
    rinv = 1.0 / r2

    S = rinv * (Y_anom.T @ Y_anom) + (Ne - 1) * np.eye(Ne)
    eps = 1e-10 * (np.trace(S) / S.shape[0])
    S_reg = S + eps * np.eye(Ne)

    rhs = rinv * (Y_anom.T @ innovation)
    w = np.linalg.solve(S_reg, rhs)

    print("Calculating adjusted state (ensemble-space)")
    adjusted_state = zkphat_mean + (Z_anom @ w)

    xkphat_ensemble = np.zeros((n_states, n_samples))
    innovation_all = yk - Y
    rhs_all = rinv * (Y_anom.T @ innovation_all)
    W_all = np.linalg.solve(S_reg, rhs_all)
    xkphat_valid = Z + (Z_anom @ W_all)
    xkphat_ensemble[:, valid_mask] = xkphat_valid
    xkphat_ensemble[:, ~valid_mask] = adjusted_state

    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    ex = xkphat_ensemble - xkphat_mean
    X = (ex @ ex.T) / (n_samples - 1) + 1e-10 * np.eye(n_states)

    adjusted_geom = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom)
    adjusted_state = align_states([initial_state, adjusted_state], vertex_count=n_vertex)[1]

    print("REACHED RETURN")
    return adjusted_state, X, xkphat_ensemble


# ============================================================================
# ENSEMBLE KALMAN FILTER — MONOLITHIC WRAPPER (kept for back-compat)
# ============================================================================

def adjusted_state_EnKF_farsite(
    initial_state, observation_state, observation_time,
    X, n_states, n_output, n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    vsize, wsize,
    dist_res, perim_res, lcppath
):
    """
    Thin wrapper around run_ensemble() and enkf_update() kept for back-compatibility.
    Same signature and return value as the original monolithic function.
    """
    if isinstance(X, list):
        X = X[0]
    X = np.asarray(X)

    target_n_states = 2 * n_vertex
    initial_state = np.asarray(initial_state).reshape(-1, 1)
    observation_state = np.asarray(observation_state).reshape(-1, 1)
    initial_state = resample_state_to_vertex_count(initial_state, n_vertex)
    observation_state = resample_state_to_vertex_count(observation_state, n_vertex)
    n_states = target_n_states

    if X.shape != (n_states, n_states):
        X = 1000.0 * np.eye(n_states)

    print("Observation bounds:", state_to_geom(observation_state).bounds)

    zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, zero_samples = run_ensemble(
        initial_state=initial_state,
        observation_time=observation_time,
        X=X,
        n_states=n_states,
        n_vertex=n_vertex,
        n_samples=n_samples,
        rng=rng,
        sampled_wslst=sampled_wslst,
        sampled_wdlst=sampled_wdlst,
        dt=dt,
        vsize=vsize,
        wsize=wsize,
        dist_res=dist_res,
        perim_res=perim_res,
        lcppath=lcppath,
    )

    adjusted_state, X, xkphat_ensemble = enkf_update(
        initial_state=initial_state,
        observation_state=observation_state,
        zkphat_ensemble=zkphat_ensemble,
        xkhat_ensemble=xkhat_ensemble,
        ykhat_ensemble=ykhat_ensemble,
        zero_samples=zero_samples,
        n_states=n_states,
        n_vertex=n_vertex,
        n_samples=n_samples,
        vsize=vsize,
        wsize=wsize,
        rng=rng,
    )

    return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble


# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def sample_geometry(geom, rng, sigma=1):
    sampled_vertices = []
    for (x, y) in geom.exterior.coords[:-1]:
        randx = rng.normal(0, sigma)
        randy = rng.normal(0, sigma)
        sampled_vertices.append((x + randx, y + randy))
    from shapely.geometry import Polygon
    return Polygon(sampled_vertices)


def sample_windspeed(loc, sigma, rng):
    ws = rng.normal(loc, sigma)
    if ws < 0:
        ws = 0
    return ws


def sample_winddirection(loc, sigma, rng):
    return np.fmod(rng.normal(loc, sigma) + 360, 360)


# ============================================================================
# I/O UTILITIES
# ============================================================================

def save_enkf_state(filepath, initial_state, adjusted_state, X,
                    zkphat_ensemble, xkhat_ensemble, ykhat_ensemble,
                    xkphat_ensemble, rng):
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
    data = np.load(filepath, allow_pickle=True)
    rng = np.random.default_rng()
    if "rng_state" in data:
        rng.bit_generator.state = data["rng_state"].item()
    return data, rng