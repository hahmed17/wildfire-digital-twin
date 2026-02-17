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
# ENSEMBLE KALMAN FILTER
# ============================================================================

def adjusted_state_EnKF_farsite(
    initial_state, observation_state, observation_time,
    X, n_states, n_output, n_vertex, n_samples, rng,
    sampled_wslst, sampled_wdlst, dt,
    vsize, wsize,
    dist_res, perim_res, lcppath
):
    """
    Ensemble Kalman Filter for fire perimeter data assimilation with FARSITE.
    
    Uses ensemble-space formulation for numerical stability with high-dimensional states.
    
    Args:
        initial_state: Initial fire perimeter state vector (2*n_vertex, 1)
        observation_state: Observed fire perimeter state vector (2*n_vertex, 1)
        observation_time: Datetime of observation
        X: State covariance matrix (n_states, n_states)
        n_states: State dimension (2 * n_vertex)
        n_output: Output dimension (equals n_states for this problem)
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
        Tuple of (adjusted_state, X, zkphat_ensemble, xkhat_ensemble, 
                 ykhat_ensemble, xkphat_ensemble)
        - adjusted_state: Analysis mean state (2*n_vertex, 1)
        - X: Updated covariance matrix (n_states, n_states)
        - zkphat_ensemble: Forecast ensemble after FARSITE (n_states, n_samples)
        - xkhat_ensemble: Prior ensemble (n_states, n_samples)
        - ykhat_ensemble: Predicted observation ensemble (n_output, n_samples)
        - xkphat_ensemble: Analysis ensemble (n_states, n_samples)
    """
    
    # ========================================================================
    # DIMENSION HANDLING
    # ========================================================================
    
    # Unwrap X if accidentally wrapped in list
    if isinstance(X, list):
        X = X[0]
    X = np.asarray(X)
    
    # Force fixed state dimension from n_vertex
    target_n_states = 2 * n_vertex
    
    initial_state = np.asarray(initial_state).reshape(-1, 1)
    observation_state = np.asarray(observation_state).reshape(-1, 1)
    
    # Resample both states to target vertex count
    initial_state = resample_state_to_vertex_count(initial_state, n_vertex)
    observation_state = resample_state_to_vertex_count(observation_state, n_vertex)
    
    # Lock dimensions
    n_states = target_n_states
    n_output = target_n_states
    
    # Ensure covariance matrix matches
    if X.shape != (n_states, n_states):
        X = 1000.0 * np.eye(n_states)
    
    # ========================================================================
    # ENSEMBLE INITIALIZATION
    # ========================================================================
    
    xkhat_ensemble = np.zeros((n_states, n_samples))
    zkphat_ensemble = np.zeros((n_states, n_samples))
    xkphat_ensemble = np.zeros((n_states, n_samples))
    ykhat_ensemble = np.zeros((n_output, n_samples))
    
    Xs = np.linalg.cholesky(X)  # Square root of covariance matrix
    zero_samples = []  # Track failed FARSITE runs
    
    # Debug: Check initial geometry bounds
    init_geom = state_to_geom(initial_state)
    print("Initial bounds:", init_geom.bounds)
    
    obs_geom = state_to_geom(observation_state)
    print("Observation bounds:", obs_geom.bounds)
    
    # ========================================================================
    # ENSEMBLE FORECAST (FARSITE FORWARD MODEL)
    # ========================================================================
    
    for s in tqdm(range(n_samples), desc="Running ensemble"):
        
        # Generate prior ensemble member
        xkhat_ensemble[:, s:(s+1)] = initial_state + Xs @ rng.normal(size=(n_states, 1))
        
        # Get weather for this ensemble member
        ws = sampled_wslst[s]
        wd = sampled_wdlst[s]
        
        # Generate predicted observation with observation noise
        ykhat_ensemble[:, s:(s+1)] = xkhat_ensemble[:, s:(s+1)] + rng.normal(0, scale=vsize, size=(n_output, 1))
        
        # ====================================================================
        # RUN FARSITE ON THIS ENSEMBLE MEMBER
        # ====================================================================
        
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
        
        # Multi-step FARSITE if needed
        for i in range(n_steps):
            forward_geom = forward_pass_farsite(
                poly=input_poly,
                params=farsite_params,
                start_time=start_time,                lcppath=lcppath,
                dist_res=dist_res,
                perim_res=perim_res
            )
            if forward_geom is None:
                break
            input_poly = forward_geom        
        # Check if FARSITE succeeded
        if forward_geom is None:
            zero_samples.append(s)
            continue
        
        # Convert FARSITE output to state vector
        forward_state = geom_to_state(forward_geom)
        
        # Align forecast state with initial state
        aligned_states = align_states([initial_state, forward_state], vertex_count=n_vertex)
        zkphat_ensemble[:, s:(s+1)] = aligned_states[1]
    
    # ========================================================================
    # HANDLE FAILED ENSEMBLE MEMBERS
    # ========================================================================
    
    valid_mask = np.ones(n_samples, dtype=bool)
    valid_mask[zero_samples] = False
    Ne = int(valid_mask.sum())
    
    print(f"Valid samples: {Ne}, Failed samples: {len(zero_samples)}")
    if Ne <= 1:
        raise RuntimeError(f"Too few valid FARSITE samples ({Ne}/{n_samples}). EnKF cannot proceed.")
    
    # ========================================================================
    # ENSEMBLE KALMAN FILTER UPDATE
    # ========================================================================
    
    # Slice valid ensembles only
    Z = zkphat_ensemble[:, valid_mask]   # Forecast ensemble (n_states, Ne)
    Y = ykhat_ensemble[:, valid_mask]    # Predicted obs ensemble (n_output, Ne)
    
    # Compute means
    zkphat_mean = Z.mean(axis=1, keepdims=True)     # (n_states, 1)
    ykhat_mean = Y.mean(axis=1, keepdims=True)      # (n_output, 1)
    
    # Compute anomalies (with model noise on forecast)
    Z_anom = Z - zkphat_mean                        # (n_states, Ne)
    if float(wsize) > 0:
        Z_anom = Z_anom + rng.normal(0.0, float(wsize), size=Z_anom.shape)
    
    Y_anom = Y - ykhat_mean                         # (n_output, Ne)
    
    # Align actual observation with initial state
    aligned_states = align_states([initial_state, observation_state], vertex_count=n_vertex)
    yk = aligned_states[1]                          # Actual observation (n_output, 1)
    
    innovation = yk - ykhat_mean                    # Innovation (n_output, 1)
    
    # ========================================================================
    # ENSEMBLE-SPACE KALMAN GAIN (numerically stable for large n_states)
    # ========================================================================
    # Assumes observation error covariance R = (vsize^2) * I
    
    r2 = float(vsize) ** 2
    if r2 <= 0:
        raise ValueError("vsize must be > 0 (observation noise std).")
    rinv = 1.0 / r2
    
    # S = (Y_anom^T R^{-1} Y_anom) + (Ne-1) I   -> (Ne, Ne)
    S = rinv * (Y_anom.T @ Y_anom) + (Ne - 1) * np.eye(Ne)
    
    # Add small ridge for numerical stability
    eps = 1e-10 * (np.trace(S) / S.shape[0])
    S_reg = S + eps * np.eye(Ne)
    
    # Solve for ensemble weights: w = S^{-1} (Y_anom^T R^{-1} innovation)
    rhs = rinv * (Y_anom.T @ innovation)            # (Ne, 1)
    w = np.linalg.solve(S_reg, rhs)                 # (Ne, 1)
    
    print("Calculating adjusted state (ensemble-space)")
    adjusted_state = zkphat_mean + (Z_anom @ w)     # Analysis mean (n_states, 1)
    
    # ========================================================================
    # UPDATE ANALYSIS ENSEMBLE
    # ========================================================================
    # Deterministic member-wise update:
    # x_a^j = z^j + Z_anom * S^{-1} * (Y_anom^T R^{-1} (y - y^j))
    
    xkphat_ensemble = np.zeros((n_states, n_samples))
    
    # Compute weights for all members at once
    innovation_all = (yk - Y)                              # (n_output, Ne)
    rhs_all = rinv * (Y_anom.T @ innovation_all)          # (Ne, Ne)
    W_all = np.linalg.solve(S_reg, rhs_all)               # (Ne, Ne)
    
    # Apply update to valid members
    xkphat_valid = Z + (Z_anom @ W_all)                   # (n_states, Ne)
    xkphat_ensemble[:, valid_mask] = xkphat_valid
    
    # Fill invalid members with mean analysis state
    xkphat_ensemble[:, ~valid_mask] = adjusted_state
    
    # ========================================================================
    # UPDATE COVARIANCE MATRIX
    # ========================================================================
    
    xkphat_mean = xkphat_ensemble.mean(axis=1, keepdims=True)
    ex = xkphat_ensemble - xkphat_mean
    X = (ex @ ex.T) / (n_samples - 1) + 1e-10 * np.eye(n_states)
    
    # ========================================================================
    # GEOMETRY VALIDATION AND FINAL ALIGNMENT
    # ========================================================================
    
    adjusted_geom = validate_geom(state_to_geom(adjusted_state))
    adjusted_state = geom_to_state(adjusted_geom)
    adjusted_state = align_states([initial_state, adjusted_state], vertex_count=n_vertex)[1]
    
    print("REACHED RETURN")
    return adjusted_state, X, zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, xkphat_ensemble


# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def sample_geometry(geom, rng, sigma=1):
    """
    Sample a perturbed geometry by adding Gaussian noise to vertices.
    
    Args:
        geom: Shapely Polygon
        rng: Numpy random number generator
        sigma: Standard deviation of noise (meters)
        
    Returns:
        Perturbed Shapely Polygon
    """
    sampled_vertices = []
    
    for (x, y) in geom.exterior.coords[:-1]:
        randx = rng.normal(0, sigma)
        randy = rng.normal(0, sigma)
        sampled_vertices.append((x + randx, y + randy))
    
    from shapely.geometry import Polygon
    return Polygon(sampled_vertices)


def sample_windspeed(loc, sigma, rng):
    """
    Sample wind speed from Gaussian distribution (non-negative).
    
    Args:
        loc: Mean wind speed
        sigma: Standard deviation
        rng: Numpy random number generator
        
    Returns:
        Sampled wind speed (>= 0)
    """
    ws = rng.normal(loc, sigma)
    if ws < 0:
        ws = 0
    return ws


def sample_winddirection(loc, sigma, rng):
    """
    Sample wind direction from Gaussian distribution (wrapped to 0-360).
    
    Args:
        loc: Mean wind direction (degrees)
        sigma: Standard deviation
        rng: Numpy random number generator
        
    Returns:
        Sampled wind direction in [0, 360)
    """
    return np.fmod(rng.normal(loc, sigma) + 360, 360)


# ============================================================================
# I/O UTILITIES
# ============================================================================

def save_enkf_state(filepath, initial_state, adjusted_state, X, 
                    zkphat_ensemble, xkhat_ensemble, ykhat_ensemble, 
                    xkphat_ensemble, rng):
    """
    Save EnKF state to disk for checkpointing or analysis.
    
    Args:
        filepath: Output .npz file path
        initial_state: Initial state vector
        adjusted_state: Analysis mean state
        X: Covariance matrix
        zkphat_ensemble: Forecast ensemble
        xkhat_ensemble: Prior ensemble
        ykhat_ensemble: Predicted observation ensemble
        xkphat_ensemble: Analysis ensemble
        rng: Random number generator (state will be saved)
    """
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
    """
    Load EnKF state from disk.
    
    Args:
        filepath: Input .npz file path
        
    Returns:
        Tuple of (data dict, rng)
    """
    data = np.load(filepath, allow_pickle=True)
    rng = np.random.default_rng()
    if "rng_state" in data:
        rng.bit_generator.state = data["rng_state"].item()
    return data, rng