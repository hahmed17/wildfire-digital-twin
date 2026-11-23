from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import time

import enkf
import farsite
import config

import geopandas as gpd
import numpy as np

import firemap
from orchestrator import checks
from orchestrator.state import ObservationEvent, ForecastPlan
from geometry import validate_geom, geom_to_state, resample_state_to_vertex_count

def load_observation(event: ObservationEvent,
                     n_vertex: int = 20) -> np.ndarray:
    """
    Given an ObservationEvent, it opens the file the event points to, 
    extracts that fire perimeter, and returns it in the form the EnKF code expects.

    Args:
        event (ObservationEvent): The observation event to be loaded.
        n_vertex (int): The number of vertices to resample the state to.
    """
    # Reproject the whole table to metres (EPSG:5070), since the EnKF code expects that.
    # This must happen on the GeoDataFrame: a single Polygon doesn't know its CRS.
    gdf = gpd.read_file(event.path).to_crs(epsg=5070)

    # The GeoJSON holds all 6 Border 2 perimeters; feature_id is the row number.
    fire_perimeter = gdf.geometry.iloc[int(event.feature_id)]

    # Clean the shape (lines -> polygon, fix invalid, keep largest piece).
    fire_perimeter = validate_geom(fire_perimeter)
    if fire_perimeter is None:
        raise ValueError(f"Feature {event.feature_id} in {event.path} has no usable polygon")

    # calls the EnKF code to convert the fire perimeter to the format it expects.
    state = geom_to_state(fire_perimeter)

    # Reshape to the specified number of vertices to match the model state
    state = resample_state_to_vertex_count(state, n_vertex)

    return state

# firemap.query_weather_for_timestep's static fallback when no RAWS data is available.
FALLBACK_WIND_SPEEDS = [10.0]
FALLBACK_WIND_DIRECTIONS = [225.0]


def fetch_weather(start: datetime, 
                  end: datetime,
                  lat: float = 32.59588662537945, 
                  lon: float = -116.85708741551528,):
    """Given a location and a time window, return the observed wind speeds 
    and directions for that window from the FIREMAP weather service.
    
    Args:
        start (datetime): The start of the time window.
        end (datetime): The end of the time window.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        dict: dictionary that contains (wind_speeds, wind_directions, used_fallback)
    """
    
    # keyword arguments: Hena's order is (lat, lon, start_time, end_time)
    speeds, directions = firemap.query_weather_for_timestep(
        lat=lat, lon=lon, start_time=start, end_time=end)
    use_fallback = (speeds == [10.0] and directions == [225.0])

    # Drop gaps before anything downstream sees them. The Firemap API returns null
    # for timesteps a station did not report, so a real RAWS window often contains
    # None: np.mean over it raises, and np.asarray(..., dtype=float) turns it into
    # NaN that flows silently into FARSITE. See checks.finite_readings.
    n_raw = len(speeds or [])
    clean_speeds, clean_directions = checks.finite_readings(speeds, directions)
    n_dropped = n_raw - len(clean_speeds)

    if not clean_speeds:
        # Nothing usable in the window: fall back rather than hand on an empty list.
        clean_speeds = list(FALLBACK_WIND_SPEEDS)
        clean_directions = list(FALLBACK_WIND_DIRECTIONS)
        use_fallback = True

    return {
        "wind_speeds": clean_speeds,
        "wind_directions": clean_directions,
        "used_fallback": use_fallback,
        "n_readings_dropped": n_dropped,
    }


# ---------------------------------------------------------------------------
# Forecast: sample winds + run the FARSITE ensemble
# ---------------------------------------------------------------------------
# Values from the experiment notebook (border2_experiment.ipynb)
LCP_PATH = config.DATA_DIR / "border2_inputs" / "landscape.lcp"
FARSITE_PARAMS = {"temperature": 66, "humidity": 0.08}   # °F, fraction (no API source)
WIND_SPEED_NOISE_SD = 2.0      # used only when there is a single wind reading
WIND_DIR_NOISE_SD = 15.0       # degrees


def sample_winds(weather: dict, n: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Give each of the n ensemble members its own wind (speed, direction).

    Same logic as the notebook:
      - several readings: each member picks one reading at random, keeping that
        reading's speed and direction together (sampling with replacement);
      - one reading (e.g. the fallback): every member gets it plus random noise.

    Args:
        weather: dict returned by fetch_weather
        n: number of ensemble members (plan.ensemble_size)
        rng: numpy random generator (fixed seed -> repeatable runs)

    Returns:
        (speeds, directions): two arrays of length n
    """
    # Filter again here rather than trusting the caller: np.asarray(..., dtype=float)
    # silently turns None into NaN, and a NaN wind reaches FARSITE as int(nan),
    # which raises inside the worker and shows up only as a failed member.
    clean_speeds, clean_directions = checks.finite_readings(
        weather.get("wind_speeds"), weather.get("wind_directions"))
    if not clean_speeds:
        clean_speeds = list(FALLBACK_WIND_SPEEDS)
        clean_directions = list(FALLBACK_WIND_DIRECTIONS)
    speeds = np.asarray(clean_speeds, dtype=float)
    directions = np.asarray(clean_directions, dtype=float)

    if len(speeds) > 1:
        # Case A: pick n random positions; use the SAME positions for both arrays
        idx = rng.integers(0, len(speeds), size=n)
        sampled_speeds = speeds[idx]
        sampled_directions = directions[idx]
    else:
        # Case B: one reading + noise
        sampled_speeds = speeds[0] + rng.normal(0, WIND_SPEED_NOISE_SD, size=n)
        sampled_directions = directions[0] + rng.normal(0, WIND_DIR_NOISE_SD, size=n)
        sampled_speeds = np.maximum(sampled_speeds, 0)        # no negative wind
        sampled_directions = sampled_directions % 360         # wrap into 0-360

    return sampled_speeds, sampled_directions


@dataclass
class EnsembleResult:
    """What run_ensemble returns. Holds big numpy arrays, so it is NOT stored in
    TwinState (TwinState keeps a file path to it instead)."""
    raw: dict                       # Hena's dict, unchanged; enkf_update needs it
    plan: ForecastPlan              # the plan that produced this ensemble
    wind_speeds: np.ndarray         # (N,) wind given to each member
    wind_directions: np.ndarray     # (N,)
    wall_s: float                   # total wall-clock time of the ensemble

    @property
    def forecast(self) -> np.ndarray:
        """(n_states, N) FARSITE forecast perimeters (zkphat). Failed members are zeros."""
        return self.raw["zkphat_ensemble"]

    @property
    def valid_mask(self) -> np.ndarray:
        """(N,) True for members that finished successfully."""
        mask = np.ones(self.plan.ensemble_size, dtype=bool)
        mask[self.raw["zero_samples"]] = False
        return mask

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def member_wall_s(self) -> list[float]:
        """Wall-clock seconds per member (what the cost model learns from)."""
        return [m["wall_s"] for m in self.raw["member_results"]]


def run_ensemble(state_vec: np.ndarray,
                 cov_X: np.ndarray,
                 plan: ForecastPlan,
                 weather: dict,
                 start_time: datetime,
                 dt: timedelta,
                 rng: np.random.Generator,
                 n_vertex: int = 20,
                 farsite_dt_minutes: int = 30,
                 lcp_path: str | Path = LCP_PATH,
                 params: dict | None = None,
                 members_csv: str | Path | None = None,
                 fire_name: str = "Border_2",
                 step_idx: int = 0,
                 run_id: str | None = None) -> EnsembleResult:
    """
    Run the FARSITE ensemble described by `plan`, starting from the current state.

    Args:
        state_vec: current fire state, (2*n_vertex, 1), metres (EPSG:5070)
        cov_X: current covariance, (2*n_vertex, 2*n_vertex)
        plan: ensemble_size -> N, resolution_m -> dist_res & perim_res, workers -> n_workers
              (plan.reuse_cache is not used here yet)
        weather: dict from fetch_weather
        start_time: start of the forecast window
        dt: length of the forecast window
        rng: numpy random generator, shared for the whole run
        n_vertex: vertices per perimeter (must match state_vec and cov_X)
        farsite_dt_minutes: FARSITE sub-step length; Hena's notebook uses 15 for
              high-wind steps, 30 otherwise
        lcp_path: FARSITE landscape file
        params: extra FARSITE params; defaults to the notebook's temperature/humidity
        members_csv, fire_name, step_idx, run_id: passed to Hena's members.csv logging

    Returns:
        EnsembleResult
    """
    # 1. One wind per member
    speeds, directions = sample_winds(weather, plan.ensemble_size, rng)

    # 2. Hena's ensemble, with our plan mapped onto her arguments
    t0 = time.perf_counter()
    raw = enkf.run_ensemble(
        initial_state=state_vec,
        observation_time=start_time,
        n_states=2 * n_vertex,
        n_vertex=n_vertex,
        n_samples=plan.ensemble_size,
        rng=rng,
        sampled_wslst=speeds,
        sampled_wdlst=directions,
        dt=dt,
        dist_res=plan.resolution_m,
        perim_res=plan.resolution_m,
        lcppath=str(lcp_path),
        X=cov_X,
        params=params if params is not None else FARSITE_PARAMS,
        n_workers=plan.workers,
        farsite_dt_minutes=farsite_dt_minutes,
        members_csv=members_csv,
        fire_name=fire_name,
        step_idx=step_idx,
        run_id=run_id,
    )
    wall_s = time.perf_counter() - t0

    # 3. Wrap the result
    return EnsembleResult(raw=raw, plan=plan, wind_speeds=speeds,
                          wind_directions=directions, wall_s=wall_s)


# Wind speed is mph everywhere in this pipeline and is never converted:
# firemap.query_weather_for_timestep returns mph because config.RAWS_UNITS == 'English',
# FARSITE is given the same numbers, and members.csv logs them unchanged. That column
# used to be named wind_speed_ms, which was a naming mistake rather than a conversion;
# it is wind_speed_mph now. Don't add a conversion helper here — if a consumer wants
# m/s it should convert at the point of display.


# ---------------------------------------------------------------------------
# EnKF update
# ---------------------------------------------------------------------------

def enkf_update(
    ens: EnsembleResult,
    obs_vec: np.ndarray,
    prior_state: np.ndarray,
    prior_cov_X: np.ndarray,
    vsize_m: float,
    wsize_m: float,
    inflation: float = 1.0,
    n_vertex: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the EnKF analysis step given a completed forecast ensemble.

    `prior_state` and `prior_cov_X` are the analysis and covariance carried forward from
    the previous cycle, and both must be passed explicitly. The experiment notebook
    threads them through its loop as `current_state` and `X`; the forecast ensemble dict
    from enkf.run_ensemble contains neither, so reading them off it would silently
    substitute the ensemble mean for the analysis and reset the covariance every cycle.

    Args:
        ens:         EnsembleResult from run_ensemble (holds raw forecast dict + plan)
        obs_vec:     (2*n_vertex, 1) observed perimeter state vector in EPSG:5070 metres
        prior_state: (2*n_vertex, 1) previous analysis — the notebook's `current_state`
        prior_cov_X: (2*n_vertex, 2*n_vertex) previous posterior covariance — the
                     notebook's `X`, initialised to 1000 * I before the first step
        vsize_m:     Observation noise std (m) — how much to trust the observation
        wsize_m:     Model noise std (m) — how much to trust the forecast
        inflation:   Ensemble spread multiplier applied before the update (1.0 = none).
                     Use > 1.0 (e.g. 1.4) after a wind shift to widen the prior spread.
        n_vertex:    Vertices per perimeter state vector (must match ens)
        rng:         Numpy RNG; uses a new default_rng() if None

    Returns:
        (analysis_vec, new_cov_X):
            analysis_vec  (2*n_vertex, 1) — posterior fire state estimate
            new_cov_X     (2*n_vertex, 2*n_vertex) — updated covariance
    """
    if rng is None:
        rng = np.random.default_rng()

    n_states  = 2 * n_vertex
    n_samples = ens.plan.ensemble_size
    raw       = ens.raw

    # Apply covariance inflation by scaling forecast anomalies.
    # We achieve this by temporarily widening zkphat_ensemble around its mean
    # before passing to enkf_update, then restoring the original dict.
    if inflation != 1.0:
        Z = raw["zkphat_ensemble"]                        # (n_states, N)
        mean = Z.mean(axis=1, keepdims=True)
        inflated_Z = mean + inflation * (Z - mean)
        raw_for_update = {**raw, "zkphat_ensemble": inflated_Z}
    else:
        raw_for_update = raw

    result = enkf.enkf_update(
        ensemble=raw_for_update,
        initial_state=prior_state,
        observation_state=obs_vec,
        X=prior_cov_X,
        n_states=n_states,
        n_output=n_states,
        n_vertex=n_vertex,
        n_samples=n_samples,
        rng=rng,
        vsize=vsize_m,
        wsize=wsize_m,
    )

    analysis_vec = result["adjusted_state"]   # (n_states, 1)
    new_cov_X    = result["X"]                # (n_states, n_states)
    return analysis_vec, new_cov_X


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(
    path: str | Path,
    state_vec: np.ndarray,
    cov_X: np.ndarray,
    rng: np.random.Generator,
    ensemble: EnsembleResult | None = None,
) -> None:
    """
    Save the current EnKF state (and optionally the last forecast ensemble) to disk.

    Args:
        path:      Output .npz path (created or overwritten)
        state_vec: (2*n_vertex, 1) analysis state vector
        cov_X:     (2*n_vertex, 2*n_vertex) posterior covariance
        rng:       Numpy RNG whose state is saved for reproducibility
        ensemble:  If provided, the forecast ensemble arrays are also saved
                   (zkphat, xkhat, ykhat, zero_samples) for agent development
                   without re-running FARSITE.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = dict(
        adjusted_state=state_vec,
        initial_state=state_vec,   # save as both for enkf.load_enkf_state compat
        X=cov_X,
        rng_state=np.array(rng.bit_generator.state, dtype=object),
    )
    if ensemble is not None:
        raw = ensemble.raw
        kwargs["zkphat_ensemble"] = raw["zkphat_ensemble"]
        kwargs["xkhat_ensemble"]  = raw["xkhat_ensemble"]
        kwargs["ykhat_ensemble"]  = raw["ykhat_ensemble"]
        kwargs["xkphat_ensemble"] = raw.get("xkphat_ensemble",
                                            np.zeros_like(raw["zkphat_ensemble"]))
        kwargs["zero_samples"]    = np.array(raw["zero_samples"], dtype=int)

    np.savez(path, **kwargs)


def load_state(
    path: str | Path,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """
    Load an EnKF state saved by save_state.

    Args:
        path: .npz file written by save_state (or enkf.save_enkf_state)
        rng:  If provided, its state is restored from the file.
              If None, a new default_rng() is returned with the saved state.

    Returns:
        (state_vec, cov_X, rng):
            state_vec  (2*n_vertex, 1) — analysis state
            cov_X      (2*n_vertex, 2*n_vertex) — posterior covariance
            rng        numpy Generator with restored state
    """
    data = np.load(path, allow_pickle=True)

    state_vec = data["adjusted_state"]
    cov_X     = data["X"]

    if rng is None:
        rng = np.random.default_rng()
    if "rng_state" in data:
        rng.bit_generator.state = data["rng_state"].item()

    return state_vec, cov_X, rng