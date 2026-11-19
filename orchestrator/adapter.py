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
    return {
        "wind_speeds": speeds,
        "wind_directions": directions,
        "used_fallback": use_fallback
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
    speeds = np.asarray(weather["wind_speeds"], dtype=float)
    directions = np.asarray(weather["wind_directions"], dtype=float)

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
    t0 = time.time()
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
    wall_s = time.time() - t0

    # 3. Wrap the result
    return EnsembleResult(raw=raw, plan=plan, wind_speeds=speeds,
                          wind_directions=directions, wall_s=wall_s)


def enkf_update(ens: EnsembleResult, obs_vec, vsize_m, wsize_m, inflation): ...
    # after refactor: her enkf_update(...) -> (analysis_vec, new_cov_X)

def save_state(path, **arrays): ...    # enkf.save_enkf_state
def load_state(path): ...         # enkf.load_enkf_state

