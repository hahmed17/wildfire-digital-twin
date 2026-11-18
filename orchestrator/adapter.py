from datetime import datetime

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


def run_ensemble(state_vec: np.ndarray, 
                 cov_X: np.ndarray,
                 plan: ForecastPlan,
                 weather: dict,
                 dt: float,
                 lcp: str) -> EnsembleResult: ...
   """Given a state vector, covariance matrix, forecast plan, and weather data, 
   run the ensemble forecast using the FARSITE model.
   """
    # Call sample_winds(weather, n, rng)
    



def enkf_update(ens: EnsembleResult, obs_vec, vsize_m, wsize_m, inflation): ...
    # after refactor: her enkf_update(...) -> (analysis_vec, new_cov_X)

def save_state(path, **arrays): ...    # enkf.save_enkf_state
def load_state(path): ...         # enkf.load_enkf_state

