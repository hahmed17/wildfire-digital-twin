"""
§3 checks.py — pure signal functions, no LLM, no FARSITE.

Each function computes one field of Signals (state.py). All geometry is in
EPSG:5070 (metres). Inputs are either numpy state vectors (2*n_vertex, 1) or
shapely Polygons; callers convert as needed.

These are forecast-independent where noted — the observation agent can call
them before the forecast runs. normalized_innovation requires the ensemble
and is called by the assimilation agent after the forecast.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely import make_valid


# ---------------------------------------------------------------------------
# Geometry helpers (shared with synthetic.py — kept local to avoid a circular
# import; these are tiny)
# ---------------------------------------------------------------------------

def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return geom
    parts = [g for g in getattr(geom, "geoms", [])
             if isinstance(g, (Polygon, MultiPolygon))]
    flat = [p for g in parts
            for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
    return max(flat, key=lambda p: p.area) if flat else None


def _state_to_polygon(state_vec: np.ndarray) -> Polygon | None:
    """Convert a (2*n_vertex, 1) or (2*n_vertex,) state vector to a shapely Polygon."""
    v = np.asarray(state_vec).ravel()
    if len(v) < 6:
        return None
    return _largest_polygon(Polygon(np.c_[v[0::2], v[1::2]]))


# ---------------------------------------------------------------------------
# § normalized_innovation  (forecast-dependent — assimilation agent only)
# ---------------------------------------------------------------------------

def normalized_innovation(
    forecast_ensemble: np.ndarray,
    obs_vec: np.ndarray,
    vsize_m: float,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Measure how surprising the observation is relative to the forecast ensemble.

    Score ≈ 1  → the observation is consistent with the ensemble spread.
    Score >> 1 → the observation is far outside what the ensemble predicted.
    Score = 0  → all valid members are identical (degenerate ensemble).

    Defined as ||innovation|| / (||ensemble_spread|| + vsize_m), where:
      innovation      = obs - ensemble_mean  (forecast-space distance)
      ensemble_spread = mean column norm of (Z - Z_mean)

    Args:
        forecast_ensemble: (n_states, N) zkphat from run_ensemble
        obs_vec:           (n_states, 1) or (n_states,) observed state
        vsize_m:           Observation noise std (m) — the denominator floor
        valid_mask:        (N,) bool; None = all members valid

    Returns:
        float >= 0
    """
    Z   = np.asarray(forecast_ensemble)                       # (n_states, N)
    obs = np.asarray(obs_vec).ravel()[:, np.newaxis]          # (n_states, 1)

    if valid_mask is not None:
        Z = Z[:, np.asarray(valid_mask, dtype=bool)]
    if Z.shape[1] == 0:
        return float("inf")

    z_mean    = Z.mean(axis=1, keepdims=True)                 # (n_states, 1)
    innovation = obs - z_mean                                  # (n_states, 1)
    inn_norm   = float(np.linalg.norm(innovation))

    # Ensemble spread: mean column norm of anomalies
    Z_anom  = Z - z_mean                                      # (n_states, Ne)
    spread  = float(np.linalg.norm(Z_anom, axis=0).mean())   # scalar

    denom = spread + float(vsize_m)
    return inn_norm / denom if denom > 0 else float("inf")


# ---------------------------------------------------------------------------
# § contains_previous  (forecast-independent)
# ---------------------------------------------------------------------------

def contains_previous(
    perimeter: Polygon,
    prev_perimeter: Polygon,
    tol: float = 0.90,
) -> bool:
    """
    Check whether the new perimeter contains (most of) the previous one.

    Fires virtually never shrink, so a new perimeter that does not contain the
    previous one is likely a bad observation. The `tol` fraction allows for
    minor GPS/mapping noise at the edge.

    Args:
        perimeter:      New observation polygon (EPSG:5070, metres)
        prev_perimeter: Previous accepted polygon (EPSG:5070, metres)
        tol:            Fraction of prev_perimeter area that must fall inside
                        the new perimeter (0.90 = 90 % threshold)

    Returns:
        True if the containment fraction >= tol, or if prev_perimeter is None/empty.
    """
    if prev_perimeter is None or prev_perimeter.is_empty:
        return True   # no previous observation to check against
    if perimeter is None or perimeter.is_empty:
        return False

    intersection = perimeter.intersection(prev_perimeter)
    if intersection.is_empty:
        return False

    fraction = intersection.area / prev_perimeter.area
    return fraction >= tol


# ---------------------------------------------------------------------------
# § growth_plausible  (forecast-independent)
# ---------------------------------------------------------------------------

# Physical upper bound: ROS ~200 m/min is an extreme crown fire (e.g. 2003 Canberra).
# Using 300 m/min as a hard ceiling to catch implausible perimeter jumps.
MAX_SPREAD_RATE_M_PER_MIN = 300.0


def growth_plausible(
    perimeter: Polygon,
    prev_perimeter: Polygon,
    dt_minutes: float,
    wind_speed_mph: float = 0.0,
    max_rate_m_per_min: float = MAX_SPREAD_RATE_M_PER_MIN,
) -> bool:
    """
    Check whether the implied spread rate is physically possible.

    Compares the equivalent-radius growth between the two perimeters against a
    wind-scaled ceiling (see spread_ceiling).

    Deliberately permissive: it is a sanity check for gross errors — a perimeter
    from the wrong fire, a units mix-up, a decimal slip — not a tight bound on
    spread. A false rejection throws away real data, which costs more than
    letting a merely fast-looking perimeter through. Displacement, where the
    perimeter is the right size but in the wrong place, is caught by
    contains_previous rather than here.

    Args:
        perimeter:         New observation polygon (EPSG:5070, metres)
        prev_perimeter:    Previous accepted polygon (EPSG:5070, metres)
        dt_minutes:        Time elapsed between the two observations (minutes)
        wind_speed_mph:    Mean wind speed for the interval (mph, 0 = calm)
        max_rate_m_per_min: Hard upper ceiling regardless of wind

    Returns:
        True if the spread rate is physically plausible.
    """
    if prev_perimeter is None or prev_perimeter.is_empty:
        return True
    if perimeter is None or perimeter.is_empty:
        return False
    if dt_minutes <= 0:
        return True   # no time elapsed: can't judge

    r_new  = math.sqrt(perimeter.area      / math.pi)
    r_prev = math.sqrt(prev_perimeter.area / math.pi)
    growth_m = max(0.0, r_new - r_prev)
    rate_m_per_min = growth_m / dt_minutes

    return rate_m_per_min <= spread_ceiling(wind_speed_mph, max_rate_m_per_min)


# Ceiling at calm, and how much each mph of 20-ft wind adds. Roughly linear in wind,
# which is the shape Rothermel's spread model gives over the range that matters here:
#   calm -> 30 m/min, 10 mph -> 130, 25 mph -> 280, 27+ mph -> capped at 300.
CALM_RATE_M_PER_MIN = 30.0
RATE_PER_MPH = 10.0


def spread_ceiling(wind_speed_mph: float,
                   max_rate_m_per_min: float = MAX_SPREAD_RATE_M_PER_MIN) -> float:
    """Fastest equivalent-radius growth (m/min) treated as physically possible."""
    return min(max_rate_m_per_min,
               CALM_RATE_M_PER_MIN + RATE_PER_MPH * max(0.0, wind_speed_mph))


# ---------------------------------------------------------------------------
# § agrees_with_firms  (forecast-independent)
# ---------------------------------------------------------------------------

def _hotspot_timestamps(df, dt_col: str = "acq_datetime"):
    """Acquisition timestamps from a FIRMS table, as a tz-naive pandas Series.

    FIRMS downloads carry `acq_date` plus `acq_time` as an HHMM integer (945 means
    09:45), not a single datetime column. data/border2_inputs/firms_hotspots.csv is
    in that raw form, so accept either shape. Returns None if neither is present.
    """
    import pandas as pd

    if dt_col in df.columns:
        ts = pd.to_datetime(df[dt_col], errors="coerce")
    elif "acq_date" in df.columns and "acq_time" in df.columns:
        hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
        ts = pd.to_datetime(df["acq_date"].astype(str) + " " + hhmm,
                            format="%Y-%m-%d %H%M", errors="coerce")
    elif "acq_date" in df.columns:
        ts = pd.to_datetime(df["acq_date"], errors="coerce")
    else:
        return None

    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return None if ts.isna().all() else ts


def load_hotspots(
    path: str,
    start: datetime,
    end: datetime,
    source_col: str = "source",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    dt_col: str = "acq_datetime",
    buffer_hours: float = 2.0,
) -> list[tuple[float, float]] | None:
    """
    Load FIRMS hotspot coordinates from a CSV file, filtered to the time window.

    Returns a list of (lon, lat) tuples in WGS84, or None if the file cannot
    be read or the filtered result is empty.

    Args:
        path:          Path to the FIRMS CSV (output of border2_prepare_data.ipynb)
        start:         Window start (tz-aware or naive; matched naively if naive)
        end:           Window end
        source_col:    Column that identifies the source (VIIRS / MODIS); not filtered
        lat_col:       Latitude column name
        lon_col:       Longitude column name
        dt_col:        Datetime column name (parsed by pandas)
        buffer_hours:  Hours of padding beyond [start, end] to catch near-window detections
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None

    ts = _hotspot_timestamps(df, dt_col)   # already tz-naive
    if ts is None:
        return None
    start_naive = start.replace(tzinfo=None) if hasattr(start, "tzinfo") and start.tzinfo else start
    end_naive   = end.replace(tzinfo=None)   if hasattr(end,   "tzinfo") and end.tzinfo   else end
    buf = pd.Timedelta(hours=buffer_hours)
    mask = (ts >= pd.Timestamp(start_naive) - buf) & (ts <= pd.Timestamp(end_naive) + buf)
    sub = df[mask]
    if sub.empty or lat_col not in sub.columns or lon_col not in sub.columns:
        return None

    return list(zip(sub[lon_col], sub[lat_col]))


def agrees_with_firms(
    perimeter: Polygon,
    hotspots_wgs84: list[tuple[float, float]] | None,
    buffer_m: float = 500.0,
) -> bool | None:
    """
    Check whether satellite hotspots fall inside (or near) the observed perimeter.

    This is forecast-independent: it only compares the observation against
    satellite active-fire detections.

    Args:
        perimeter:        Observed polygon in EPSG:5070 (metres)
        hotspots_wgs84:   List of (lon, lat) tuples in WGS84 from load_hotspots,
                          or None if no detections are available.
        buffer_m:         Buffer (m) added to the perimeter before checking;
                          accounts for VIIRS/MODIS pixel-centre offset (~375 m / ~500 m)

    Returns:
        True  — at least one hotspot falls inside the buffered perimeter
        False — hotspots exist but none fall inside
        None  — no hotspots available for this window (can't judge)
    """
    if hotspots_wgs84 is None or len(hotspots_wgs84) == 0:
        return None
    if perimeter is None or perimeter.is_empty:
        return False

    # Reproject hotspots from WGS84 to EPSG:5070 for spatial comparison
    try:
        import pyproj
        from shapely.ops import transform as shp_transform
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070",
                                                   always_xy=True)
        buffered = perimeter.buffer(buffer_m)
        for lon, lat in hotspots_wgs84:
            x, y = transformer.transform(lon, lat)
            from shapely.geometry import Point
            if buffered.contains(Point(x, y)):
                return True
        return False
    except Exception:
        return None   # treat reprojection failure as "no data"


# ---------------------------------------------------------------------------
# § wind_shift  (forecast-independent)
# ---------------------------------------------------------------------------

def wind_shift(
    weather_prev: dict | None,
    weather_now: dict,
    deg_thresh: float = 45.0,
    speed_thresh_mph: float = 10.0,
) -> bool:
    """
    Detect a significant wind shift between two consecutive weather windows.

    A shift is flagged when the direction rotates by more than `deg_thresh`
    degrees OR the speed changes by more than `speed_thresh_mph` mph.

    Args:
        weather_prev:      Weather dict from the previous cycle's cache["weather_last"],
                           or None if this is the first cycle.
        weather_now:       Weather dict from the current fetch_weather call.
        deg_thresh:        Direction change threshold (degrees; default 45°)
        speed_thresh_mph:  Speed change threshold (mph; default 10 mph)

    Returns:
        True if a wind shift is detected.
    """
    if weather_prev is None:
        return False   # no previous window to compare

    def _mean_speed(w: dict) -> float:
        speeds = w.get("wind_speeds", [])
        return float(np.mean(speeds)) if speeds else 0.0

    def _mean_dir_deg(w: dict) -> float:
        dirs = w.get("wind_directions", [])
        if not dirs:
            return 0.0
        rads = np.deg2rad(dirs)
        return float(math.degrees(math.atan2(np.sin(rads).mean(),
                                             np.cos(rads).mean())) % 360)

    spd_prev = _mean_speed(weather_prev)
    spd_now  = _mean_speed(weather_now)
    dir_prev = _mean_dir_deg(weather_prev)
    dir_now  = _mean_dir_deg(weather_now)

    spd_delta = abs(spd_now - spd_prev)

    # Circular difference (always in [0, 180])
    raw_diff = abs(dir_now - dir_prev) % 360
    dir_delta = min(raw_diff, 360 - raw_diff)

    return dir_delta >= deg_thresh or spd_delta >= speed_thresh_mph