"""
WIFIRE Firemap data retrieval utilities.
https://firemap.sdsc.edu/

Provides two functions:
  - fetch_fire_perimeters()  : Historical fire perimeter polygons via WFS
  - fetch_weather()          : Weather observations via pylaski station API
"""

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from config import FIREMAP_WFS_URL, FIREMAP_WX_URL, FARSITE_CRS


# ============================================================================
# PERIMETER RETRIEVAL
# ============================================================================

def _multipolygon_to_polygon(geom):
    """Return the largest polygon from a MultiPolygon, or the polygon itself."""
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


def fetch_fire_perimeters(fire_name, year, verbose=True):
    """
    Fetch all mapped perimeters for a fire from WIFIRE Firemap GeoServer (WFS).

    Args:
        fire_name: Fire name exactly as it appears in the database (e.g. "BORDER 2")
        year: Fire year (e.g. 2025)
        verbose: Print progress

    Returns:
        GeoDataFrame with columns including 'datetime', 'acres', 'geometry',
        in EPSG:5070, sorted oldest to newest.
    """
    if verbose:
        print(f"Fetching perimeters for '{fire_name}' ({year})...")

    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    "WIFIRE:view_historical_fires",
        "CQL_FILTER":   f"fire_name = '{fire_name}' AND year = {year}",
        "OUTPUTFORMAT": "application/json",
        "SRSNAME":      "EPSG:4326",
    }

    response = requests.get(FIREMAP_WFS_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    if not features:
        raise ValueError(
            f"No perimeters found for fire_name='{fire_name}', year={year}.\n"
            f"Check the fire name is an exact case-sensitive match."
        )

    if verbose:
        print(f"  Retrieved {len(features)} perimeter(s)")

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    # Parse datetime — format is "2025-01-24Z" or "2025-01-24T00:00:00Z"
    gdf['datetime'] = pd.to_datetime(
        gdf['perimeter_timestamp'].str.rstrip('Z'), utc=False
    )

    # MultiPolygon -> largest single Polygon
    gdf['geometry'] = gdf['geometry'].apply(_multipolygon_to_polygon)

    # Sort oldest to newest and reindex
    gdf = gdf.sort_values('datetime', ascending=True).reset_index(drop=True)

    # Reproject to FARSITE CRS
    gdf = gdf.to_crs(FARSITE_CRS)

    if verbose:
        print(f"\n✓ {len(gdf)} perimeters ready")
        print(f"  Oldest: {gdf['datetime'].iloc[0]}")
        print(f"  Newest: {gdf['datetime'].iloc[-1]}")
        print(f"  Area range: "
              f"{gdf.geometry.area.min()/1e6:.2f} – "
              f"{gdf.geometry.area.max()/1e6:.2f} km²")
        print(f"\n  Perimeter timeline:")
        for i, row in gdf.iterrows():
            print(f"    [{i}] {row['datetime'].date()}  —  "
                  f"{row.geometry.area/1e6:.2f} km²  "
                  f"({row['acres']:.0f} acres)")

    return gdf


# ============================================================================
# WEATHER RETRIEVAL
# ============================================================================

def fetch_weather(lat, lon, start_dt, end_dt, verbose=True):
    """
    Fetch weather observations from WIFIRE Firemap pylaski station API.

    Queries the nearest weather stations to the given location and returns
    wind speed and direction observations for the given time window.

    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)
        start_dt: Start datetime (datetime object or ISO string)
        end_dt: End datetime (datetime object or ISO string)
        verbose: Print progress

    Returns:
        dict with keys:
            'windspeed'     : wind speed in mph (float)
            'winddirection' : wind direction in degrees (float)
            'observations'  : raw DataFrame of all observations
        Falls back to config defaults if no data is retrieved.
    """
    from config import DEFAULT_HUMIDITY, DEFAULT_TEMPERATURE

    if verbose:
        print(f"  Querying weather: {start_dt} to {end_dt}")

    # Convert datetimes to strings if needed
    if hasattr(start_dt, 'strftime'):
        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
        end_str   = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        start_str = str(start_dt)
        end_str   = str(end_dt)

    params = {
        'latitude':  lat,
        'longitude': lon,
        'start':     start_str,
        'end':       end_str,
        'features':  'wind',
    }

    try:
        response = requests.get(FIREMAP_WX_URL, params=params, timeout=15)
        response.raise_for_status()
        wx_data = response.json()

        features = wx_data.get('features', [])
        if not features:
            raise ValueError('features')

        # Parse observations into a flat DataFrame
        records = []
        for station in features:
            props = station.get('properties', {})
            obs_list = props.get('observations', [])
            for obs in obs_list:
                records.append({
                    'station':       props.get('stationName', ''),
                    'datetime':      pd.to_datetime(obs.get('date')),
                    'windspeed':     obs.get('windSpeed'),
                    'winddirection': obs.get('windDirection'),
                })

        if not records:
            raise ValueError('no observations parsed')

        obs_df = pd.DataFrame(records).dropna(subset=['windspeed', 'winddirection'])

        if obs_df.empty:
            raise ValueError('all observations are NaN')

        # Use the mean over the window
        ws = float(obs_df['windspeed'].mean())
        wd = float(obs_df['winddirection'].mean())

        if verbose:
            print(f"  Retrieved {len(obs_df)} observations")
            print(f"  Wind: {ws:.1f} mph @ {wd:.0f}°")

        return {
            'windspeed':     ws,
            'winddirection': wd,
            'observations':  obs_df,
        }

    except Exception as e:
        if verbose:
            print(f"  WARNING: Weather query failed: {e}")
            print(f"  Using fallback values")

        from config import DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY
        return {
            'windspeed':     5.0,   # mph fallback
            'winddirection': 270.0, # degrees fallback (westerly)
            'observations':  pd.DataFrame(),
        }
