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
# ACTIVE FIRE DETECTIONS
# ============================================================================
def get_fire_detections(firms_map_key, bbox_str, start_date, satellite_source="LANDSAT_NRT", day_range=5):
    """
    Fetch active fire detections from NASA FIRMS API. 
    
    Args:
        firms_map_key (str): Access key to query NASA FIRMS API.
        bbox (str): "minLon, minLat, maxLon, maxLat" (WGS84)
        start_time (str): Start datetime ("%Y-%m-%d")
        satellite_source (str): satelite source name (options: https://firms.modaps.eosdis.nasa.gov/api/area/)
            default: "LANDSAT_NRT" (US/Canada only)
        day_range (int): number between 1-5 of days to query
            default: 5
        
    Returns:
        dict
    """
    # Set up base URL
    FIRMS_API_URL = f"https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/{firms_map_key}/{satellite_source}/{bbox}/{day_range}/{start_date}"

    try:
        response = requests.get(FIRMS_API_URL, timeout=30)
        response.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        csv_data = StringIO(response.text)
        
        # Read into DataFrame
        hotspots_df = pd.read_csv(csv_data)
        
        print(f"\n✓ Retrieved {len(hotspots_df)} fire detections")
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print("\n⚠ No fire detections found in this area and time range")
            hotspots_df = pd.DataFrame()
        else:
            print(f"\n❌ API Error: {e}")
            print("Check your MAP_KEY and try again")
            raise
    
    # Display dataframe keys
    if not hotspots_df.empty:
        col_names = list(hotspots_df.columns.values)
        print(f"\nData columns:\n{col_names}")
    
    return hotspots_df






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


def fetch_fire_perimeters(fire_name, verbose=True, synthetic=False):
    """
    Fetch all mapped perimeters for a fire from WIFIRE Firemap GeoServer (WFS).

    Args:
        fire_name: Fire name exactly as it appears in the database (e.g. "BORDER 2")
        year: Fire year (e.g. 2025)
        verbose: Print progress
        synthetic: Loads the synthetic fire from the local filesystem; no Firemap query

    Returns:
        GeoDataFrame with columns including 'datetime', 'acres', 'geometry',
        in EPSG:5070, sorted oldest to newest.
    """
    if verbose:
        print(f"Fetching perimeters for '{fire_name}')...")

    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    "WIFIRE:view_historical_fires",
        "CQL_FILTER":   f"fire_name = '{fire_name}'",
        "OUTPUTFORMAT": "application/json",
        "SRSNAME":      "EPSG:4326",
    }

    if synthetic:
        raise NotImplementedError('Load synthetic fire perimeter not yet implemented!!')
        data = load_synthetic_fire_perimeter()
    else:
        response = requests.get(FIREMAP_WFS_URL, params=params, timeout=30)
        response.raise_for_status()
        print(f"\nPerimeters found at: {response.url}")
        data = response.json()

    features = data.get("features", [])
    if not features:
        raise ValueError(
            f"No perimeters found for fire_name='{fire_name}'.\n"
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
def query_weather_for_timestep(lat, lon, start_time, end_time, verbose=False, synthetic=False):
    """
    Query weather data for a specific timestep.
    
    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)
        start_time: Start datetime (pandas Timestamp)
        end_time: End datetime (pandas Timestamp)
        verbose: Print query details
        synthetic: Loads the synthetic weather data from the local filesystem
        
    Returns:
        tuple: (wind_speed_list, wind_direction_list)
    """
    # Convert to ISO format
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()
    
    if verbose:
        print(f"  Querying weather: {start_time} to {end_time}")


    if synthetic:
        wind_speed_list = [20]
        wind_direction_list = [90]

        return wind_speed_list, wind_direction_list
    else:
        # Query Firemap API
        timestamp = int(time.time() * 1000)
        wx_params = {
            'selection': 'closestTo',
            'lat': str(lat),
            'lon': str(lon),
            'observable': ['wind_speed', 'wind_direction'],
            'from': start_iso,
            'to': end_iso,
            'callback': 'wxData',
            '_': str(timestamp)
        }

        try:
            wx_response = requests.get(FIREMAP_WX_URL, params=wx_params, timeout=10)
            wx_text = wx_response.text.strip()
            
            # Remove JSONP wrapper
            if wx_text.startswith('wxData(') and wx_text.endswith(')'):
                wx_json = wx_text[len('wxData('):-1]
                wx_obs = json.loads(wx_json)
            else:
                wx_obs = wx_response.json()
            
            wind_speed_list = wx_obs["features"][0]["properties"]["wind_speed"]
            wind_direction_list = wx_obs["features"][0]["properties"]["wind_direction"]
            
            if verbose:
                print(f"  Retrieved {len(wind_speed_list)} observations")
                print(f"  Wind: {np.mean(wind_speed_list):.1f} mph @ {np.mean(wind_direction_list):.0f}°")
            
            return wind_speed_list, wind_direction_list
            
        except Exception as e:
            print(f"  WARNING: Weather query failed: {e}")
            print(f"  Using fallback values")
            # Return fallback values
            return [10.0], [225.0]  # Default 10 mph from SW


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
