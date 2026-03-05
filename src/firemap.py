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

import time
import json
import numpy as np


from config import FIREMAP_WFS_URL, FIREMAP_WX_URL, FARSITE_CRS
import warnings

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


def fetch_fire_perimeters(fire_name, year=2025, geoserver_layer='WIFIRE:view_historical_fires', verbose=True):
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
        print(f"Fetching perimeters for '{fire_name}' ({year})...")

    params = {
        "SERVICE":      "WFS",
        "VERSION":      "2.0.0",
        "REQUEST":      "GetFeature",
        "TYPENAMES":    f"{geoserver_layer}",
        "CQL_FILTER":   f"fire_name = '{fire_name}' AND year = {year}",  # Add year filter after updating data
        "OUTPUTFORMAT": "application/json",
        "SRSNAME":      "EPSG:4326",
    }

    warnings.warn(
        "The 'year' parameter is currently not used for filtering perimeters. "
        "Ensure the fire name is unique or includes the year if needed."
    )

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
                #   f"({row['acres']:.0f} acres)"
                    )
    return gdf


# ============================================================================
# WEATHER RETRIEVAL
# ============================================================================
def query_weather_for_timestep(lat, lon, start_time, end_time, verbose=False):
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

def create_bbox_from_point(lon, lat, radius_in_miles=10.0, write_geojson=False, output_path="initial_bbox.geojson"):
    """
    Create a bounding box with center point (lon, lat) and radius buffer.
    
    Args:
        lon: Longitude of center point (WGS84)
        lat: Latitude of center point (WGS84)
        radius_in_miles: Radius of bounding box in miles (default: 10.0)
        write_geojson: Whether to write bbox to GeoJSON file (default: False)
        output_path: Path for output GeoJSON if write_geojson=True
        
    Returns:
        GeoDataFrame with bounding box in EPSG:5070 (FARSITE CRS)
    """
    # Create center point
    center_point = Point(lon, lat)
    pt = gpd.GeoSeries([center_point], crs="EPSG:4326")
    
    # Project to local UTM for accurate buffering
    pt_utm = pt.to_crs(pt.estimate_utm_crs())
    
    # Create buffer
    radius_meters = radius_in_miles * 1609.344
    buffer_utm = pt_utm.buffer(radius_meters)
    
    # Get bounding box
    minx, miny, maxx, maxy = buffer_utm.total_bounds
    
    # Convert bbox corners back to lon/lat
    corners_utm = gpd.GeoSeries(
        [Point(minx, miny), Point(maxx, maxy)],
        crs=pt_utm.crs
    ).to_crs("EPSG:4326")
    
    min_lon, min_lat = corners_utm.iloc[0].x, corners_utm.iloc[0].y
    max_lon, max_lat = corners_utm.iloc[1].x, corners_utm.iloc[1].y
    
    # Build bbox polygon in WGS84
    bbox_polygon_wgs84 = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])
    
    # Create GeoDataFrame in WGS84 then convert to FARSITE CRS
    bbox_gdf = gpd.GeoDataFrame(
        [{'type': 'bounding_box', 'radius_miles': radius_in_miles}],
        geometry=[bbox_polygon_wgs84],
        crs="EPSG:4326"
    )
    bbox_gdf = bbox_gdf.to_crs(FARSITE_CRS)
    
    # Optional: Write to GeoJSON
    if write_geojson:
        bbox_feature = geojson.Feature(
            geometry=bbox_polygon_wgs84,
            properties={
                "type": "bounding_box",
                "radius_miles": radius_in_miles,
                "center_lon": lon,
                "center_lat": lat,
            }
        )
        feature_collection = geojson.FeatureCollection([bbox_feature])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(feature_collection, f, indent=2)
        print(f"✓ Bounding box saved to {output_path}")
    
    return bbox_gdf




def verify_landscape_file(lcp_path):
    """
    Verify that a landscape (.lcp) file exists.
    
    Args:
        lcp_path: Path to landscape file
        
    Returns:
        bool: True if file exists, False otherwise
    """
    exists = Path(lcp_path).exists()
    
    if exists:
        print(f"✓ Landscape file found: {lcp_path}")
    else:
        print(f"✗ Landscape file not found: {lcp_path}")
        print("You need to generate or download a .lcp file for your domain.")
        print("See FARSITE documentation for landscape file creation.")
    
    return exists



def extract_fire_timeline(perimeters_gdf, verbose=True):
    """
    Extract ignition and containment dates from perimeter GeoDataFrame.
    
    Args:
        perimeters_gdf: GeoDataFrame with perimeter updates (must have 'datetime' column)
        verbose: Print timeline information
        
    Returns:
        dict with keys: 'ignition_date', 'containment_date', 'duration', 'n_updates'
    """
    ignition_date = perimeters_gdf['datetime'].iloc[0]
    containment_date = perimeters_gdf['datetime'].iloc[-1]
    duration = containment_date - ignition_date
    n_updates = len(perimeters_gdf)
    
    timeline = {
        'ignition_date': ignition_date,
        'containment_date': containment_date,
        'duration': duration,
        'n_updates': n_updates
    }
    
    if verbose:
        print(f"\nFire Timeline:")
        print(f"  First observation (ignition): {ignition_date}")
        print(f"  Last observation (containment): {containment_date}")
        print(f"  Total duration: {duration}")
        print(f"  Number of updates: {n_updates}")
    
    return timeline



def fetch_fire_perimeters_firms_ogc(fire_name="BORDER 2", bbox=None, 
                                    start_date=None, end_date=None,
                                    fire_id=None, verbose=True):
    """
    Fetch fire perimeters from FIRMS OGC API.
    
    Args:
        fire_name: Name of fire (for display only)
        bbox: Bounding box [minLon, minLat, maxLon, maxLat] in WGS84
        start_date: Start date (YYYY-MM-DD or datetime)
        end_date: End date (YYYY-MM-DD or datetime)
        fire_id: Fire ID to filter (required for accurate results)
        verbose: Print progress
        
    Returns:
        GeoDataFrame with perimeters (already in EPSG:5070, sorted oldest→newest)
    """
    if verbose:
        print(f"Fetching perimeters for: {fire_name}")
        print(f"Data source: FIRMS OGC API (actual perimeter polygons)")
    
    # Border 2 defaults
    if bbox is None:
        bbox = [-117.36, 32.54, -116.04, 33.31]  # San Diego County
    
    if start_date is None:
        start_date = "2025-01-23"
    
    if end_date is None:
        end_date = "2025-01-30"
    
    # Convert dates to ISO format
    if isinstance(start_date, str) and 'T' not in start_date:
        start_date = f"{start_date}T00:00:00"
    if isinstance(end_date, str) and 'T' not in end_date:
        end_date = f"{end_date}T23:59:59"
    
    # Fetch from FIRMS
    from firms_utils import FIRMSPerimeters
    
    client = FIRMSPerimeters()
    perimeters_gdf = client.fetch_fire_perimeters(
        bbox=bbox,
        start_datetime=start_date,
        end_datetime=end_date,
        fire_id=fire_id,
        progress=verbose
    )
    
    # Already sorted oldest→newest and in EPSG:5070
    if verbose:
        print(f"\n✓ Retrieved {len(perimeters_gdf)} perimeter updates")
        print(f"Order: Index 0 = oldest, Index {len(perimeters_gdf)-1} = newest")
    
    return perimeters_gdf




def fetch_weather_data(lat, lon, start_time, end_time, verbose=True):
    """
    Fetch weather observations from Firemap API.
    
    Args:
        lat: Latitude (WGS84)
        lon: Longitude (WGS84)
        start_time: Start datetime (ISO format string or datetime)
        end_time: End datetime (ISO format string or datetime)
        verbose: Print progress messages
        
    Returns:
        dict with keys: 'location', 'time_range', 'wind_speed', 'wind_direction'
    """
    # Convert datetimes to ISO format if needed
    if isinstance(start_time, pd.Timestamp):
        start_time = start_time.isoformat()
    if isinstance(end_time, pd.Timestamp):
        end_time = end_time.isoformat()
    
    if verbose:
        print(f"Querying weather data...")
        print(f"  Location: {lat:.4f}, {lon:.4f}")
        print(f"  From: {start_time}")
        print(f"  To: {end_time}")
    
    # Query Firemap API
    timestamp = int(time.time() * 1000)
    wx_params = {
        'selection': 'closestTo',
        'lat': str(lat),
        'lon': str(lon),
        'observable': ['wind_speed', 'wind_direction'],
        'from': start_time,
        'to': end_time,
        'callback': 'wxData',
        '_': str(timestamp)
    }
    
    wx_response = requests.get(FIREMAP_WX_URL, params=wx_params)
    wx_text = wx_response.text.strip()
    
    # Remove JSONP wrapper
    if wx_text.startswith('wxData(') and wx_text.endswith(')'):
        wx_json = wx_text[len('wxData('):-1]
        wx_obs = json.loads(wx_json)
    else:
        wx_obs = wx_response.json()
    
    # Extract wind data
    wind_speed_list = wx_obs["features"][0]["properties"]["wind_speed"]
    wind_direction_list = wx_obs["features"][0]["properties"]["wind_direction"]
    
    weather_data = {
        "location": {"lat": lat, "lon": lon},
        "time_range": {"start": start_time, "end": end_time},
        "wind_speed": wind_speed_list,
        "wind_direction": wind_direction_list
    }
    
    if verbose:
        print(f"\n✓ Retrieved {len(wind_speed_list)} weather observations")
        print(f"  Wind speed: {min(wind_speed_list):.1f} - {max(wind_speed_list):.1f} mph (mean: {np.mean(wind_speed_list):.1f})")
        print(f"  Wind direction: {min(wind_direction_list):.0f} - {max(wind_direction_list):.0f}° (mean: {np.mean(wind_direction_list):.0f})")
    
    return weather_data


def get_weather_location_from_fire(perimeters_gdf, to_wgs84=True):
    """
    Get weather query location from fire perimeter centroid.
    
    Args:
        perimeters_gdf: GeoDataFrame with fire perimeters
        to_wgs84: Convert to WGS84 coordinates (required for weather API)
        
    Returns:
        tuple: (lat, lon) in WGS84 if to_wgs84=True, else in original CRS
    """
    # Use first perimeter centroid
    fire_centroid = perimeters_gdf.geometry.iloc[0].centroid
    
    if to_wgs84:
        # Convert to WGS84 for weather API
        transformer = Transformer.from_crs(perimeters_gdf.crs, "EPSG:4326", always_xy=True)
        lon_wgs, lat_wgs = transformer.transform(fire_centroid.x, fire_centroid.y)
        return lat_wgs, lon_wgs
    else:
        return fire_centroid.y, fire_centroid.x



def plot_perimeter_evolution(perimeters_gdf, fire_name="Fire", add_basemap=True):
    """
    Plot fire perimeter evolution over time with OpenStreetMap basemap.
    
    Args:
        perimeters_gdf: GeoDataFrame with perimeter updates (must have 'datetime' column)
        fire_name: Name of fire for plot title
        add_basemap: Add OpenStreetMap basemap (default: True)
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    n_perims = len(perimeters_gdf)
    colors = plt.cm.Reds(np.linspace(0.3, 1, n_perims))
    
    # Plot perimeters
    for idx, (_, row) in enumerate(perimeters_gdf.iterrows()):
        boundary = row.geometry.boundary
        if boundary.geom_type == 'LineString':
            x, y = boundary.xy
            ax.plot(x, y, color=colors[idx], linewidth=2.5, zorder=2)
        elif boundary.geom_type == 'MultiLineString':
            for line in boundary.geoms:
                x, y = line.xy
                ax.plot(x, y, color=colors[idx], linewidth=2.5, zorder=2)
    
    # Add basemap
    if add_basemap:
        try:
            ctx.add_basemap(
                ax=ax, 
                source=ctx.providers.OpenStreetMap.Mapnik, 
                crs=FARSITE_CRS,
                alpha=0.6,
                zorder=1
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    ax.set_aspect('equal')
    
    start_date = perimeters_gdf['datetime'].iloc[0].date()
    end_date = perimeters_gdf['datetime'].iloc[-1].date()
    ax.set_title(f"{fire_name} - Perimeter Evolution\n{start_date} to {end_date}", fontsize=16, weight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=n_perims-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Time progression (older → newer)', fontsize=12)
    
    plt.tight_layout()
    plt.show()



def plot_weather_data(weather_data):
    """
    Plot weather observations (wind speed and direction).
    
    Args:
        weather_data: Dictionary returned by fetch_weather_data()
    """
    wind_speed = weather_data['wind_speed']
    wind_direction = weather_data['wind_direction']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(wind_speed, 'b-', linewidth=1)
    ax1.axhline(np.mean(wind_speed), color='r', linestyle='--', 
               label=f'Mean: {np.mean(wind_speed):.1f} mph')
    ax1.set_ylabel('Wind Speed (mph)')
    ax1.set_title('Weather Observations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wind_direction, 'g-', linewidth=1)
    ax2.axhline(np.mean(wind_direction), color='r', linestyle='--', 
               label=f'Mean: {np.mean(wind_direction):.0f}°')
    ax2.set_ylabel('Wind Direction (degrees)')
    ax2.set_xlabel('Observation Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def save_workflow_config(fire_name, lcp_path, timeline, weather_location, 
                        domain_bounds=None, output_path=None):
    """
    Save complete workflow configuration.
    
    Args:
        fire_name: Name of fire
        lcp_path: Path to landscape file
        timeline: Dictionary from extract_fire_timeline()
        weather_location: (lat, lon) tuple
        domain_bounds: Optional domain bounds
        output_path: Output path (default: DATA_DIR/workflow_config.json)
        
    Returns:
        Path to saved configuration file
    """
    config_data = {
        # Static data
        "lcp_path": str(lcp_path),
        "domain_bounds": domain_bounds,
        "crs": FARSITE_CRS,
        
        # Fire information
        "fire_name": fire_name,
        "ignition_date": timeline['ignition_date'].isoformat(),
        "containment_date": timeline['containment_date'].isoformat(),
        "n_perimeters": timeline['n_updates'],
        
        # Weather location
        "weather_location": {"lat": weather_location[0], "lon": weather_location[1]},
    }
    
    if output_path is None:
        output_path = DATA_DIR / "workflow_config.json"
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✓ Configuration saved to {output_path}")
    return output_path


def save_perimeters(perimeters_gdf, fire_name, output_dir=None):
    """
    Save perimeter GeoDataFrame to file.
    
    Args:
        perimeters_gdf: GeoDataFrame with perimeters
        fire_name: Name of fire
        output_dir: Output directory (default: OUTPUT_DIR)
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    filename = f"{fire_name.lower().replace(' ', '_')}_perimeters.geojson"
    output_path = output_dir / filename
    
    perimeters_gdf.to_file(output_path, driver="GeoJSON")
    
    print(f"✓ Perimeters saved to {output_path}")
    print(f"  Order: Index 0 = oldest (ignition), Index {len(perimeters_gdf)-1} = newest (containment)")
    
    return output_path


def save_weather_data(weather_data, output_dir=None):
    """
    Save weather data to JSON file.
    
    Args:
        weather_data: Dictionary from fetch_weather_data()
        output_dir: Output directory (default: DATA_DIR)
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    output_path = output_dir / "weather_observations.json"
    
    with open(output_path, 'w') as f:
        json.dump(weather_data, f, indent=2)
    
    print(f"✓ Weather data saved to {output_path}")
    
    return output_path