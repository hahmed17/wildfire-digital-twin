"""
Synthetic replacement for a rejected observation: grow the last accepted perimeter.

Independent of the FARSITE forecast, so assimilating it still adds information:
  - spread rate: the fire's own recent growth (equivalent-radius growth between the last two
    accepted perimeters), or DEFAULT_RATE_M_PER_H when there is only one;
  - shape: stretched downwind by Anderson's (1983) length-to-breadth ratio for the wind,
    the fire-shape relation FARSITE itself uses.
All geometry is in EPSG:5070 (metres). No FARSITE / src imports.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely import make_valid

from orchestrator.state import ObservationEvent

DEFAULT_RATE_M_PER_H = 50.0    # flank/back spread when the fire's own rate is unknown
MIN_RATE_M_PER_H = 5.0         # never assume a stalled fire (keeps "contains previous" true)
WIND_ADJUSTMENT = 0.4          # 20-ft wind → midflame wind (typical, partially sheltered fuels)
SWEEP_STEPS = 8                # copies used to sweep the shape downwind


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    flat = [p for g in parts for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
    return max(flat, key=lambda p: p.area) if flat else None


def equivalent_radius(poly: Polygon) -> float:
    """Radius of the circle with the same area (m)."""
    return math.sqrt(poly.area / math.pi)


def observed_spread_rate(older: Polygon, newer: Polygon, dt_h: float) -> float | None:
    """The fire's recent spread rate (m/h): growth of the equivalent radius per hour."""
    if dt_h <= 0:
        return None
    return max(0.0, (equivalent_radius(newer) - equivalent_radius(older)) / dt_h)


def length_to_breadth(wind_20ft_mph: float, wind_adjustment: float = WIND_ADJUSTMENT) -> float:
    """Anderson (1983) fire length-to-breadth ratio from midflame wind (mph). 1 = no wind."""
    u = max(0.0, wind_20ft_mph) * wind_adjustment
    lb = 0.936 * math.exp(0.2566 * u) + 0.461 * math.exp(-0.1548 * u) - 0.397
    return max(1.0, lb)


def mean_wind(speeds, directions_from_deg) -> tuple[float, float]:
    """Mean speed and vector-mean direction (direction the wind blows FROM, degrees)."""
    s = np.asarray(speeds, float)
    d = np.deg2rad(np.asarray(directions_from_deg, float))
    if s.size == 0:
        return 0.0, 0.0
    deg = math.degrees(math.atan2(np.sin(d).mean(), np.cos(d).mean())) % 360
    return float(s.mean()), deg


def grow_perimeter(prev: Polygon, dt_h: float, rate_m_per_h: float,
                   wind_speed_mph: float = 0.0, wind_from_deg: float = 0.0,
                   wind_adjustment: float = WIND_ADJUSTMENT) -> Polygon:
    """Grow `prev` for dt_h hours.

    Every edge moves out by d = rate × dt (backing/flanking spread). The shape is then swept
    downwind by 2·d·(LB − 1), so a point ignition becomes an ellipse-like shape with
    length / breadth = LB. The result always contains `prev`.
    """
    d = max(rate_m_per_h, MIN_RATE_M_PER_H) * max(dt_h, 0.0)
    base = prev.buffer(d) if d > 0 else prev
    lb = length_to_breadth(wind_speed_mph, wind_adjustment)
    sweep = 2.0 * d * (lb - 1.0)
    if sweep <= 1.0:
        return largest_polygon(base)
    toward = math.radians((wind_from_deg + 180.0) % 360.0)   # fire is pushed downwind
    ux, uy = math.sin(toward), math.cos(toward)              # EPSG:5070: x east, y north
    copies = [translate(base, ux * sweep * t, uy * sweep * t) for t in np.linspace(0, 1, SWEEP_STEPS)]
    return largest_polygon(unary_union(copies))


# ---------------------------------------------------------------------------
# From events to a synthetic event
# ---------------------------------------------------------------------------
def read_perimeter(event: ObservationEvent) -> Polygon:
    """The event's perimeter as a polygon in EPSG:5070 (same reading as adapter.load_observation)."""
    import geopandas as gpd
    gdf = gpd.read_file(event.path).to_crs(epsg=5070)
    poly = largest_polygon(gdf.geometry.iloc[int(event.feature_id or 0)])
    if poly is None:
        raise ValueError(f"{event.path} feature {event.feature_id}: no usable polygon")
    return poly


def make_synthetic_event(rejected: ObservationEvent, good_events: list[ObservationEvent],
                         weather: dict | None, out_dir: str | Path) -> tuple[ObservationEvent | None, dict]:
    """Replace `rejected` with a grown copy of the last accepted perimeter.

    Args:
        rejected: the observation the observation agent rejected.
        good_events: accepted observations so far, oldest first (the last one or two are used).
        weather: last cached weather dict (adapter.fetch_weather format: wind_speeds,
                 wind_directions); None = no wind (circular growth).
        out_dir: where to write the synthetic GeoJSON.

    Returns:
        (synthetic event or None if there's no accepted perimeter to grow, details for the log)
    """
    good = [e for e in good_events if e.obs_time <= rejected.obs_time]
    if not good:
        return None, {"reason": "no earlier accepted perimeter to grow"}
    last = good[-1]
    prev = read_perimeter(last)
    dt_h = (rejected.obs_time - last.obs_time).total_seconds() / 3600

    rate, rate_source = None, "default"
    if len(good) >= 2:
        older = good[-2]
        rate = observed_spread_rate(read_perimeter(older), prev,
                                    (last.obs_time - older.obs_time).total_seconds() / 3600)
        rate_source = f"observed {older.obs_time:%m-%d %H:%M} → {last.obs_time:%m-%d %H:%M}"
    if rate is None:
        rate = DEFAULT_RATE_M_PER_H

    speed, direction = (mean_wind(weather.get("wind_speeds", []), weather.get("wind_directions", []))
                        if weather else (0.0, 0.0))
    grown = grow_perimeter(prev, dt_h, rate, speed, direction)

    import geopandas as gpd
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"synthetic_{rejected.obs_time:%Y%m%dT%H%M}.geojson"
    gpd.GeoDataFrame({"source": ["synthetic_perimeter"], "replaces": [rejected.path]},
                     geometry=[grown], crs=5070).to_crs(4326).to_file(path, driver="GeoJSON")

    event = ObservationEvent(obs_time=rejected.obs_time, arrival_time=rejected.arrival_time,
                             source="synthetic_perimeter", path=str(path), feature_id="0",
                             injected=rejected.injected)
    details = {"grown_from": last.model_dump(mode="json"), "dt_h": round(dt_h, 2),
               "rate_m_per_h": round(max(rate, MIN_RATE_M_PER_H), 1), "rate_source": rate_source,
               "wind_mph": round(speed, 1), "wind_from_deg": round(direction),
               "length_to_breadth": round(length_to_breadth(speed), 2),
               "area_km2": round(grown.area / 1e6, 3), "prev_area_km2": round(prev.area / 1e6, 3)}
    return event, details
