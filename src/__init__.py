"""
Fire modeling workflow package.
Provides tools for wildfire simulation and data assimilation.
"""

import config
import geometry
import farsite
import enkf
import firemap

__version__ = "0.1.0"

from geometry import (
    geom_to_state,
    state_to_geom,
    validate_geom,
    align_states,
    plot_geometry,
)

from farsite import (
    forward_pass_farsite,
    forward_pass_farsite_24h,
    cleanup_farsite_outputs,
)

from enkf import (
    adjusted_state_EnKF_farsite,
    sample_windspeed,
    sample_winddirection,
)

from firemap import (
    fetch_fire_perimeters,
    fetch_weather,
)

__all__ = [
    'config', 'geometry', 'farsite', 'enkf', 'firemap',
    'geom_to_state', 'state_to_geom', 'validate_geom',
    'align_states', 'plot_geometry',
    'forward_pass_farsite', 'forward_pass_farsite_24h', 'cleanup_farsite_outputs',
    'adjusted_state_EnKF_farsite', 'sample_windspeed', 'sample_winddirection',
    'fetch_fire_perimeters', 'fetch_weather',
]
