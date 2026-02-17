"""
Configuration file for fire modeling workflow.
Contains shared constants, default parameters, and file paths.

Paths are resolved relative to this file so the repo is fully portable —
clone anywhere and it just works.
"""
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS  (all relative to repo root — no hard-coding)
# ============================================================================
SRC_DIR  = Path(__file__).parent          # .../repo/src/
BASE_DIR = SRC_DIR.parent                 # .../repo/
DATA_DIR = BASE_DIR / "data"              # .../repo/data/
OUTPUT_DIR = BASE_DIR / "outputs"         # .../repo/outputs/
TMP_DIR  = BASE_DIR / "tmp"               # .../repo/tmp/

# Executables and static assets live in data/
FARSITE_EXECUTABLE = DATA_DIR / "TestFARSITE"
LCPMAKE_EXECUTABLE = DATA_DIR / "lcpmake"
NO_BARRIER_PATH    = DATA_DIR / "NoBarrier" / "NoBarrier.shp"
LCP_PATH           = DATA_DIR / "landscape.lcp"

# Temporary working directory for FARSITE runs
FARSITE_TMP_DIR = TMP_DIR


# ============================================================================
# FARSITE PARAMETERS
# ============================================================================
DEFAULT_DIST_RES  = 150   # Distance resolution (meters)
DEFAULT_PERIM_RES = 150   # Perimeter resolution (meters)

DEFAULT_FARSITE_TIMESTEP = 30   # minutes
MAX_FARSITE_TIMESTEP     = 30   # Maximum FARSITE timestep (minutes)

FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
FARSITE_SPOT_GRID_RESOLUTION         = 60.0
FARSITE_SPOT_PROBABILITY             = 0      # Set to 0.9 to enable spotting
FARSITE_SPOT_IGNITION_DELAY          = 0
FARSITE_MINIMUM_SPOT_DISTANCE        = 60
FARSITE_ACCELERATION_ON              = 1
FARSITE_FILL_BARRIERS                = 1
SPOTTING_SEED                        = 253114

# Weather defaults
DEFAULT_TEMPERATURE  = 66   # Fahrenheit
DEFAULT_HUMIDITY     = 8    # percent
DEFAULT_PRECIPITATION = 0   # inches
DEFAULT_CLOUDCOVER   = 0    # percent

# Fuel moisture defaults
FUEL_MOISTURES_DATA = [[0, 3, 4, 6, 30, 60]]

# Crown fire settings
FOLIAR_MOISTURE_CONTENT = 100
CROWN_FIRE_METHOD       = 'ScottReinhardt'

# RAWS defaults
RAWS_ELEVATION = 2501   # feet
RAWS_UNITS     = 'English'

# Output settings
WRITE_OUTPUTS_EACH_TIMESTEP = 0


# ============================================================================
# ENKF PARAMETERS
# ============================================================================
MAX_ENSEMBLE_SIZE       = 300
ENSEMBLE_SIZE_MULTIPLIER = 5

DEFAULT_OBSERVATION_NOISE = 500    # vsize (meters)
DEFAULT_MODEL_NOISE       = 200    # wsize (meters)
DEFAULT_COVARIANCE_SCALE  = 1000.0
DEFAULT_VERTEX_COUNT      = None   # None = auto-detect


# ============================================================================
# COORDINATE REFERENCE SYSTEMS
# ============================================================================
FARSITE_CRS = "EPSG:5070"
WGS84_CRS   = "EPSG:4326"


# ============================================================================
# DATA API ENDPOINTS
# ============================================================================
FIREMAP_WFS_URL = "https://firemap.sdsc.edu/geoserver/wfs"
FIREMAP_WX_URL  = "https://firemap.sdsc.edu/pylaski/stations/data"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def get_ensemble_size(n_vertices):
    """Calculate ensemble size based on number of vertices."""
    return min(MAX_ENSEMBLE_SIZE, ENSEMBLE_SIZE_MULTIPLIER * n_vertices)


# Ensure directories exist on import
ensure_directories()
