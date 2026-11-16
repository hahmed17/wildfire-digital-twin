"""
Standalone FARSITE fire simulation module.
Exact copy of the working farsite.py from the Firemap repo, with config
and geometry imports inlined so no external dependencies are needed.

Directory layout:
    ./
    ├── farsite.py       (this file)
    ├── TestFARSITE
    ├── lcpmake
    ├── landscape.lcp
    ├── NoBarrier/
    │   └── NoBarrier.shp
    └── tmp/             (created automatically)
"""
import datetime
import io
import os
import time
import uuid
import zipfile
import subprocess
import shutil
import glob
import warnings
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely import make_valid


# ============================================================================
# PATHS  (inlined from config.py — all relative to this file)
# ============================================================================
_DIR               = Path(__file__).parent
FARSITE_EXECUTABLE = _DIR / "TestFARSITE"
NO_BARRIER_PATH    = _DIR / "NoBarrier" / "NoBarrier.shp"
FARSITE_TMP_DIR    = _DIR / "tmp"

# ============================================================================
# CONSTANTS  (inlined from config.py — identical values)
# ============================================================================
MAX_FARSITE_TIMESTEP                 = 30

FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
FARSITE_SPOT_GRID_RESOLUTION         = 60.0
FARSITE_SPOT_PROBABILITY             = .5
FARSITE_SPOT_IGNITION_DELAY          = 0
FARSITE_MINIMUM_SPOT_DISTANCE        = 60
FARSITE_ACCELERATION_ON              = 1
FARSITE_FILL_BARRIERS                = 1
SPOTTING_SEED                        = 253114

FUEL_MOISTURES_DATA         = [[0, 7, 4, 7, 30, 60]]
RAWS_ELEVATION              = 2501
RAWS_UNITS                  = 'English'
DEFAULT_TEMPERATURE         = 66
DEFAULT_HUMIDITY            = .08
DEFAULT_PRECIPITATION       = 0
DEFAULT_CLOUDCOVER          = 0
FOLIAR_MOISTURE_CONTENT     = 100
CROWN_FIRE_METHOD           = 'ScottReinhardt'
WRITE_OUTPUTS_EACH_TIMESTEP = 0

DEFAULT_DIST_RES  = 150
DEFAULT_PERIM_RES = 150


# ============================================================================
# GEOMETRY UTILITIES  (inlined from geometry.py)
# ============================================================================

def validate_geom(poly):
    """Validate and clean a geometry, returning the largest polygon."""
    poly = make_valid(poly)
    if isinstance(poly, (GeometryCollection, MultiPolygon)):
        poly = max(poly.geoms, key=lambda g: g.area)
    if isinstance(poly, MultiPolygon): 
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


# ============================================================================
# FARSITE CONFIGURATION FILE
# ============================================================================

class Config_File:
    """Generates FARSITE configuration (.cfg) files."""

    def __init__(self,
                 FARSITE_START_TIME: datetime.datetime,
                 FARSITE_END_TIME: datetime.datetime,
                 windspeed: int,
                 winddirection: int,
                 temperature: int,
                 humidity: float,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):

        self.__set_default()

        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME   = FARSITE_END_TIME
        total_minutes = int((FARSITE_END_TIME - FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP      = min(MAX_FARSITE_TIMESTEP, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES  = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed             = windspeed
        self.winddirection         = winddirection
        self.temperature = temperature
        self.humidity = humidity

    def __set_default(self):
        self.version                              = 1.0
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = FARSITE_MIN_IGNITION_VERTEX_DISTANCE
        self.FARSITE_SPOT_GRID_RESOLUTION         = FARSITE_SPOT_GRID_RESOLUTION
        self.FARSITE_SPOT_PROBABILITY             = FARSITE_SPOT_PROBABILITY
        self.FARSITE_SPOT_IGNITION_DELAY          = FARSITE_SPOT_IGNITION_DELAY
        self.FARSITE_MINIMUM_SPOT_DISTANCE        = FARSITE_MINIMUM_SPOT_DISTANCE
        self.FARSITE_ACCELERATION_ON              = FARSITE_ACCELERATION_ON
        self.FARSITE_FILL_BARRIERS                = FARSITE_FILL_BARRIERS
        self.SPOTTING_SEED                        = SPOTTING_SEED
        self.FUEL_MOISTURES_DATA                  = FUEL_MOISTURES_DATA
        self.RAWS_ELEVATION                       = RAWS_ELEVATION
        self.RAWS_UNITS                           = RAWS_UNITS
        self.FOLIAR_MOISTURE_CONTENT              = FOLIAR_MOISTURE_CONTENT
        self.CROWN_FIRE_METHOD                    = CROWN_FIRE_METHOD
        self.WRITE_OUTPUTS_EACH_TIMESTEP          = WRITE_OUTPUTS_EACH_TIMESTEP
        self.temperature                          = DEFAULT_TEMPERATURE
        self.humidity                             = DEFAULT_HUMIDITY
        self.precipitation                        = DEFAULT_PRECIPITATION
        self.cloudcover                           = DEFAULT_CLOUDCOVER

    def tostring(self):
        config_text  = f'FARSITE INPUTS FILE VERSION {self.version}\n'
        str_start    = f'{self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d}'
        config_text += f'FARSITE_START_TIME: {str_start}\n'
        str_end      = f'{self.FARSITE_END_TIME.month} {self.FARSITE_END_TIME.day} {self.FARSITE_END_TIME.hour:02d}{self.FARSITE_END_TIME.minute:02d}'
        config_text += f'FARSITE_END_TIME: {str_end}\n'
        config_text += f'FARSITE_TIMESTEP: {self.FARSITE_TIMESTEP}\n'
        config_text += f'FARSITE_DISTANCE_RES: {self.FARSITE_DISTANCE_RES}\n'
        config_text += f'FARSITE_PERIMETER_RES: {self.FARSITE_PERIMETER_RES}\n'
        config_text += f'FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE}\n'
        config_text += f'FARSITE_SPOT_GRID_RESOLUTION: {self.FARSITE_SPOT_GRID_RESOLUTION}\n'
        config_text += f'FARSITE_SPOT_PROBABILITY: {self.FARSITE_SPOT_PROBABILITY}\n'
        config_text += f'FARSITE_SPOT_IGNITION_DELAY: {self.FARSITE_SPOT_IGNITION_DELAY}\n'
        config_text += f'FARSITE_MINIMUM_SPOT_DISTANCE: {self.FARSITE_MINIMUM_SPOT_DISTANCE}\n'
        config_text += f'FARSITE_ACCELERATION_ON: {self.FARSITE_ACCELERATION_ON}\n'
        config_text += f'FARSITE_FILL_BARRIERS: {self.FARSITE_FILL_BARRIERS}\n'
        config_text += f'SPOTTING_SEED: {self.SPOTTING_SEED}\n'
        config_text += f'FUEL_MOISTURES_DATA: {len(self.FUEL_MOISTURES_DATA)}\n'
        for data in self.FUEL_MOISTURES_DATA:
            config_text += f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]}\n'
        config_text += f'RAWS_ELEVATION: {self.RAWS_ELEVATION}\n'
        config_text += f'RAWS_UNITS: {self.RAWS_UNITS}\n'
        config_text += 'RAWS: 1\n'
        config_text += (f'{self.FARSITE_START_TIME.year} {self.FARSITE_START_TIME.month} '
                        f'{self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}'
                        f'{self.FARSITE_START_TIME.minute:02d} {self.temperature} '
                        f'{self.humidity} {self.precipitation} {self.windspeed} '
                        f'{self.winddirection} {self.cloudcover}\n')
        config_text += f'FOLIAR_MOISTURE_CONTENT: {self.FOLIAR_MOISTURE_CONTENT}\n'
        config_text += f'CROWN_FIRE_METHOD: {self.CROWN_FIRE_METHOD}\n'
        config_text += f'WRITE_OUTPUTS_EACH_TIMESTEP: {self.WRITE_OUTPUTS_EACH_TIMESTEP}'
        return config_text

    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE RUN FILE
# ============================================================================

class Run_File:
    """Generates FARSITE run files."""

    def __init__(self, lcppath, cfgpath, ignitepath, barrierpath, outpath):
        self.lcppath     = lcppath
        self.cfgpath     = cfgpath
        self.ignitepath  = ignitepath
        self.barrierpath = barrierpath
        self.outpath     = outpath

    def tostring(self):
        return f'{self.lcppath} {self.cfgpath} {self.ignitepath} {self.barrierpath} {self.outpath} -1'

    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE SIMULATION WRAPPER
# ============================================================================

class Farsite:
    """Wrapper for a single FARSITE simulation run."""

    def __init__(self, initial: Polygon, params: dict,
                 start_time: datetime.datetime,
                 lcppath: str = None, barrierpath: str = None,
                 dist_res: int = DEFAULT_DIST_RES, perim_res: int = DEFAULT_PERIM_RES,
                 debug: bool = False):

        self.farsitepath = str(FARSITE_EXECUTABLE)
        self.id          = uuid.uuid4().hex

        self.tmpfolder = str(FARSITE_TMP_DIR)
        Path(self.tmpfolder).mkdir(parents=True, exist_ok=True)

        self.lcppath = lcppath

        # Parse start time
        if isinstance(start_time, datetime.datetime):
            start_dt = start_time
        else:
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        end_dt        = start_dt + params['dt']
        windspeed     = params['windspeed']
        winddirection = params['winddirection']
        temperature =     params['temperature']
        humidity = params['humidity']

        # Config file
        self.config     = Config_File(start_dt, end_dt, windspeed, winddirection, temperature, humidity, dist_res, perim_res)
        self.configpath = os.path.join(self.tmpfolder, f'{self.id}_config.cfg')
        self.config.to_file(self.configpath)

        # Barrier
        self.barrierpath = barrierpath if barrierpath else str(NO_BARRIER_PATH)

        # Ignition shapefile
        self.ignitepath = os.path.join(self.tmpfolder, f'{self.id}_ignite.shp')
        ignite_gdf = gpd.GeoDataFrame({'FID': [0], 'geometry': [initial]}, crs="EPSG:5070")
        ignite_gdf.to_file(self.ignitepath)

        # Output path — FARSITE treats this as a file prefix, writing {outpath}_Perimeters.shp
        self.outpath = os.path.join(self.tmpfolder, f'{self.id}_out')

        # Run file — no extension on runpath, matching the working repo exactly
        self.runfile = Run_File(self.lcppath, self.configpath,
                                self.ignitepath, self.barrierpath, self.outpath)
        self.runpath = os.path.join(self.tmpfolder, f'{self.id}_run')
        self.runfile.to_file(self.runpath)

        self.debug = debug

    def run(self, timeout=20, ncores=4):
        """Execute FARSITE. Returns subprocess return code."""
        log_dir = Path(self.tmpfolder) / "farsite_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        out_log = log_dir / f"{self.id}.out"
        err_log = log_dir / f"{self.id}.err"

        cmd = ["timeout", f"{timeout}m", self.farsitepath, self.runpath, str(ncores)]

        with open(out_log, "w") as fout, open(err_log, "w") as ferr:
            p = subprocess.run(cmd, stdout=fout, stderr=ferr)

        return p.returncode

    def output_geom(self):
        """
        Extract output geometry from FARSITE shapefile results.
        FARSITE writes {outpath}_Perimeters.shp as a prefix in tmp/.
        """
        output_path = self.outpath + '_Perimeters.shp'
        if not os.path.exists(output_path):
            return None

        gdf = gpd.read_file(output_path)
        if len(gdf) == 0:
            return None

        geom = gdf['geometry'][0]
        return Polygon(geom.coords)
        


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup_farsite_outputs(run_id, base_dir):
    """Delete all files/directories starting with run_id in base_dir."""
    base_dir = Path(base_dir)
    for p in base_dir.glob(f"{run_id}_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ============================================================================
# HIGH-LEVEL FORWARD PASS
# ============================================================================

def forward_pass_farsite(poly, params, start_time, lcppath,
                         dist_res=30, perim_res=60, debug=False):
    """
    Run FARSITE forward simulation for specified time period.
    Exact copy of the working implementation from the Firemap repo.

    Args:
        poly:        Initial fire perimeter (Shapely Polygon, EPSG:5070)
        params:      Dict with 'windspeed' (int), 'winddirection' (int), 'dt' (timedelta)
        start_time:  Start time (datetime or "YYYY-MM-DD HH:MM:SS" string)
        lcppath:     Path to .lcp landscape file
        dist_res:    Distance resolution (meters)
        perim_res:   Perimeter resolution (meters)
        debug:       Keep intermediate files if True

    Returns:
        Final fire perimeter as Shapely Polygon, or None on failure
    """
    dt      = params['dt']
    MAX_SIM = int(dt.total_seconds() / 60)

    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500

    run_id = uuid.uuid4().hex

    # Run multiple FARSITE steps if needed
    number_of_farsites = dt.seconds // (MAX_SIM * 60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed':     params['windspeed'],
            'winddirection': params['winddirection'],
            'temperature':     params['temperature'],
            'humidity': params['humidity'],
            'dt':            datetime.timedelta(minutes=MAX_SIM)
        }
        farsite = Farsite(poly, new_params, start_time=start_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print("FARSITE output geometry is None")
            return None
        poly = validate_geom(out)

    # Handle remaining time
    remaining_dt = dt - number_of_farsites * datetime.timedelta(minutes=MAX_SIM)
    if remaining_dt < datetime.timedelta(minutes=10):
        cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
        return poly

    new_params = {
        'windspeed':     params['windspeed'],
        'winddirection': params['winddirection'],
        'temperature':     params['temperature'],
        'humidity': params['humidity'],
        'dt':            remaining_dt
    }
    farsite = Farsite(poly, new_params, start_time=start_time,
                      lcppath=lcppath, dist_res=dist_res,
                      perim_res=perim_res, debug=debug)
    farsite.run()
    out = farsite.output_geom()

    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None

    cleanup_farsite_outputs(farsite.id, str(FARSITE_TMP_DIR))
    return out



def forward_pass_farsite_multistep(poly, params, start_time, lcppath,
                         dist_res=30, perim_res=60, debug=False):
    dt      = params['dt']
    STEP_MINUTES = int(params.get('farsite_dt_minutes', 30))  # step size

    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500

    total_minutes = int(dt.total_seconds() / 60)
    number_of_steps = total_minutes // STEP_MINUTES
    remaining_minutes = total_minutes % STEP_MINUTES

    current_time = (datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    if isinstance(start_time, str)
                    else start_time)

    run_id = uuid.uuid4().hex

    for i in range(number_of_steps):
        step_params = {
            'windspeed':     params['windspeed'],
            'winddirection': params['winddirection'],
            'temperature':   params['temperature'],
            'humidity':      params['humidity'],
            'dt':            datetime.timedelta(minutes=STEP_MINUTES)
        }
        farsite = Farsite(poly, step_params,
                          start_time=current_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print(f"FARSITE output None at step {i+1}/{number_of_steps}")
            return None
        poly = validate_geom(out)
        current_time += datetime.timedelta(minutes=STEP_MINUTES)
        print(f"  Step {i+1}/{number_of_steps}: {poly.area/1e6:.3f} km2")

    # Handle remainder
    if remaining_minutes >= 10:
        step_params = {
            'windspeed':     params['windspeed'],
            'winddirection': params['winddirection'],
            'temperature':   params['temperature'],
            'humidity':      params['humidity'],
            'dt':            datetime.timedelta(minutes=remaining_minutes)
        }
        farsite = Farsite(poly, step_params,
                          start_time=current_time,
                          lcppath=lcppath, dist_res=dist_res,
                          perim_res=perim_res, debug=debug)
        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print("FARSITE output None on remainder step")
            return None
        poly = validate_geom(out)

    cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
    return poly

# ============================================================================
# LANDFIRE DOWNLOAD + LCP GENERATION
# ============================================================================

_LANDFIRE_SUBMIT_URL = 'https://lfps.usgs.gov/api/job/submit'
_LANDFIRE_STATUS_URL = 'https://lfps.usgs.gov/api/job/status'
_LANDFIRE_LAYERS     = 'LF2025_CBD;LF2025_CBH;LF2025_CC;LF2025_CH;LF2025_FBFM40;LF2020_Asp;LF2020_Elev;LF2020_SlpP'
_LANDFIRE_BAND_NAMES = ['250CBD','250CBH','250CC','250CH','250FBFM40',
                         'ASP2020','ELEV2020','SLPP2020']


def download_landfire_data(poly, output_dir, email, priority_code, verbose=True):
    """
    Download LANDFIRE raster data for the bounding box of a polygon.

    Args:
        poly:         Shapely geometry in EPSG:4326
        output_dir:   Directory to save downloaded rasters
        email:        Registered LANDFIRE email (from user_info.env)
        priority_code: LANDFIRE priority code (from user_info.env)
        verbose:      Print progress

    Returns:
        Dict mapping band roles to Path objects for each ASCII raster:
        {'elevation', 'slope', 'aspect', 'fuel', 'canopy_cover',
         'canopy_height', 'canopy_base', 'canopy_density'}
    """
    from osgeo import gdal
    gdal.UseExceptions()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Buffer slightly so the domain is larger than the ignition perimeter
    buffered = poly.buffer(0.5, cap_style='flat', join_style='bevel')
    minx, miny, maxx, maxy = buffered.bounds

    if verbose:
        print('Submitting LANDFIRE request...')
        print(f'  Bounding box: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]')

    params = {
        'Email':               email,
        'Layer_List':          _LANDFIRE_LAYERS,
        'Area_of_Interest':    f'{minx} {miny} {maxx} {maxy}',
        'Output_Projection':   '5070',
        'Resample_Resolution': '90',
        'Priority_Code':       priority_code,
    }
    resp = requests.get(_LANDFIRE_SUBMIT_URL, params=params, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()['jobId']
    if verbose:
        print(f'  Job ID: {job_id}')

    # Poll until complete
    t_start = time.time()
    while True:
        s = requests.get(
            f'{_LANDFIRE_STATUS_URL}?JobId={job_id}', timeout=30
        ).json()
        status = s.get('status', '').lower()
        if verbose:
            print(f'  [{int(time.time()-t_start)}s] {status}')
        if status == 'succeeded':
            download_url = s['outputFile']
            break
        elif status in ('failed', 'canceled'):
            errors = [m['description'] for m in s.get('messages', [])
                      if 'Error' in m.get('type', '')]
            raise RuntimeError(f'LANDFIRE job {status}: {errors}')
        time.sleep(10)

    # Download and extract zip
    if verbose:
        print('  Downloading...')
    zip_resp = requests.get(download_url, stream=True, timeout=120)
    zip_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        zf.extractall(output_dir)

    multi_tif = next(output_dir.glob('*.tif'))

    # Extract each band to its own ASCII raster
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for band_idx, name in enumerate(_LANDFIRE_BAND_NAMES, start=1):
            asc_path = str(output_dir / f'{name}.asc')
            gdal.Translate(
                asc_path, str(multi_tif),
                format='AAIGrid', bandList=[band_idx],
            )
            if verbose:
                print(f'  ✓ {name}.asc')

    return {
        'elevation':      output_dir / 'ELEV2020.asc',
        'slope':          output_dir / 'SLPP2020.asc',
        'aspect':         output_dir / 'ASP2020.asc',
        'fuel':           output_dir / '250FBFM40.asc',
        'canopy_cover':   output_dir / '250CC.asc',
        'canopy_height':  output_dir / '250CH.asc',
        'canopy_base':    output_dir / '250CBH.asc',
        'canopy_density': output_dir / '250CBD.asc',
    }


def generate_lcp_from_rasters(output_path, rasters, latitude, verbose=True):
    """
    Run lcpmake to build a FARSITE landscape (.lcp) file from ASCII rasters.

    Args:
        output_path: Desired path for the output .lcp file
        rasters:     Dict returned by download_landfire_data()
        latitude:    Latitude of the fire area (degrees)
        verbose:     Print progress

    Returns:
        Path to the generated .lcp file
    """
    lcpmake = Path(os.path.abspath('lcpmake'))
    if not lcpmake.exists():
        # Also check next to this file (src/)
        lcpmake = _DIR / 'lcpmake'
    if not lcpmake.exists():
        raise FileNotFoundError(
            f'lcpmake not found. Expected at {lcpmake} or ./lcpmake'
        )

    output_path = Path(output_path)
    cmd = [
        str(lcpmake),
        '-latitude',  str(int(latitude)),
        '-landscape', str(output_path.with_suffix('')),
        '-elevation', str(rasters['elevation']),
        '-slope',     str(rasters['slope']),
        '-aspect',    str(rasters['aspect']),
        '-fuel',      str(rasters['fuel']),
        '-cover',     str(rasters['canopy_cover']),
        '-height',    str(rasters['canopy_height']),
        '-base',      str(rasters['canopy_base']),
        '-density',   str(rasters['canopy_density']),
        '-fb40',
    ]
    if verbose:
        print('Running lcpmake...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'lcpmake failed:\n{result.stderr}')

    lcp = output_path.with_suffix('.lcp')
    if not lcp.exists():
        raise RuntimeError('lcpmake ran but .lcp was not created')
    if verbose:
        print(f'  ✓ {lcp}  ({lcp.stat().st_size / 1024:.1f} KB)')
    return lcp