"""
FARSITE fire simulation module.
Provides wrapper classes and functions for running FARSITE simulations.
"""
import datetime
import os
import uuid
import subprocess
import shutil
import glob
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from config import (
    FARSITE_EXECUTABLE,
    FARSITE_TMP_DIR,
    NO_BARRIER_PATH,
    FARSITE_MIN_IGNITION_VERTEX_DISTANCE,
    FARSITE_SPOT_GRID_RESOLUTION,
    FARSITE_SPOT_PROBABILITY,
    FARSITE_SPOT_IGNITION_DELAY,
    FARSITE_MINIMUM_SPOT_DISTANCE,
    FARSITE_ACCELERATION_ON,
    FARSITE_FILL_BARRIERS,
    SPOTTING_SEED,
    FUEL_MOISTURES_DATA,
    RAWS_ELEVATION,
    RAWS_UNITS,
    DEFAULT_TEMPERATURE,
    DEFAULT_HUMIDITY,
    DEFAULT_PRECIPITATION,
    DEFAULT_CLOUDCOVER,
    FOLIAR_MOISTURE_CONTENT,
    CROWN_FIRE_METHOD,
    WRITE_OUTPUTS_EACH_TIMESTEP,
    MAX_FARSITE_TIMESTEP
)
from geometry import validate_geom


# ============================================================================
# FARSITE CONFIGURATION FILE
# ============================================================================

class Config_File:
    """
    Generates FARSITE configuration (.cfg) files.
    """
    def __init__(self, 
                 FARSITE_START_TIME: datetime.datetime,
                 FARSITE_END_TIME: datetime.datetime,
                 windspeed: int, 
                 winddirection: int,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):
        
        self.__set_default()
        
        # Set the parameters
        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME = FARSITE_END_TIME
        total_minutes = int((self.FARSITE_END_TIME - self.FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP = min(MAX_FARSITE_TIMESTEP, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed = windspeed
        self.winddirection = winddirection

    def __set_default(self):
        """Set default FARSITE configuration parameters."""
        self.version = 1.0
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = FARSITE_MIN_IGNITION_VERTEX_DISTANCE
        self.FARSITE_SPOT_GRID_RESOLUTION = FARSITE_SPOT_GRID_RESOLUTION
        self.FARSITE_SPOT_PROBABILITY = FARSITE_SPOT_PROBABILITY
        self.FARSITE_SPOT_IGNITION_DELAY = FARSITE_SPOT_IGNITION_DELAY
        self.FARSITE_MINIMUM_SPOT_DISTANCE = FARSITE_MINIMUM_SPOT_DISTANCE
        self.FARSITE_ACCELERATION_ON = FARSITE_ACCELERATION_ON
        self.FARSITE_FILL_BARRIERS = FARSITE_FILL_BARRIERS
        self.SPOTTING_SEED = SPOTTING_SEED
        
        self.FUEL_MOISTURES_DATA = FUEL_MOISTURES_DATA
        
        self.RAWS_ELEVATION = RAWS_ELEVATION
        self.RAWS_UNITS = RAWS_UNITS
              
        self.FOLIAR_MOISTURE_CONTENT = FOLIAR_MOISTURE_CONTENT
        self.CROWN_FIRE_METHOD = CROWN_FIRE_METHOD
        
        self.WRITE_OUTPUTS_EACH_TIMESTEP = WRITE_OUTPUTS_EACH_TIMESTEP
        
        self.temperature = DEFAULT_TEMPERATURE
        self.humidity = DEFAULT_HUMIDITY
        self.precipitation = DEFAULT_PRECIPITATION
        self.cloudcover = DEFAULT_CLOUDCOVER
        
    def tostring(self):
        """Generate FARSITE configuration file content as string."""
        config_text = f'FARSITE INPUTS FILE VERSION {self.version}\n'
        
        # Start time
        str_start = f'{self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d}'
        config_text += f'FARSITE_START_TIME: {str_start}\n'

        # End time
        str_end = f'{self.FARSITE_END_TIME.month} {self.FARSITE_END_TIME.day} {self.FARSITE_END_TIME.hour:02d}{self.FARSITE_END_TIME.minute:02d}'
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
        
        # Fuel moistures
        config_text += f'FUEL_MOISTURES_DATA: {len(self.FUEL_MOISTURES_DATA)}\n'
        for data in self.FUEL_MOISTURES_DATA:
            config_text += f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]}\n'
            
        config_text += f'RAWS_ELEVATION: {self.RAWS_ELEVATION}\n'
        config_text += f'RAWS_UNITS: {self.RAWS_UNITS}\n'
        
        # Weather data
        config_text += 'RAWS: 1\n'
        config_text += f'{self.FARSITE_START_TIME.year} {self.FARSITE_START_TIME.month} {self.FARSITE_START_TIME.day} {self.FARSITE_START_TIME.hour:02d}{self.FARSITE_START_TIME.minute:02d} {self.temperature} {self.humidity} {self.precipitation} {self.windspeed} {self.winddirection} {self.cloudcover}\n'
        
        config_text += f'FOLIAR_MOISTURE_CONTENT: {self.FOLIAR_MOISTURE_CONTENT}\n'
        config_text += f'CROWN_FIRE_METHOD: {self.CROWN_FIRE_METHOD}\n'
        config_text += f'WRITE_OUTPUTS_EACH_TIMESTEP: {self.WRITE_OUTPUTS_EACH_TIMESTEP}'
        
        return config_text
    
    def to_file(self, filepath: str):
        """Write configuration to file."""
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE RUN FILE
# ============================================================================

class Run_File:
    """
    Generates FARSITE run files that specify input/output paths.
    """
    def __init__(self, lcppath: str, cfgpath: str, ignitepath: str, 
                 barrierpath: str, outpath: str):
        self.lcppath = lcppath
        self.cfgpath = cfgpath
        self.ignitepath = ignitepath
        self.barrierpath = barrierpath
        self.outpath = outpath

    def tostring(self):
        """Generate run file content as string."""
        return f'{self.lcppath} {self.cfgpath} {self.ignitepath} {self.barrierpath} {self.outpath} -1'
    
    def to_file(self, filepath: str):
        """Write run file to disk."""
        with open(filepath, mode='w') as file:
            file.write(self.tostring())


# ============================================================================
# FARSITE SIMULATION WRAPPER
# ============================================================================

class Farsite:
    """
    Wrapper class for running a single FARSITE simulation.
    """
    def __init__(self, initial: Polygon, params: dict, 
                 start_time: datetime.datetime,
                 lcppath: str = None, barrierpath: str = None,
                 dist_res: int = 30, perim_res: int = 60,
                 debug: bool = False):
        """
        Initialize FARSITE simulation.
        
        Args:
            initial: Initial fire perimeter (Shapely Polygon in EPSG:5070)
            params: Dictionary with keys 'windspeed', 'winddirection', 'dt' (timedelta)
            start_time: Simulation start time
            lcppath: Path to landscape (.lcp) file
            barrierpath: Path to barrier shapefile (None = no barriers)
            dist_res: Distance resolution (meters)
            perim_res: Perimeter resolution (meters)
            debug: If True, keep intermediate files for debugging
        """
        self.farsitepath = str(FARSITE_EXECUTABLE)
        self.id = uuid.uuid4().hex
        
        self.tmpfolder = str(FARSITE_TMP_DIR)
        Path(self.tmpfolder).mkdir(parents=True, exist_ok=True)
        
        self.lcppath = lcppath
        
        # Parse start time
        if isinstance(start_time, datetime.datetime):
            start_dt = start_time
        else:
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        
        end_dt = start_dt + params['dt']
        windspeed = params['windspeed']
        winddirection = params['winddirection']
        
        # Create configuration file
        self.config = Config_File(start_dt, end_dt, windspeed, winddirection, dist_res, perim_res)
        self.configpath = os.path.join(self.tmpfolder, f'{self.id}_config.cfg')
        self.config.to_file(self.configpath)
        
        # Set barrier path
        self.barrierpath = barrierpath if barrierpath else str(NO_BARRIER_PATH)
        
        # Create ignition shapefile
        self.ignitepath = os.path.join(self.tmpfolder, f'{self.id}_ignite.shp')
        ignite_gdf = gpd.GeoDataFrame({'FID': [0], 'geometry': [initial]}, crs="EPSG:5070")
        ignite_gdf.to_file(self.ignitepath)
        
        # Set output path
        self.outpath = os.path.join(self.tmpfolder, f'{self.id}_out')
        
        # Generate run file
        self.runfile = Run_File(self.lcppath, self.configpath, self.ignitepath, 
                                self.barrierpath, self.outpath)
        self.runpath = os.path.join(self.tmpfolder, f'{self.id}_run')
        self.runfile.to_file(self.runpath)
        
        self.debug = debug

    def run(self, timeout=20, ncores=1):
        """
        Run FARSITE simulation.
        
        Args:
            timeout: Maximum runtime in minutes
            ncores: Number of cores to use
            
        Returns:
            Return code from FARSITE
        """
        # Create log directory
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
        
        Returns:
            Shapely geometry or None if no output found
        """
        base = self.outpath
        out_dir = os.path.dirname(base)
        
        # Look for shapefiles with matching prefix
        candidates = []
        candidates += glob.glob(base + "*.shp")
        candidates += glob.glob(os.path.join(out_dir, "*.shp"))
        
        # Deduplicate and sort by modification time
        candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
        
        if not candidates:
            print(f"No shapefile candidates found for outpath prefix={base} (dir={out_dir})")
            return None
        
        # Try newest files first
        for shp in candidates[:25]:
            try:
                gdf = gpd.read_file(shp)
            except Exception:
                continue
            if len(gdf) == 0:
                continue
            
            geom = unary_union(list(gdf.geometry.values))
            return geom
        
        return None


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================

def cleanup_farsite_outputs(run_id, base_dir):
    """
    Delete all files/directories starting with f"{run_id}_" in base_dir.

    Args:
        run_id: Unique run identifier (uuid hex string)
        base_dir: Base directory to clean
    """
    base_dir = Path(base_dir)
    for p in base_dir.glob(f"{run_id}_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


# ============================================================================
# HIGH-LEVEL FARSITE FUNCTIONS
# ============================================================================

def forward_pass_farsite(poly, params, start_time, lcppath, 
                        dist_res=30, perim_res=60, debug=False):
    """
    Run FARSITE forward simulation for specified time period.
    Splits long simulations into multiple MAX_FARSITE_TIMESTEP chunks.
    
    Args:
        poly: Initial fire perimeter (Shapely Polygon)
        params: Dict with 'windspeed', 'winddirection', 'dt' (timedelta)
        start_time: Start time (datetime or string)
        lcppath: Path to landscape file
        dist_res: Distance resolution (meters)
        perim_res: Perimeter resolution (meters)
        debug: Keep intermediate files if True
        
    Returns:
        Final fire perimeter geometry or None
    """
    dt = params['dt']
    MAX_SIM = int(dt.total_seconds() / 60)
    
    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) must be 1-500. Setting to 500')
        dist_res = 500
    
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) must be 1-500. Setting to 500')
        perim_res = 500
    
    run_id = uuid.uuid4().hex   # Single unique ID for this forward pass
    
    # Run multiple FARSITE steps if needed
    number_of_farsites = dt.seconds // (MAX_SIM * 60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed': params['windspeed'],
            'winddirection': params['winddirection'],
            'dt': datetime.timedelta(minutes=MAX_SIM)
        }
        
        farsite = Farsite(
            poly, new_params,
            start_time=start_time,
            lcppath=lcppath,            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug
        )
        farsite.id = run_id   # Share ID so cleanup catches all files
        
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
        print("FARSITE outputs cleaned")
        return poly
    
    new_params = {
        'windspeed': params['windspeed'],
        'winddirection': params['winddirection'],
        'dt': remaining_dt
    }
    
    farsite = Farsite(
        poly, new_params,
        start_time=start_time,
        lcppath=lcppath,        dist_res=dist_res,
        perim_res=perim_res,
        debug=debug
    )
    farsite.id = run_id   # Share ID so cleanup catches all files
    
    farsite.run()
    out = farsite.output_geom()
    
    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None
    
    cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
    print("FARSITE outputs cleaned")
    
    return out


def forward_pass_farsite_24h(poly, params, start_time, lcppath,
                             dist_res=30, perim_res=60, debug=False,
                             max_step_minutes=30, min_final_minutes=1):
    """
    Run FARSITE forward simulation for extended periods (e.g., 24 hours).
    Automatically splits into manageable timesteps.
    
    Args:
        poly: Initial fire perimeter (Shapely Polygon)
        params: Dict with 'windspeed', 'winddirection', 'dt' (timedelta)
        start_time: Start time (datetime or string "YYYY-mm-dd HH:MM:SS")
        lcppath: Path to landscape file
        dist_res: Distance resolution (meters)
        perim_res: Perimeter resolution (meters)
        debug: Keep intermediate files if True
        max_step_minutes: Maximum timestep per FARSITE run (minutes)
        min_final_minutes: Minimum final timestep to run (skip if smaller)
        
    Returns:
        Final fire perimeter geometry or None
    """
    total_dt = params["dt"]
    if not isinstance(total_dt, datetime.timedelta):
        raise TypeError("params['dt'] must be a datetime.timedelta")
    
    # Normalize start_time
    if isinstance(start_time, str):
        start_time = start_time.replace("T", " ")
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    elif not isinstance(start_time, datetime.datetime):
        raise TypeError("start_time must be datetime or string 'YYYY-mm-dd HH:MM:SS'")
    
    if dist_res > 500:
        warnings.warn(f"dist_res ({dist_res}) must be 1-500. Setting to 500")
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f"perim_res ({perim_res}) must be 1-500. Setting to 500")
        perim_res = 500

    run_id    = uuid.uuid4().hex
    
    max_step = datetime.timedelta(minutes=max_step_minutes)
    remaining = total_dt
    step_idx = 0
    
    while remaining > datetime.timedelta(0):
        step_dt = min(max_step, remaining)
        print(f"{remaining} remaining.")
        
        # Skip tiny remainder
        if step_dt < datetime.timedelta(minutes=min_final_minutes):
            break
        
        new_params = {
            "windspeed": params["windspeed"],
            "winddirection": params["winddirection"],
            "dt": step_dt,
        }
        
        farsite = Farsite(
            poly, new_params,
            start_time=start_time,
            lcppath=lcppath,            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug,
        )
        farsite.id = run_id   # Share ID so cleanup catches all files
        farsite.run(ncores=1)
        
        out = farsite.output_geom()
        
        if out is None:
            print("FARSITE output geometry is None. Returning last valid geometry.")
            cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
            return poly
        
        poly = validate_geom(out)
        
        # Advance time
        start_time = start_time + step_dt
        remaining = remaining - step_dt
        step_idx += 1
    
    cleanup_farsite_outputs(run_id, str(FARSITE_TMP_DIR))
    print("FARSITE outputs cleaned")
    
    return poly