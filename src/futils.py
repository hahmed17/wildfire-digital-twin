import datetime
import os
import pathlib
import uuid
import requests
import time
import zipfile
import rasterio
import io
from pathlib import Path 
import shutil
import glob
import re

from datetime import timedelta
from shapely import Polygon, make_valid, GeometryCollection, MultiPolygon, box
from pyproj import Transformer
from osgeo import gdal
from putils import calculate_max_area_geom, validate_geom

import geopandas as gpd

import warnings



class Config_File:
    def __init__(self, 
                 FARSITE_START_TIME: datetime.datetime,
                 FARSITE_END_TIME: datetime.datetime,
                 windspeed: int, winddirection: int,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):
        
        self.__set_default()
        
        # Set the parameters
        MAX_STEP_MINUTES = 30
        
        self.FARSITE_START_TIME = FARSITE_START_TIME
        self.FARSITE_END_TIME = FARSITE_END_TIME
        total_minutes = int((self.FARSITE_END_TIME - self.FARSITE_START_TIME).total_seconds() / 60)
        self.FARSITE_TIMESTEP = min(MAX_STEP_MINUTES, max(1, total_minutes))
        self.FARSITE_DISTANCE_RES = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed = windspeed
        self.winddirection = winddirection
        self.configpath = "WIFIRE-Digital-Twinners/dt-refactored"

    def __set_default(self):
        self.version = 1.0
        self.FARSITE_DISTANCE_RES = 60
        self.FARSITE_PERIMETER_RES = 120
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
        self.FARSITE_SPOT_GRID_RESOLUTION = 60.0
        self.FARSITE_SPOT_PROBABILITY = 0  # 0.9
        self.FARSITE_SPOT_IGNITION_DELAY = 0
        self.FARSITE_MINIMUM_SPOT_DISTANCE = 60
        self.FARSITE_ACCELERATION_ON = 1
        self.FARSITE_FILL_BARRIERS = 1
        self.SPOTTING_SEED = 253114
        
        self.FUEL_MOISTURES_DATA = [[0, 3, 4, 6, 30, 60]]
        
        self.RAWS_ELEVATION = 2501
        self.RAWS_UNITS = 'English'
        # Add self.raws from the init
              
        self.FOLIAR_MOISTURE_CONTENT = 100
        self.CROWN_FIRE_METHOD = 'ScottReinhardt'
        
        self.WRITE_OUTPUTS_EACH_TIMESTEP = 0
        
        self.temperature = 66
        self.humidity = 8
        self.precipitation = 0
        self.cloudcover = 0
        
    def tostring(self):
        config_text = 'FARSITE INPUTS FILE VERSION {}\n'.format(self.version)
        
        
        str_start = '{month} {day} {time}'.format(
                            month = self.FARSITE_START_TIME.month,
                            day = self.FARSITE_START_TIME.day,
                            time = '{:02d}{:02d}'.format(
                                    self.FARSITE_START_TIME.hour,
                                    self.FARSITE_START_TIME.minute))
        config_text += 'FARSITE_START_TIME: {}\n'.format(str_start)

        str_end = '{month} {day} {time}'.format(
                            month = self.FARSITE_END_TIME.month,
                            day = self.FARSITE_END_TIME.day,
                            time = '{:02d}{:02d}'.format(
                                    self.FARSITE_END_TIME.hour,
                                    self.FARSITE_END_TIME.minute))
        config_text += 'FARSITE_END_TIME: {}\n'.format(str_end)
        
        config_text += 'FARSITE_TIMESTEP: {}\n'.format(self.FARSITE_TIMESTEP)
        config_text += 'FARSITE_DISTANCE_RES: {}\n'.format(self.FARSITE_DISTANCE_RES)
        config_text += 'FARSITE_PERIMETER_RES: {}\n'.format(self.FARSITE_PERIMETER_RES)
        config_text += 'FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {}\n'.format(self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE)
        config_text += 'FARSITE_SPOT_GRID_RESOLUTION: {}\n'.format(self.FARSITE_SPOT_GRID_RESOLUTION)
        config_text += 'FARSITE_SPOT_PROBABILITY: {}\n'.format(self.FARSITE_SPOT_PROBABILITY)
        config_text += 'FARSITE_SPOT_IGNITION_DELAY: {}\n'.format(self.FARSITE_SPOT_IGNITION_DELAY)              
        config_text += 'FARSITE_MINIMUM_SPOT_DISTANCE: {}\n'.format(self.FARSITE_MINIMUM_SPOT_DISTANCE)
        config_text += 'FARSITE_ACCELERATION_ON: {}\n'.format(self.FARSITE_ACCELERATION_ON)
        config_text += 'FARSITE_FILL_BARRIERS: {}\n'.format(self.FARSITE_FILL_BARRIERS)
        config_text += 'SPOTTING_SEED: {}\n'.format(self.SPOTTING_SEED)
        
        # Fuel moistures
        config_text += 'FUEL_MOISTURES_DATA: {}\n'.format(len(self.FUEL_MOISTURES_DATA))
        for data in self.FUEL_MOISTURES_DATA:
            config_text += '{} {} {} {} {} {}\n'.format(data[0], data[1], data[2],
                                                      data[3], data[4], data[5])
            
        config_text += 'RAWS_ELEVATION: {}\n'.format(self.RAWS_ELEVATION)
        config_text += 'RAWS_UNITS: {}\n'.format(self.RAWS_UNITS)
        
        # Weather data (currently only a single weather data)
        config_text += 'RAWS: 1\n'
        config_text += '{year} {month} {day} {time} {temperature} {humidity} {precipitation} {windspeed} {winddirection} {cloudcover}\n'.format(
                                year = self.FARSITE_START_TIME.year,
                                month = self.FARSITE_START_TIME.month,
                                day = self.FARSITE_START_TIME.day,
                                time = '{:02d}{:02d}'.format(
                                    self.FARSITE_START_TIME.hour, 
                                    self.FARSITE_START_TIME.minute),
                                temperature = self.temperature,
                                humidity = self.humidity,
                                precipitation = self.precipitation,
                                windspeed = self.windspeed,
                                winddirection = self.winddirection,
                                cloudcover = self.cloudcover
                            )
        config_text += 'FOLIAR_MOISTURE_CONTENT: {}\n'.format(self.FOLIAR_MOISTURE_CONTENT)
        config_text += 'CROWN_FIRE_METHOD: {}\n'.format(self.CROWN_FIRE_METHOD)
        config_text += 'WRITE_OUTPUTS_EACH_TIMESTEP: {}'.format(self.WRITE_OUTPUTS_EACH_TIMESTEP)
        
        return config_text
    
    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())
            
            
class Run_File:
    def __init__(self, lcppath: str, cfgpath: str, ignitepath: str, barrierpath: str, outpath: str):
        self.lcppath = lcppath
        self.cfgpath = cfgpath
        self.ignitepath = ignitepath
        self.barrierpath = barrierpath
        self.outpath = outpath

        # print(f"lcppath: {lcppath}")
        # print(f"cfgpath: {cfgpath}")
        # print(f"ignitepath: {ignitepath}")
        # print(f"barrierpath: {barrierpath}")
        # print(f"outpath: {outpath}")
        

    def tostring(self):
        return '{lcpath} {cfgpath} {ignitepath} {barrierpath} {outpath} -1'.format(
                                lcpath =  self.lcppath, 
                                cfgpath = self.cfgpath, 
                                ignitepath = self.ignitepath, 
                                barrierpath = self.barrierpath, 
                                outpath = self.outpath)
    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())

 
class Farsite:
    def __init__(self, initial: Polygon, params: dict, irwinid: str, start_time: datetime.datetime,
                 description: str = None,
                 lcppath: str = None, barrierpath: str = None,
                 dist_res:int = 30, perim_res: int = 60,
                 debug:bool = False):
        
        self.farsitepath = "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored/src/TestFARSITE" # in hotshot.sdsc.edu        
        self.id = uuid.uuid4().hex

        self.tmpfolder = "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored"                           
        pathlib.Path(self.tmpfolder).mkdir(parents=True, exist_ok=True) 

        self.lcppath = lcppath

        
        # start_dt = datetime.datetime(year=2019, month=1, day=1, hour=10, minute=0)
        if isinstance(start_time, datetime.datetime):
            start_dt = start_time
        else:
            start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        end_dt = start_dt + params['dt']
        windspeed = params['windspeed']
        winddirection = params['winddirection']
        
        #### RUN FILE PREPARATION ####
        # Config file
        self.config = Config_File(start_dt, end_dt, windspeed, winddirection, dist_res, perim_res)
        self.configpath = os.path.join(self.tmpfolder, f'{irwinid}_config_{self.id}.cfg')
        self.config.to_file(self.configpath)
               
        # Barrier file - NoBarrier is just zeroes
        self.barrierpath = barrierpath
        if self.barrierpath == None: # Create No Barrier
            self.barrierpath = "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored/NoBarrier/NoBarrier.shp"
            
        
        # Ignite path
        self.ignitepath = os.path.join(self.tmpfolder, f'{irwinid}_ignite_{self.id}.shp')
        ignite_gdf = gpd.GeoDataFrame({'FID': [0], 'geometry': [initial]}, crs="EPSG:5070")
        ignite_gdf.to_file(self.ignitepath, driver="ESRI Shapefile")  
       

        PRJ_5070 = 'PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101],TOWGS84[1,1,-1,0,0,0,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5070"]]'

        # Create projection file
        prj_filename = self.ignitepath.replace(".shp", ".prj")
        with open(prj_filename, "w") as f:
            f.write(PRJ_5070)
        
        
        # Output path
        self.outpath = os.path.join(self.tmpfolder, f'{irwinid}_out_{self.id}')
        # Path(self.outpath).mkdir(parents=True, exist_ok=True)
                                    
        
        # Generate RunFile
        self.runfile = Run_File(self.lcppath, self.configpath, self.ignitepath, self.barrierpath, self.outpath)
        self.runpath = os.path.join(self.tmpfolder, f'{irwinid}_run_{self.id}')
        self.runfile.to_file(self.runpath)
        
        # Debugging keeps the files
        self.debug = debug

    def run(self, timeout=5, ncores=4):
        self.command = f'timeout {timeout}m {self.farsitepath} {self.runpath} {ncores} > output.out 2> output.err'  # donot run 
        os.system(self.command)
        
    # def output_geom(self):
    #     # Look in both: output directory and "next to" output prefix
    #     candidates = []
    #     candidates += glob.glob(os.path.join(self.outpath, "*Perim*.shp"))
    #     candidates += glob.glob(os.path.join(self.outpath, "*Perimeter*.shp"))
    #     candidates += glob.glob(os.path.join(self.outpath, "*.shp"))
    
    #     candidates += glob.glob(self.outpath + "*Perim*.shp")
    #     candidates += glob.glob(self.outpath + "*Perimeter*.shp")
    #     candidates += glob.glob(self.outpath + "*.shp")
    
    #     print(f"Out perimeter candidates: {candidates[:5]}{'...' if len(candidates) > 5 else ''}")
    
    #     if not candidates:
    #         return None
    
    #     shp = max(candidates, key=os.path.getmtime)
    #     print("Reading perimeter from:", shp)
    
    #     gdf = gpd.read_file(shp)
    #     if len(gdf) == 0:
    #         return None
    
    #     geom = gdf.geometry.iloc[-1]
    #     # Don't rebuild Polygon from coords (drops holes, breaks MultiPolygon)
    #     return geom


    def output_geom(self):
        output_path = self.outpath + '_Perimeters.shp'
        if not os.path.exists(output_path):
            print(f"{output_path} does not exist.")
            return None
        
        gdf = gpd.read_file(output_path)
        if len(gdf) == 0:
            return None
        
        geom = gdf['geometry'][0]
        return Polygon(geom.coords)



def cleanup_farsite_outputs(irwinid, base_dir):
    """
    Deletes all files/directories starting with f"{irwinid}_"
    in base_dir.
    """
    base_dir = Path(base_dir)

    for p in base_dir.glob(f"{irwinid}_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

def create_projection_file(filename, crs):
    return
    

def forward_pass_farsite(poly, params, start_time, lcppath, irwinid, dist_res=30, perim_res=60, debug=False):
    dt = params['dt']
    MAX_SIM = int(dt.total_seconds() / 60)

    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) has to be 1-->500. Setting to 500')
        dist_res=500

    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) has to be 1-->500. Setting to 500')
        perim_res=500

    number_of_farsites = dt.seconds//(MAX_SIM*60)
    for i in range(number_of_farsites):
        new_params = {
            'windspeed': params['windspeed'],
            'winddirection': params['winddirection'],
            'dt': datetime.timedelta(minutes=MAX_SIM)
        }

        farsite = Farsite(
            poly,
            new_params,
            start_time=start_time,
            lcppath=lcppath,
            irwinid=irwinid,
            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug
        )

        farsite.run()
        out = farsite.output_geom()
        if out is None:
            print("farsite output geometry is None")
            return None

        poly = validate_geom(out)

    remaining_dt = dt - number_of_farsites*datetime.timedelta(minutes=MAX_SIM)
    if remaining_dt < datetime.timedelta(minutes=10):
        return poly

    new_params = {
        'windspeed': params['windspeed'],
        'winddirection': params['winddirection'],
        'dt': remaining_dt
    }

    farsite = Farsite(
        poly,
        new_params,
        start_time=start_time,
        lcppath=lcppath,
        irwinid=irwinid,
        dist_res=dist_res,
        perim_res=perim_res,
        debug=debug
    )
    farsite.run()

    out = farsite.output_geom()
    
    if out is None:
        print("No output perimeter produced; keeping outputs for inspection.")
        return None

    cleanup_farsite_outputs(irwinid, "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored")
    print("FARSITE outputs cleaned")

    return out


def forward_pass_farsite_24h(
    poly,
    params,
    start_time,
    lcppath,
    irwinid,
    dist_res=30,
    perim_res=60,
    debug=False,
    max_step_minutes=30,   # <-- FARSITE per-run cap
    min_final_minutes=1,   # <-- run the last small remainder (set to 10 if you want your old behavior)
):
    total_dt = params["dt"]
    if not isinstance(total_dt, datetime.timedelta):
        raise TypeError("params['dt'] must be a datetime.timedelta")

    # --- normalize start_time ---
    if isinstance(start_time, str):
        # supports "YYYY-mm-dd HH:MM:SS" and "YYYY-mm-ddTHH:MM:SS"
        start_time = start_time.replace("T", " ")
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    elif not isinstance(start_time, datetime.datetime):
        raise TypeError("start_time must be a datetime.datetime or a str like 'YYYY-mm-ddTHH:MM:SS'")
        

    if dist_res > 500:
        warnings.warn(f"dist_res ({dist_res}) has to be 1-->500. Setting to 500")
        dist_res = 500
    if perim_res > 500:
        warnings.warn(f"perim_res ({perim_res}) has to be 1-->500. Setting to 500")
        perim_res = 500

    max_step = datetime.timedelta(minutes=max_step_minutes)

    # Use total_seconds so days are handled correctly
    remaining = total_dt
    step_idx = 0

    while remaining > datetime.timedelta(0):
        step_dt = min(max_step, remaining)
        print(f"{remaining} steps remaining.")

        # optionally skip tiny remainder
        if step_dt < datetime.timedelta(minutes=min_final_minutes):
            break

        new_params = {
            "windspeed": params["windspeed"],         # scalar
            "winddirection": params["winddirection"], # scalar
            "dt": step_dt,
        }

        farsite = Farsite(
            poly,
            new_params,
            start_time=start_time,
            lcppath=lcppath,
            irwinid=irwinid,
            dist_res=dist_res,
            perim_res=perim_res,
            debug=debug,
        )
        farsite.run()

        out = farsite.output_geom()
        if out is None:
            print("farsite output geometry is None. Returning last valid geometry.")
            cleanup_farsite_outputs(irwinid, "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored")
            return poly
            
        poly = validate_geom(out)

        # advance time and remaining horizon
        start_time = start_time + step_dt
        remaining = remaining - step_dt
        step_idx += 1

    cleanup_farsite_outputs(irwinid, "/home/jovyan/work/WIFIRE-Digital-Twinners/dt-refactored")
        
    print("FARSITE outputs cleaned")

    return poly
