# Wildfire Digital Twin — WIFIRE Firemap with EnKF Data Assimilation

This workspace extends the Firemap simulation pipeline with **Ensemble Kalman Filter (EnKF) data assimilation**. Rather than running FARSITE in open-loop (prediction only), this workflow incorporates observed fire perimeters at each timestep to continuously correct the simulation — producing a *analysis* state that is more accurate than either the model prediction or the observation alone.

The example fire used throughout this workspace is the **Border 2 fire**. Synthetic perimeters have been interpolated between real observations to increase temporal resolution, giving the assimilation loop more frequent correction opportunities than sparse satellite observations alone would provide.

The workflow is split across two notebooks that must be run in order: one that prepares all input data, and one that runs the assimilation loop and produces results.

---

## Requirements

**Hardware**
- At least 4 CPUs
- Sufficient memory for ensemble FARSITE+EnKF runs (~1 hour per timestep)

**Dependencies**

Run the following from the workspace root to install all required packages:

```bash
chmod +x install_packages.sh
./install_packages.sh
```

A full list of Python dependencies is available in `requirements.txt`.

---

## Workspace Structure

```
wildfire-digital-twin/
│
├── data_preparation.ipynb          # Step 1: Retrieve and prepare all input data
├── enkf_data_assimilation.ipynb    # Step 2: Run EnKF assimilation loop and visualize results
│
├── data/                           # Input data and configuration
│   ├── workflow_config.json        # Fire parameters shared between notebooks
│   ├── fire_perimeters/            # Observed and synthetic fire perimeter files
│   ├── firms_border_2/             # FIRMS active fire detection data for Border 2
│   ├── landfire_maria_fire/        # LANDFIRE landscape data
│   ├── weather_data/               # Retrieved weather records
│   ├── FIRMS_detections.ipynb      # Data exploration: active fire detections
│   ├── Firemap_perimeters.ipynb    # Data exploration: fire perimeters
│   ├── Firemap_weather.ipynb       # Data exploration: weather data
│   └── LANDFIRE.ipynb              # Data exploration: landscape layers
│
├── outputs/                        # Simulation results
│   ├── border2_bbox.geojson        # Bounding box of the Border 2 fire area
│   ├── border_2_perimeters.geojson # Full set of perimeters used in the simulation
│   ├── border_2_perimeter_maps.png # Visualization of predicted vs. observed perimeters
│   └── syntheticfire_v100/         # Ensemble and analysis outputs from the assimilation run
│
├── src/                            # Source modules used by the notebooks
│   ├── farsite.py                  # FARSITE execution wrapper
│   ├── enkf.py                     # Ensemble Kalman Filter implementation
│   ├── firemap.py                  # Firemap catalog data retrieval (weather, perimeters)
│   ├── geometry.py                 # Coordinate and geometry utilities
│   ├── config.py                   # Shared configuration and default parameters
│   ├── lcpmake                     # Landscape file builder for FARSITE
│   ├── NoBarrier/                  # FARSITE dependency
│   └── TestFARSITE                 # FARSITE executable
│
├── install_packages.sh             # Dependency installation script
└── requirements.txt                # Python package requirements
```

---

## Notebook Execution Order

### 1. `data_preparation.ipynb`

This notebook retrieves and assembles all the inputs the simulation needs before the assimilation loop can run. It queries the WIFIRE platform for each data component and writes them to the `data/` directory, along with a `workflow_config.json` file that records fire parameters so they are shared automatically with the assimilation notebook.

The data components it retrieves are:

- **Active fire detections** — satellite-based observations (VIIRS/MODIS via FIRMS) identifying where fire was detected on the ground, used to establish the initial fire perimeter
- **Observed fire perimeters** — recorded boundary polygons of the fire at points in time, sorted chronologically; includes both real observations and synthetically interpolated perimeters for the Border 2 fire
- **Weather data** — wind speed, wind direction, temperature, and humidity for the fire's location and time window, sourced from real-time sensor networks or NOAA forecast products
- **Landscape data (LANDFIRE)** — static spatial layers describing surface fuel type, canopy characteristics, and topography (elevation, slope, aspect) across the fire area, used to build the FARSITE landscape file (LCP)

Run this notebook once before running `enkf_data_assimilation.ipynb`. It does not need to be re-run unless you are changing the fire or time window.

### 2. `enkf_data_assimilation.ipynb`

This notebook runs the EnKF assimilation loop and produces the final outputs. It loads the configuration and data written by `data_preparation.ipynb` and steps through the observed perimeters sequentially.

At each timestep, the loop executes two steps:

1. **FARSITE forward prediction** — weather is fetched for the interval and FARSITE is run forward from the current fire state. This produces the model's prior estimate of where the fire will be at the next observation time. An ensemble of FARSITE runs is generated by sampling weather uncertainty, and the spread of the ensemble characterizes model uncertainty.

2. **EnKF data assimilation** — the ensemble forecast is combined with the actual observed perimeter at the end of the interval. The EnKF weights these two sources of information against each other based on their respective uncertainties (controlled by the `vsize` and `wsize` parameters), producing an *analysis* state — the best estimate of the true fire boundary given both model and observation.

The analysis state is passed forward as the initial condition for the next timestep, creating a feedback loop where observations continuously correct the simulation throughout its run. Results are checkpointed to disk after each timestep so progress is not lost if the run is interrupted.

> **Note:** each timestep takes approximately 1 hour to run due to the ensemble FARSITE calls.

---

## Outputs

After running `enkf_data_assimilation.ipynb`, results are written to the `outputs/` directory:

| File | Description |
|------|-------------|
| `border_2_perimeters.geojson` | Full set of fire perimeters (real + synthetic) used in the simulation |
| `border2_bbox.geojson` | Bounding box of the Border 2 fire area |
| `border_2_perimeter_maps.png` | Map panels comparing FARSITE prediction, EnKF analysis, and observed perimeters at each timestep |
| `<fire_name>_sequential_analysis.geojson` | EnKF analysis perimeters for all timesteps (WGS84) |
| `<fire_name>_enkf_results.pkl` | Full results dictionary including all geometries, ensemble diagnostics, and weather records |
| `<fire_name>_final_state.npz` | Final EnKF state vector and covariance matrix, enabling the simulation to be resumed without rerunning from the beginning |
| `syntheticfire_v100/` | Ensemble member outputs and intermediate files from the assimilation run |

### Reading the output map

Each panel in the perimeter map corresponds to one simulation timestep and shows four boundaries overlaid on a basemap:

- **Gray** — the open-loop FARSITE prediction from timestep 0, shown as a static baseline reference
- **Blue (dashed)** — the open-loop FARSITE prediction for this timestep (no assimilation applied)
- **Red** — the EnKF analysis for this timestep (assimilation applied)
- **Green (dotted)** — the observed perimeter (ground truth)

The panel title shows the area error for both FARSITE and EnKF, making it straightforward to see where assimilation improved on the uncorrected prediction. A summary table printed below the map reports these figures across all timesteps.

---

## How EnKF Improves on Open-Loop FARSITE

Running FARSITE without data assimilation (as in the Firemap workflow) produces predictions that can drift from reality over time — small errors in weather, fuel classification, or the initial fire boundary compound across timesteps. The EnKF addresses this by treating the fire perimeter as a state to be continuously estimated rather than simply propagated forward.

At each timestep, the filter balances two sources of information:

- **Model uncertainty** (`wsize`) — how much uncertainty to assign to the FARSITE prediction, in meters. Larger values cause the filter to trust observations more.
- **Observation uncertainty** (`vsize`) — how much uncertainty to assign to the observed perimeter, in meters. Larger values give the model more freedom to deviate from observations.

The result is that errors do not accumulate unchecked: each new observation anchors the simulation back toward reality before the next forward pass begins.

---

## Configuring a Different Fire

To run the workflow on a different fire, open `data_preparation.ipynb` and update the fire parameters at the top — fire name, ignition and containment dates, and the coordinate point for weather queries. Then re-run `data_preparation.ipynb` in full before running `enkf_data_assimilation.ipynb`.

The landscape file (LCP) must cover the geographic extent of the new fire. If a pre-built LCP is not available, `src/lcpmake` can be used to generate one from LANDFIRE data for any area of interest.

---

## About

## About

This workspace is part of the NSF-funded [National Data Platfrom](https://nationaldataplatform.org/) project at UC San Diego.
