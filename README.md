# Wildfire Digital Twin — Agentic Orchestration for EnKF Fire Spread Assimilation

This repository implements an **agentic orchestration framework** for wildfire digital twins on the National Data Platform (NDP). The orchestrator adaptively selects simulation parameters at runtime — ensemble size *N*, spatial resolution, and parallelism — based on empirical cost and accuracy models derived from the Border 2 fire experiment. This allows the system to meet a time budget while maximising forecast accuracy, rather than using a fixed configuration.

The underlying simulation pipeline couples **FARSITE fire-spread modelling** with **Ensemble Kalman Filter (EnKF) data assimilation**, continuously correcting the fire state estimate as new perimeter observations arrive. The experiment fire is the **Border 2 fire** (San Diego County, January 2025), using 6 real WFIGS perimeters.

---

## Motivation

Running FARSITE without data assimilation produces predictions that drift from reality over time — small errors in weather inputs or the initial perimeter compound across timesteps. The EnKF addresses this by treating the fire perimeter as a state to be continuously estimated, anchoring the simulation back toward reality at each new observation.

The key challenge for operational deployment is that ensemble FARSITE is expensive: each assimilation cycle runs *N* forward simulations in parallel, and the right choice of *N*, resolution, and core count depends on available compute, the forecast interval, and the current fire geometry. The orchestrator solves this by learning a runtime model from the parameter sweep and using it to pick configuration at each step.

---

## Repository Structure

```
wildfire-digital-twin/
│
├── border2_prepare_data.ipynb   # Step 1: Fetch and validate all inputs
├── border2_experiment.ipynb     # Step 2: EnKF assimilation loop + parameter sweep logging
│
├── data/
│   └── border2_inputs/          # All inputs written by Step 1
│       ├── border_2_perimeters.geojson   # Fire perimeters (live fetch or repo fallback)
│       ├── firms_hotspots.csv            # NASA FIRMS active fire detections
│       ├── weather_obs.csv               # WIFIRE firewx station observations
│       └── landscape.lcp                 # FARSITE landscape file (from LANDFIRE)
│
├── src/
│   ├── farsite.py               # FARSITE wrapper + LANDFIRE download + lcpmake
│   ├── enkf.py                  # EnKF: adjusted_state_EnKF_farsite, run_ensemble, enkf_update
│   ├── firemap.py               # Perimeter fetch (Firemap / WFIGS / NIFC) + weather
│   ├── geometry.py              # Coordinate and geometry utilities
│   ├── config.py                # Shared configuration
│   ├── lcpmake                  # Landscape file builder binary (in src/)
│   ├── NoBarrier/               # FARSITE no-barrier shapefile
│   └── TestFARSITE              # FARSITE executable
│
└── outputs/                     # Simulation results and sweep CSVs
```

---

## Workflow

### Step 0 - Set up `gdal` conda environment
`gdal` is a required dependency for building FARSITE's landscape inputs. Run the following commands in terminal (or at the top of a Jupyter Notebook) and switch to the new conda environment before proceeding to the next steps.


```
chmod +x src/TestFARSITE src/lcpmake install_packages.sh
bash install_packages.sh
```
**Important**: Switch to the Python kernel named `Python (gdal_env)` before running any Jupyter Notebooks.



### Step 1 — `border2_prepare_data.ipynb`

Fetches and validates all inputs. Every step is idempotent — re-running skips outputs that already exist.

| Step | Data | Source | Output |
|------|------|--------|--------|
| 1 | Fire perimeters | Firemap → WFIGS → NIFC → repo fallback | `border_2_perimeters.geojson` |
| 2 | FIRMS hotspots | NASA FIRMS CSV API | `firms_hotspots.csv` |
| 3 | Weather observations | WIFIRE Firemap pylaski | `weather_obs.csv` |
| 4 | Landscape (LCP) | LANDFIRE → `lcpmake` | `landscape.lcp` |

Perimeter fetch tries remote sources in waterfall order and falls back to the repo copy if all fail. The LCP step requires `LANDFIRE_EMAIL` and `LANDFIRE_PRIORITY_CODE` in `user_info.env`; if the file already exists it is skipped.

---

### Step 2 — `border2_experiment.ipynb`

Runs the EnKF assimilation loop over the Border 2 observation window (Jan 23–27 2025) and logs the data needed to build the orchestrator's cost and accuracy models.

**Assimilation loop**

At each timestep:
1. **`run_ensemble(t, obs)`** — runs *N* FARSITE members in parallel, sampling weather uncertainty; returns the forecast ensemble and a list of which members succeeded or failed.
2. **`enkf_update(ensemble, obs)`** — pure EnKF math: combines the ensemble forecast with the observed perimeter, weighted by `vsize` (observation noise) and `wsize` (model noise), and returns the corrected analysis state.

The analysis state feeds forward as the initial condition for the next timestep, so observations continuously anchor the simulation.

**Parameter sweep**

The experiment runs a structured sweep to build the orchestrator's cost model:

| Parameter | Values |
|-----------|--------|
| Ensemble size *N* | 10, 25, 50, 100, 300 (baseline) |
| Resolution (dist_res = perim_res) | 75 m, 150 m, 300 m |
| Cores per member | 1, 4 |
| Parallel workers | 1, 4, 8 |
| Random seeds | 2–3 seeds for N ∈ {10, 25, 50} |

Run the **baseline** (N=300, 150 m, serial) first — it is the anchor row. Then vary one parameter at a time over 3–4 Border 2 timesteps for an apples-to-apples comparison.

**Logging (three CSV files)**

Each run writes three CSV files the orchestrator learns from:

- **`members.csv`** — one row per FARSITE member run: `run_id`, `fire`, `step`, `member`, `N`, `dist_res`, `perim_res`, `ncores`, `n_parallel_workers`, `wall_s`, `cpu_s`, `peak_mem_mb`, `status`, `forecast_interval_min`, `perimeter_vertex_count`, `perimeter_area_m2`, `wind_speed_ms`, `wind_dir_deg`
- **`steps.csv`** — one row per assimilation cycle: `step`, `t_start`, `t_end`, `interval_min`, `N`, `n_valid`, phase timings (`t_input_prep_s`, `t_ensemble_s`, `t_enkf_s`, `t_checkpoint_s`), error metrics (open-loop / prior mean / analysis area error, IoU), drift signals (`innovation_size`, `ensemble_spread`, `obs_is_synthetic`, `vsize`, `wsize`)
- **`prep.csv`** — one row per data-prep action: `action` (landscape build / weather fetch / perimeter fetch), `wall_s`, `size_mb`, `reusable`

---

### Step 3 — Orchestrator (forthcoming)

The orchestrator uses the sweep outputs to build two empirical models:

- **Runtime model**: `wall_s ≈ fixed_cost + N × per_member_time / n_workers` — fit from `members.csv`
- **Error-vs-N curve** at each resolution — from `steps.csv` area errors

At runtime the orchestrator picks the *(N, resolution, workers)* configuration that minimises analysis error subject to a wall-clock budget, replacing the fixed-policy baseline. The `step()` interface (`run_ensemble` → `enkf_update`) is the hook the orchestrator calls; it can invoke forecast-only or forecast+correct depending on observation availability.

---

## Credentials (`user_info.env`)

Create this file in the repo root (gitignored):

```
LANDFIRE_EMAIL=<your registered LANDFIRE email>
LANDFIRE_PRIORITY_CODE=<your LANDFIRE priority code>
FIRMS_MAP_KEY=<your NASA FIRMS API key>
```

- **LANDFIRE** — register at https://lfps.usgs.gov
- **FIRMS** — free key at https://firms.modaps.eosdis.nasa.gov/api/area/

---

## Requirements

**Hardware**: at least 4 CPUs; ~1 hour per timestep at N=300.

```bash
chmod +x install_packages.sh && ./install_packages.sh
```

---

## About

This workspace is part of the NSF-funded [National Data Platform](https://nationaldataplatform.org/) project at UC San Diego, developed in collaboration with the Societal Computing and Innovation (SCIL) Lab.