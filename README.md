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
│   ├── enkf.py                  # EnKF core: run_ensemble, enkf_update, step()
│   ├── firemap.py               # Perimeter fetch (Firemap / WFIGS / NIFC) + weather
│   ├── geometry.py              # Coordinate and geometry utilities
│   ├── config.py                # Shared configuration
│   ├── lcpmake                  # Landscape file builder binary (in src/)
│   ├── NoBarrier/               # FARSITE no-barrier shapefile
│   └── TestFARSITE              # FARSITE executable
│
└── outputs/                     # Simulation results and sweep CSVs
    └── Border_2_Real/
        ├── members.csv                      # Per-FARSITE-member timing + status
        ├── steps.csv                        # Per-assimilation-cycle summary
        ├── prep.csv                         # Data-prep actions
        └── Border_2_Real_enkf_results.pkl   # Full results checkpoint (per step)
```
 
---
 
## Workflow
 
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
 
**EnKF module architecture (`src/enkf.py`)**
 
The EnKF is split into three composable functions:
 
| Function | Role |
|----------|------|
| `run_ensemble(...)` | Runs *N* FARSITE members in parallel via `ProcessPoolExecutor`; returns forecast ensemble dict |
| `enkf_update(ensemble, obs, ...)` | Pure EnKF math — no FARSITE calls; returns analysis state + updated covariance |
| `step(t, obs, ...)` | Orchestrator interface: `run_ensemble` → `enkf_update` → CSV logging |
 
The notebook calls `step()` at each assimilation cycle. The analysis state and updated covariance `X` feed forward to the next step — `X` is passed into `run_ensemble` so prior perturbations (Cholesky decomposition) tighten as the filter converges.
 
**Setting the number of parallel workers**
 
In the **Experiment Parameters** cell of `border2_experiment.ipynb`:
 
```python
N_WORKERS = 12   # number of FARSITE members to run simultaneously
```
 
Each worker runs one FARSITE simulation in a separate process. Set this to the number of CPU cores available. Each FARSITE process is single-threaded, so `N_WORKERS = n_cores` gives near-linear speedup up to memory limits.
 
| Hardware | Recommended `N_WORKERS` |
|----------|------------------------|
| 4 cores / 16 GB | 4 |
| 6 cores / 32 GB | 6 |
| 8 cores / 32 GB | 6–8 |
| 16 cores / 64 GB | 12–16 |
 
**Assimilation loop**
 
At each timestep the notebook:
1. Queries weather for the forecast interval (Firemap / fallback)
2. Samples wind speed and direction for each ensemble member
3. Calls `enkf_step(t, ...)` — dispatches *N* FARSITE runs across `N_WORKERS` processes, then runs the EnKF update
4. Checkpoints `results` to `Border_2_Real_enkf_results.pkl` after each step
**Parameter sweep**
 
After the baseline run, vary one parameter at a time to build the orchestrator's cost model:
 
| Parameter | Values |
|-----------|--------|
| Ensemble size *N* | 10, 25, 50, 100, 300 (baseline) |
| Resolution (`dist_res` = `perim_res`) | 75 m, 150 m, 300 m |
| Parallel workers (`N_WORKERS`) | 1, 4, 8 |
| Random seeds | 2–3 seeds for N ∈ {10, 25, 50} |
 
Run the **baseline** (N=300, 150 m, N_WORKERS=12) first — it is the anchor row. Change only the parameter under test; keep everything else fixed.
 
**Logging (three CSV files)**
 
Each run appends to three CSV files the orchestrator learns from:
 
- **`members.csv`** — one row per FARSITE member: `run_id`, `fire`, `step`, `member`, `N`, `dist_res`, `perim_res`, `ncores`, `n_parallel_workers`, `wall_s`, `cpu_s`, `status`, `wind_speed_ms`, `wind_dir_deg`
- **`steps.csv`** — one row per assimilation cycle: `run_id`, `step`, `t_obs`, `N`, `n_valid`, `t_ensemble_s`, `t_enkf_s`, `vsize`, `wsize`
- **`prep.csv`** — one row per data-prep action: `action`, `step`, `detail`, `elapsed_s`, `status`
**Cleaning up between runs**
 
Before starting a fresh run (e.g. a new sweep configuration), use the cleanup cell at the top of the notebook to remove FARSITE temp files and stale CSVs from the previous run.
 
---
 
### Step 3 — Orchestrator (forthcoming)
 
The orchestrator uses the sweep outputs to build two empirical models:
 
- **Runtime model**: `wall_s ≈ fixed_cost + N × per_member_time / n_workers` — fit from `members.csv`
- **Error-vs-N curve** at each resolution — from `steps.csv` area errors
At runtime the orchestrator picks the *(N, resolution, workers)* configuration that minimises analysis error subject to a wall-clock budget, replacing the fixed-policy baseline. The `step()` interface (`run_ensemble` → `enkf_update`) is the hook the orchestrator calls; it can invoke forecast-only or forecast+correct depending on observation availability.
 
---
 
## Credentials (`user_info.env`)
 
Create this file in the repo root (gitignored) or copy `user_info_template.env`:
 
```
LANDFIRE_EMAIL=<your registered LANDFIRE email>
LANDFIRE_PRIORITY_CODE=<your LANDFIRE priority code>
FIRMS_MAP_KEY=<your NASA FIRMS API key>
```
 
- **LANDFIRE** — register at https://lfps.usgs.gov
- **FIRMS** — free key at https://firms.modaps.eosdis.nasa.gov/api/area/
---
 
## Requirements
 
**Hardware**: at least 4 CPUs and 16 GB RAM recommended; ~20-25 min per timestep at N=300 with 12 workers (vs ~4 hours serial).
 
```bash
chmod +x install_packages.sh && ./install_packages.sh
```
 
---

## About

This workspace is part of the NSF-funded [National Data Platform](https://nationaldataplatform.org/) project at UC San Diego, developed in collaboration with the Societal Computing and Innovation Lab (SCIL).