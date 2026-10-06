# Modular High-Altitude Balloon Mission SDK

This repository contains a modular software development kit (SDK) for simulating high-altitude balloon and near-space flight trajectories. It supports multiple physical models, environmental data sources, and analysis workflows, with an emphasis on extensibility and verification rather than a single fixed prediction pipeline.

The core solver implements a three-degree-of-freedom (3-DOF) inertial flight model for high-altitude balloons, including buoyant ascent, burst detection, and parachute-controlled descent. Atmospheric properties are computed using the 1976 U.S. Standard Atmosphere. Horizontal wind forcing, when enabled, is obtained from U.S. National Oceanic and Atmospheric Administration (NOAA) Global Forecast System (GFS) and/or High Resolution Rapid Refresh (HRRR) forecast products using cadence-aware spatial and temporal interpolation. Time integration is performed using a classical fourth-order Runge–Kutta (RK4) scheme formulated for second-order systems.

Multiple model fidelities are supported within a shared numerical framework. A wind-free (altitude-only) configuration is retained for baseline verification and rapid design-space exploration, while a wind-coupled configuration enables higher-fidelity trajectory prediction and mission analysis. Both configurations share the same integrator, atmospheric model, and vehicle dynamics logic, allowing direct comparison and incremental validation.

The SDK is structured to support future extensions, including alternative atmospheric data sources, additional vehicle configurations, and integration with real flight data or high-altitude platform testing workflows.

## Quick Start (Windows)

This project uses **Miniforge + conda-forge** to ensure reliable builds for geospatial and GRIB libraries.

Download and install **Miniforge3 (Windows x86_64)**: https://conda-forge.org/miniforge/

```powershell
git clone https://github.com/Open-Orbit-Foundation/HAB_SDK.git
mamba env create -f environment.yml
conda activate hab-sdk
python main.py
```

### Rebuilding the Environment

If the environment becomes corrupted:

```powershell
conda deactivate
mamba env remove -n hab-sdk

# fallback cleanup (important on Windows)
Remove-Item -Recurse -Force "$env:USERPROFILE\miniforge3\envs\hab-sdk" -ErrorAction SilentlyContinue

mamba env create -f environment.yml
conda activate hab-sdk
```

## Dependencies
```
dependencies:
  - python=3.12
  - numpy
  - pandas
  - scipy
  - matplotlib
  - requests
  - tqdm
  - xarray
  - netcdf4
  - cfgrib
  - eccodes
  - herbie-data
  - pyproj
  - rasterio
  - geopandas
  - contextily
  - cartopy
  - jupyterlab
  - ipykernel
  ```