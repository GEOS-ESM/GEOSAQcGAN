## Directory Structure

Before running any of the evaluation and plotting workflows, load the GEOS
Python environment:

```bash
module load python/GEOSpyD/24.11.3-0/3.12
```

The plotting scripts use the Basemap package, which is installed in the shared
`p271` Python package directory. Add this directory to your `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH:-}:/gpfsm/dnb06/projects/p271/python_packages"
```

The plotting scripts are configured to use Matplotlib's non-interactive `Agg`
backend, making them compatible with Discover login and compute nodes. No
additional environment variables are required.



```text
evaluation/
├── create_hadisst_land_sea_mask.py
├── 1_leadmean_global_and_regional_timeseries/
├── 2_leadmean_global_maps/
├── 3_startdate_global_and_regional_timeseries/
└── 4_startdate_global_maps/
```

### `create_hadisst_land_sea_mask.py`

Creates the global land–sea mask used by the evaluation scripts to calculate
regional statistics and separate land and ocean diagnostics.

### `1_leadmean_global_and_regional_timeseries/`

Generates **lead-time mean global and regional time-series diagnostics** by
averaging prediction, truth, and RMSE values over all forecast initializations
for each of the 80 forecast lead times from 3 to 240 hours.

The resulting time series summarize AQcGAN performance as a function of
forecast lead time across multiple predefined regions.

The released figures and supporting data products are located at:

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_and_regional_timeseries/
```

**Purpose:** Reference and reproducibility. The released lead-time mean
diagnostics have already been generated, so users normally do not need to run
these scripts.

### `2_leadmean_global_maps/`

Generates **lead-time mean global maps** by averaging prediction, truth, and
absolute-error fields over all forecast initializations for each of the 80
forecast lead times from 3 to 240 hours.

The resulting maps summarize AQcGAN performance globally at every forecast
lead time.

The released figures and supporting map data are located at:

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_maps/
```

**Purpose:** Reference and reproducibility. The released lead-time mean maps
have already been generated, so users normally do not need to run these
scripts.

### `3_startdate_global_and_regional_timeseries/`

Generates **global and regional time-series diagnostics for a single forecast
initialization**.

Users select the following:

- Forecast initialization time
- Lead day
- Aerosol species
- Geographic region

The script produces prediction, truth, and RMSE time-series plots.

**Purpose:** Primary user-facing evaluation tool.

Run:

```bash
cd src/geosaqcgan/evaluation/v2_2_0/3_startdate_global_and_regional_timeseries

python plot_startdate_global_and_regional_timeseries.py
```

Outputs are written to:

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/startdate_global_and_regional_timeseries/
```

### `4_startdate_global_maps/`

Generates **global map diagnostics for a single forecast initialization**.

Users select the following:

- Forecast initialization time
- Lead day
- Aerosol species
- Forecast output frame

The script produces truth, prediction, and absolute-error maps.

**Purpose:** Primary user-facing evaluation tool.

Run:

```bash
cd src/geosaqcgan/evaluation/v2_2_0/4_startdate_global_maps

python plot_startdate_global_map.py
```

Outputs are written to:

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/startdate_global_maps/
```

---

## Included Plot Metadata

All plotting metadata has already been generated and is included with the
AQcGAN v2.2.0 evaluation products.

The metadata is located at:

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/
```

This directory contains the following precomputed files:

```text
hadisst_land_sea_mask_181x360.npz
leadmean_global_and_regional_timeseries_scales.npz
leadmean_global_map_scales.npz
startdate_global_and_regional_timeseries_scales.npz
startdate_global_map_scales.npz
```

The plotting scripts automatically use these files to provide consistent
plotting scales and regional land–sea masks.

**Users do not need to regenerate these files.**

They only need to be regenerated if:

- The evaluation dataset changes
- AQcGAN is retrained
- The evaluated variables or spatial grid changes
