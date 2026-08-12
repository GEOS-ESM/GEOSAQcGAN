## Lead-Time Mean Global and Regional Time Series

This directory contains the scripts used to generate and visualize **lead-time
mean global and regional time-series diagnostics** for AQcGAN v2.2.0 aerosol
forecasts.

Unlike the start-date diagnostics, which examine a single forecast
initialization, these diagnostics average **all forecast initializations** in
the evaluation dataset. They provide an overall assessment of AQcGAN
performance as a function of forecast lead time by averaging statistics across
all forecast initializations.

For each aerosol variable and region, the evaluation computes:

- Mean AQcGAN prediction
- Mean truth
- Root Mean Square Error (RMSE)

using all available forecasts from lead days 1–10 (3–240 forecast hours).

---

## Directory Contents

| File | Description |
|------|-------------|
| `compute_leadmean_global_and_regional_timeseries_data_and_scales.py` | Computes lead-time mean global and regional statistics and fixed plotting scales. |
| `plot_leadmean_global_and_regional_timeseries.py` | Generates lead-time mean figures from the precomputed datasets. |
| `run_compute_leadmean_global_and_regional_timeseries_data_and_scales.sh` | SLURM launcher for the computation script. |

---

## Important

The evaluation products in this directory have already been generated for the
AQcGAN v2.2.0 release.

Most users **do not need to rerun** either the computation or plotting scripts.

The scripts are included for:

- Reproducibility
- Regeneration of the evaluation products after retraining AQcGAN
- Adapting the workflow to a different evaluation dataset

---

## Input Data

The computation script reads the AQcGAN prediction and truth files located in

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
```

using the following file patterns:

```text
test_ens_pred_stats_days{DAY}_chkpt100.npz
test_ens_stats_days{DAY}.npz
```

where

```text
DAY = 1 ... 10
```

Each file contains

```text
samples × variables × forecast frames × latitude × longitude
```

The computation also uses the HadISST-derived land/ocean mask

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    plot_metadata/
        hadisst_land_sea_mask_181x360.npz
```

to compute global land, global ocean, and predefined regional statistics.

---

## Precomputed Time-Series Data

The lead-time mean datasets are included with the AQcGAN v2.2.0 release and
are located in

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    leadmean_global_and_regional_timeseries/
```

Each variable-region pair is stored as

```text
{VARIABLE}_{REGION}_mean_timeseries.npz
```

For example

```text
SO4_asia_mean_timeseries.npz
BCPHILIC_global_mean_timeseries.npz
SS005_global_land_mean_timeseries.npz
```

Each file contains

- Forecast lead times
- Regional mean prediction
- Regional mean truth
- Regional RMSE
- Variable metadata
- Region metadata
- Sample counts

These files are included with the AQcGAN v2.2.0 release and normally do not
need to be regenerated.

---

## Plotting Metadata

To ensure consistent axes across figures, the plotting script uses the
precomputed scale file

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    plot_metadata/
        leadmean_global_and_regional_timeseries_scales.npz
```

This file is included with the AQcGAN v2.2.0 release and is automatically used
by the plotting script. It normally does not need to be regenerated.

---

## Generated Figures

The released figures are already included with the AQcGAN v2.2.0 evaluation
products. Users normally do not need to regenerate them.

The figures are stored in

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    leadmean_global_and_regional_timeseries/
        plots/
```

Each figure is saved as

```text
{VARIABLE}_{REGION}_mean_timeseries.png
```

Examples

```text
SO4_asia_mean_timeseries.png
DU003_global_ocean_mean_timeseries.png
```

Each figure contains

- Regional mean truth
- Regional mean AQcGAN prediction
- Regional RMSE

as a function of forecast lead time.

---

## Supported Regions

The scripts support the following regions:

- global
- global_land
- global_ocean
- north_america
- south_america
- europe
- africa
- asia
- australia
- tropics
- arctic
- antarctic

---

## Viewing Existing Results

To regenerate the figures from the existing lead-time mean datasets, run

```bash
python plot_leadmean_global_and_regional_timeseries.py
```

The plotting script can generate individual plots or all available
variable-region combinations.

For example

```bash
python plot_leadmean_global_and_regional_timeseries.py \
    --variable SO4 \
    --region asia
```

or

```bash
python plot_leadmean_global_and_regional_timeseries.py --all
```

By default, figures are written to

```text
leadmean_global_and_regional_timeseries/plots/
```

---

## Regenerating the Evaluation Products

Recomputation is **not required** for normal use.

Recompute the lead-time mean datasets only if

- The AQcGAN prediction files change
- The truth dataset changes
- The evaluation checkpoint changes
- The regional definitions are modified

On NASA Discover, regenerate the datasets with

```bash
sbatch run_compute_leadmean_global_and_regional_timeseries_data_and_scales.sh
```

This recreates the lead-time mean datasets in

```text
leadmean_global_and_regional_timeseries/
```

and updates

```text
plot_metadata/
    leadmean_global_and_regional_timeseries_scales.npz
```

After recomputing the datasets, regenerate the figures with

```bash
python plot_leadmean_global_and_regional_timeseries.py --all
```

---

## Requirements

The scripts require

- Python 3.12
- NumPy
- Matplotlib
- Basemap
- NetCDF4 (required only for land/ocean mask generation)

The provided SLURM launcher is configured for the NASA Discover GEOSpyD
environment.