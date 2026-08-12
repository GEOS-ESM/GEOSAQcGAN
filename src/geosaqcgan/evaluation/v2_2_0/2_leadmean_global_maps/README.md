# Lead-Time Mean Global Maps

## Overview

This directory contains the scripts used to generate and visualize **lead-time
mean global maps** for AQcGAN v2.2.0 aerosol forecasts.

Unlike the start-date diagnostics, which visualize a single forecast
initialization, these diagnostics average **all forecast initializations** in
the evaluation dataset. They provide an overall assessment of AQcGAN
performance as a function of forecast lead time by averaging statistics across
all forecast initializations.

For each aerosol variable and forecast lead time, the evaluation computes:

- Mean AQcGAN prediction
- Mean truth
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

The resulting maps summarize AQcGAN performance at every forecast lead time.

---

## Directory Contents

| File | Description |
|------|-------------|
| `compute_leadmean_global_maps_data_and_scales.py` | Computes lead-time mean global map statistics and fixed color scales. |
| `plot_leadmean_global_maps.py` | Generates lead-time mean global map figures from the precomputed datasets. |

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

---

## Precomputed Map Data

The lead-time mean map datasets are included with the AQcGAN v2.2.0 release and
are located in

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    leadmean_global_maps/
```

Each aerosol variable is stored as

```text
{VARIABLE}_leadmean_maps.npz
```

Examples

```text
SO4_leadmean_maps.npz
BCPHILIC_leadmean_maps.npz
DU003_leadmean_maps.npz
```

Each file contains

- Mean AQcGAN prediction
- Mean truth
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Forecast lead times
- Latitude and longitude grids

These files are included with the AQcGAN v2.2.0 release and normally do not
need to be regenerated.

---

## Plotting Metadata

To ensure consistent color scales across all figures, the plotting script uses
the precomputed metadata file

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    plot_metadata/
        leadmean_global_map_scales.npz
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
    leadmean_global_maps/
        plots/
```

Each figure is saved as

```text
{VARIABLE}_lead{LEAD}_truth_pred_rmse.png
```

Examples

```text
SO4_lead024_truth_pred_rmse.png
BCPHILIC_lead120_truth_pred_rmse.png
DU003_lead240_truth_pred_rmse.png
```

Each figure contains three global maps:

1. Mean truth
2. Mean AQcGAN prediction
3. Root Mean Square Error (RMSE)

---

## Viewing Existing Results

To regenerate figures from the existing lead-time mean datasets, run

```bash
python plot_leadmean_global_maps.py \
    --var BCPHILIC \
    --leads 24 120 240
```

or generate figures for all forecast lead times

```bash
python plot_leadmean_global_maps.py \
    --var BCPHILIC \
    --leads $(seq 3 3 240)
```

By default, figures are written to

```text
leadmean_global_maps/plots/
```

---

## Regenerating the Evaluation Products

Recomputation is **not required** for normal use.

Recompute the lead-time mean datasets only if

- The AQcGAN prediction files change
- The truth dataset changes
- The evaluation checkpoint changes
- Additional forecast cases are added

Regenerate the datasets with

```bash
python compute_leadmean_global_maps_data_and_scales.py
```

This recreates the lead-time mean datasets in

```text
leadmean_global_maps/
```

and updates

```text
plot_metadata/
    leadmean_global_map_scales.npz
```

After recomputing the datasets, regenerate the figures with

```bash
python plot_leadmean_global_maps.py \
    --var BCPHILIC \
    --leads $(seq 3 3 240)
```

---

## Requirements

The scripts require

- Python 3.12
- NumPy
- Matplotlib
- Basemap

The scripts are compatible with the NASA Discover GEOSpyD environment.