# Start-Date Global and Regional Time Series

## Overview

This directory contains the scripts used to generate **global and regional
time-series diagnostics for a single AQcGAN v2.2.0 forecast initialization**.

Given a forecast lead day, aerosol species, forecast initialization time, and
region, the plotting script computes the regional mean prediction, regional
mean truth, and regional RMSE for the eight 3-hour forecast frames and
generates a time-series figure.

To ensure consistent y-axis limits across all figures, the plotting script uses
precomputed plotting scales included with the AQcGAN v2.2.0 release.

---

## Directory Contents

| File | Description |
|------|-------------|
| `compute_startdate_global_and_regional_timeseries_scales.py` | Computes fixed plotting scales for all aerosol species and regions. |
| `plot_startdate_global_and_regional_timeseries.py` | Generates a regional time-series plot for a selected forecast initialization. |
| `run_compute_startdate_global_and_regional_timeseries_scales.sh` | SLURM launcher for computing the plotting scales. |

---

## Important

The plotting scale file required by the plotting script is already included
with the AQcGAN v2.2.0 release.

Most users only need to run the plotting script.

The computation script is included for

- Reproducibility
- Regeneration of the plotting scales after retraining AQcGAN
- Adapting the workflow to a different evaluation dataset

---

## Plotting Scales

The plotting script uses the precomputed scale file

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    plot_metadata/
        startdate_global_and_regional_timeseries_scales.npz
```

This file is included with the AQcGAN v2.2.0 release and is automatically used
by the plotting script. It normally does not need to be regenerated.

If the evaluation dataset changes, regenerate the plotting scales with

```bash
sbatch run_compute_startdate_global_and_regional_timeseries_scales.sh
```

or

```bash
python compute_startdate_global_and_regional_timeseries_scales.py
```

---

## Input Data

The plotting script reads

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/

    test_ens_pred_stats_days{DAY}_chkpt100.npz
    test_ens_stats_days{DAY}.npz
```

It also uses

```text
plot_metadata/
    hadisst_land_sea_mask_181x360.npz
    startdate_global_and_regional_timeseries_scales.npz
```

for the land/ocean masks and fixed plotting scales.

---

## Outputs

The plotting script writes

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_and_regional_timeseries/
```

Each run produces

```text
{VARIABLE}_{REGION}_day{DAY}_{START_TIME}_pred_truth_rmse.png
```

and

```text
{VARIABLE}_{REGION}_day{DAY}_{START_TIME}_pred_truth_rmse.csv
```

The CSV contains

- Valid forecast times
- Regional mean prediction
- Regional mean truth
- Regional RMSE
- Region information
- Variable information
- Plotting limits

The figure contains two panels:

- Regional mean prediction and truth
- Regional RMSE

for the selected forecast initialization.

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

## Example

Run the plotting script

```bash
python plot_startdate_global_and_regional_timeseries.py
```

The script will display the available regions and then prompt for the required
inputs.

Example session:

```text
Available regions:
  global
  global_land
  global_ocean
  north_america
  south_america
  europe
  africa
  asia
  australia
  tropics
  arctic
  antarctic

Lead day, 1-10: 3
Variable/species, e.g. BCPHILIC: SO4
Forecast start date/time, e.g. 2018-06-01 00: 2018-07-15 12
Region, e.g. global, global_land, global_ocean, asia: asia
```

The script will then

1. Read the AQcGAN prediction and truth files for the selected lead day.
2. Locate the requested forecast initialization.
3. Compute the regional mean prediction, regional mean truth, and regional RMSE.
4. Apply the precomputed plotting scales.
5. Save both a PNG figure and a CSV containing the plotted values.

For the example above, the outputs would be

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_and_regional_timeseries/
        SO4_asia_day3_20180715_1200_pred_truth_rmse.png

/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_and_regional_timeseries/
        SO4_asia_day3_20180715_1200_pred_truth_rmse.csv
```

---

## Requirements

The scripts require

- Python 3.12
- NumPy
- Pandas
- Matplotlib

The provided SLURM launcher is configured for the NASA Discover GEOSpyD
environment.