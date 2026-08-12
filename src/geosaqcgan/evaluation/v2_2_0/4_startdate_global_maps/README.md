# Start-Date Global Maps

## Overview

This directory contains the scripts used to generate **global map diagnostics
for a single AQcGAN v2.2.0 forecast initialization**.

Given a forecast lead day, aerosol species, forecast initialization time, and
3-hour forecast output frame, the plotting script extracts the corresponding
prediction and truth fields, computes the absolute error, and generates a
three-panel global diagnostic figure.

To ensure consistent color scales across all figures, the plotting script uses
precomputed color-scale metadata included with the AQcGAN v2.2.0 release.

---

## Directory Contents

| File | Description |
|------|-------------|
| `compute_startdate_global_map_scales.py` | Computes fixed color scales for each aerosol species. |
| `plot_startdate_global_map.py` | Generates a global diagnostic map for a selected forecast initialization and output frame. |
| `run_compute_startdate_global_map_scales.sh` | SLURM launcher for computing the color-scale metadata. |

---

## Important

The color-scale metadata required by the plotting script is already included
with the AQcGAN v2.2.0 release.

Most users only need to run the plotting script.

The computation script is included for

- Reproducibility
- Regeneration of the color-scale metadata after retraining AQcGAN
- Adapting the workflow to a different evaluation dataset

---

## Color-Scale Metadata

The plotting script uses the precomputed color-scale metadata

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    plot_metadata/
        startdate_global_map_scales.npz
```

This file is included with the AQcGAN v2.2.0 release and is automatically used
by the plotting script. It normally does not need to be regenerated.

If the evaluation dataset changes, regenerate the color-scale metadata with

```bash
sbatch run_compute_startdate_global_map_scales.sh
```

or

```bash
python compute_startdate_global_map_scales.py
```

---

## Input Data

The plotting script reads

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/

    test_ens_pred_stats_days{DAY}_chkpt100.npz
    test_ens_stats_days{DAY}.npz
```

It also uses the precomputed color-scale metadata

```text
plot_metadata/
    startdate_global_map_scales.npz
```

---

## Outputs

The plotting script writes its outputs to

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_maps/
```

Each run produces

```text
{VARIABLE}_day{DAY}_{START_TIME}_valid{VALID_TIME}_maps.png
```

and

```text
{VARIABLE}_day{DAY}_{START_TIME}_valid{VALID_TIME}_maps.npz
```

The NPZ file contains

- Prediction map
- Truth map
- Absolute-error map
- Latitude and longitude grids
- Forecast metadata
- Color-scale limits

The figure contains three global maps:

- Truth
- AQcGAN Prediction
- Absolute Error

---

## Example

Run

```bash
python plot_startdate_global_map.py
```

Example session:

```text
Lead day, 1-10: 3
Variable/species, e.g. BCPHILIC: SO4
Forecast start date/time, e.g. 2018-06-01 00: 2018-07-15 12

Available output frames for this forecast:
  0: 2018-07-18 12Z
  1: 2018-07-18 15Z
  2: 2018-07-18 18Z
  3: 2018-07-18 21Z
  4: 2018-07-19 00Z
  5: 2018-07-19 03Z
  6: 2018-07-19 06Z
  7: 2018-07-19 09Z

Choose output frame index, 0-7: 4
```

The script will then

1. Read the AQcGAN prediction and truth files for the selected lead day.
2. Locate the requested forecast initialization.
3. Extract the selected 3-hour forecast frame.
4. Compute the absolute error.
5. Apply the precomputed color scales.
6. Save both a PNG figure and an NPZ file containing the plotted data.

For the example above, the outputs would be

```text
/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_maps/
        SO4_day3_20180715_1200_valid20180719_0000_maps.png

/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
    startdate_global_maps/
        SO4_day3_20180715_1200_valid20180719_0000_maps.npz
```

---

## Requirements

The scripts require

- Python 3.12
- NumPy
- Matplotlib
- Basemap

The provided SLURM launcher is configured for the NASA Discover GEOSpyD
environment.