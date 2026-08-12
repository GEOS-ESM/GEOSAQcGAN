#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute fixed y-axis scales for AQcGAN v2.2.0 regional time-series plots.

This script reads AQcGAN prediction and truth files for lead days 1–10 and
computes fixed truth/prediction and RMSE plotting limits for each aerosol
variable and region.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
        test_ens_pred_stats_days{DAY}_chkpt100.npz
        test_ens_stats_days{DAY}.npz

Land and ocean regions use:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/hadisst_land_sea_mask_181x360.npz

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/startdate_global_and_regional_timeseries_scales.npz

The output is used by:
    plot_startdate_global_and_regional_timeseries.py

This script computes plotting scales only and does not create figures.
"""

from pathlib import Path
import numpy as np

BASE = Path("/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0")
OUT_DIR = BASE / "plot_metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "startdate_global_and_regional_timeseries_scales.npz"
MASK_FILE = OUT_DIR / "hadisst_land_sea_mask_181x360.npz"

CHKPT = 100
START_DAY = 1
END_DAY = 10

SPECIES = [
    "BCPHILIC", "BCPHOBIC",
    "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a",
    "NO3an1", "NO3an2", "NO3an3",
    "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]

UNITS = np.array(["kg/kg"] * len(SPECIES))

REGIONS = {
    "global": None,
    "global_land": "land",
    "global_ocean": "ocean",
    "north_america": {"lat": (5, 85), "lon": (-170, -50)},
    "south_america": {"lat": (-60, 15), "lon": (-90, -30)},
    "europe": {"lat": (35, 72), "lon": (-25, 45)},
    "africa": {"lat": (-35, 38), "lon": (-20, 55)},
    "asia": {"lat": (0, 80), "lon": (45, 180)},
    "australia": {"lat": (-50, 0), "lon": (110, 180)},
    "tropics": {"lat": (-23.5, 23.5), "lon": (-180, 180)},
    "arctic": {"lat": (66.5, 90), "lon": (-180, 180)},
    "antarctic": {"lat": (-90, -66.5), "lon": (-180, 180)},
}


def make_region_mask(region_name, nlat, nlon):
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon, endpoint=False)

    region = REGIONS[region_name]

    if region is None:
        return np.ones((nlat, nlon), dtype=bool)

    if isinstance(region, str) and region in {"land", "ocean"}:
        mask_data = np.load(MASK_FILE)
        mask = mask_data["land_mask"] if region == "land" else mask_data["ocean_mask"]
        if mask.shape != (nlat, nlon):
            raise ValueError(f"Mask shape {mask.shape} does not match {(nlat, nlon)}")
        return mask.astype(bool)

    lat_mask = (lat >= region["lat"][0]) & (lat <= region["lat"][1])

    lon_min, lon_max = region["lon"]
    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    return lat_mask[:, None] & lon_mask[None, :]


def masked_mean(data, mask):
    masked = np.where(mask[None, :, :], data, np.nan)
    return np.nanmean(masked, axis=(1, 2))


def masked_rmse(pred, truth, mask):
    sqerr = (pred - truth) ** 2
    masked = np.where(mask[None, :, :], sqerr, np.nan)
    return np.sqrt(np.nanmean(masked, axis=(1, 2)))


def padded_limits(vmin, vmax, floor_zero=False):
    if np.isclose(vmin, vmax):
        pad = abs(vmax) * 0.05 if vmax != 0 else 1.0
    else:
        pad = 0.08 * (vmax - vmin)

    ymin = vmin - pad
    ymax = vmax + pad

    if floor_zero:
        ymin = 0.0

    return ymin, ymax


region_names = list(REGIONS.keys())
n_regions = len(region_names)
n_species = len(SPECIES)

main_vmin = np.full((n_regions, n_species), np.inf, dtype=np.float64)
main_vmax = np.full((n_regions, n_species), -np.inf, dtype=np.float64)
rmse_vmin = np.zeros((n_regions, n_species), dtype=np.float64)
rmse_vmax = np.full((n_regions, n_species), -np.inf, dtype=np.float64)
region_cell_count = np.zeros(n_regions, dtype=np.int64)
counts_by_day = []

region_masks = None

for day in range(START_DAY, END_DAY + 1):
    print("=" * 80, flush=True)
    print(f"Processing lead day {day}", flush=True)

    pred_file = BASE / f"test_ens_pred_stats_days{day}_chkpt{CHKPT}.npz"
    truth_file = BASE / f"test_ens_stats_days{day}.npz"

    print(f"Prediction: {pred_file}", flush=True)
    print(f"Truth     : {truth_file}", flush=True)

    with np.load(pred_file) as pnpz, np.load(truth_file) as tnpz:
        pred_all = pnpz["ens_mean_pred_test_inv"]
        truth_all = tnpz["ens_mean_test_inv"]

        print(f"Shape: {pred_all.shape}", flush=True)

        if region_masks is None:
            nlat = pred_all.shape[3]
            nlon = pred_all.shape[4]
            region_masks = [
                make_region_mask(region_name, nlat, nlon)
                for region_name in region_names
            ]
            region_cell_count[:] = [mask.sum() for mask in region_masks]

        counts_by_day.append(pred_all.shape[0])

        for region_idx, region_name in enumerate(region_names):
            mask = region_masks[region_idx]
            print(f"  Region: {region_name} cells={mask.sum()}", flush=True)

            for var_idx, var_name in enumerate(SPECIES):
                pred = pred_all[:, var_idx]      # samples, 8, lat, lon
                truth = truth_all[:, var_idx]    # samples, 8, lat, lon

                # Flatten samples and 3-hour frames together.
                pred_2d = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
                truth_2d = truth.reshape(-1, truth.shape[-2], truth.shape[-1])

                pred_mean = masked_mean(pred_2d, mask)
                truth_mean = masked_mean(truth_2d, mask)
                rmse = masked_rmse(pred_2d, truth_2d, mask)

                main_vmin[region_idx, var_idx] = min(
                    main_vmin[region_idx, var_idx],
                    np.nanmin(pred_mean),
                    np.nanmin(truth_mean),
                )
                main_vmax[region_idx, var_idx] = max(
                    main_vmax[region_idx, var_idx],
                    np.nanmax(pred_mean),
                    np.nanmax(truth_mean),
                )
                rmse_vmax[region_idx, var_idx] = max(
                    rmse_vmax[region_idx, var_idx],
                    np.nanmax(rmse),
                )

                print(f"    {var_idx:02d} {var_name}", flush=True)

    main_ymin = np.empty_like(main_vmin)
    main_ymax = np.empty_like(main_vmax)
    rmse_ymin = np.zeros_like(rmse_vmin)
    rmse_ymax = np.empty_like(rmse_vmax)

    for region_idx in range(n_regions):
        for var_idx in range(n_species):
            main_ymin[region_idx, var_idx], main_ymax[region_idx, var_idx] = padded_limits(
                main_vmin[region_idx, var_idx],
                main_vmax[region_idx, var_idx],
                floor_zero=False,
            )
            rmse_ymin[region_idx, var_idx], rmse_ymax[region_idx, var_idx] = padded_limits(
                0.0,
                rmse_vmax[region_idx, var_idx],
                floor_zero=True,
            )

    np.savez_compressed(
        OUT_FILE,
        regions=np.array(region_names),
        species=np.array(SPECIES),
        units=UNITS,
        main_vmin=main_vmin,
        main_vmax=main_vmax,
        main_ymin=main_ymin,
        main_ymax=main_ymax,
        rmse_vmin=rmse_vmin,
        rmse_vmax=rmse_vmax,
        rmse_ymin=rmse_ymin,
        rmse_ymax=rmse_ymax,
        region_cell_count=region_cell_count,
        counts_by_day=np.array(counts_by_day),
    )

    print(f"Checkpoint saved: {OUT_FILE}", flush=True)

print("Done.", flush=True)
print(f"Saved: {OUT_FILE}", flush=True)