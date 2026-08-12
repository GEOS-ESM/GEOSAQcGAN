#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_leadmean_global_and_regional_timeseries_data_and_scales.py

Compute regional lead-time mean statistics for AQcGAN v2.2.0 aerosol forecasts.

This script processes AQcGAN prediction and truth files for all 10 forecast days
(3-240 hour lead times) and computes global and regional lead-mean prediction, truth, and RMSE
time series for each aerosol variable. It saves one output file per
variable-region pair along with a metadata file containing fixed plotting scales
used by the regional time-series plotting scripts.

Inputs:
    - /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/test_ens_pred_stats_days{day}_chkpt100.npz
    - /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/test_ens_stats_days{day}.npz
    - /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/hadisst_land_sea_mask_181x360.npz

Outputs:
    - /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_and_regional_timeseries/{VARIABLE}_{REGION}_mean_timeseries.npz
    - /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/leadmean_global_and_regional_timeseries_scales.npz
"""


import argparse
from pathlib import Path

import numpy as np


VARS = [
    "BCPHILIC", "BCPHOBIC",
    "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a",
    "NO3an1", "NO3an2", "NO3an3",
    "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]

UNITS = np.array(["kg/kg"] * len(VARS))

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


def print_regions():
    print("Regions:")
    for name, region in REGIONS.items():
        if region is None:
            print(f"  {name:15s} lat: -90 to 90      lon: -180 to 180")
        elif region == "land":
            print(f"  {name:15s} lat: -90 to 90      lon: -180 to 180      land only")
        elif region == "ocean":
            print(f"  {name:15s} lat: -90 to 90      lon: -180 to 180      ocean only")
        else:
            lat0, lat1 = region["lat"]
            lon0, lon1 = region["lon"]
            print(f"  {name:15s} lat: {lat0:6.1f} to {lat1:6.1f}   lon: {lon0:7.1f} to {lon1:7.1f}")


def make_region_mask(region_name, nlat, nlon, mask_file):
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon, endpoint=False)

    region = REGIONS[region_name]

    if region is None:
        return np.ones((nlat, nlon), dtype=bool)

    if isinstance(region, str) and region in {"land", "ocean"}:
        mask_data = np.load(mask_file)
        key = "land_mask" if region == "land" else "ocean_mask"
        mask = mask_data[key].astype(bool)

        if mask.shape != (nlat, nlon):
            raise ValueError(f"{key} shape {mask.shape} does not match {(nlat, nlon)}")

        return mask

    lat_mask = (lat >= region["lat"][0]) & (lat <= region["lat"][1])

    lon_min, lon_max = region["lon"]
    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    return lat_mask[:, None] & lon_mask[None, :]


def masked_mean_over_samples_and_space(data, mask):
    masked = np.where(mask[None, None, :, :], data, np.nan)
    return np.nanmean(masked, axis=(0, 2, 3))


def masked_rmse_over_samples_and_space(pred, truth, mask):
    sqerr = (pred - truth) ** 2
    masked = np.where(mask[None, None, :, :], sqerr, np.nan)
    return np.sqrt(np.nanmean(masked, axis=(0, 2, 3)))


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


def save_variable_region_file(
    out_dir,
    var_name,
    var_idx,
    region_name,
    leads,
    pred_ts,
    truth_ts,
    rmse_ts,
    region_cell_count,
    counts_by_day,
):
    out_file = out_dir / f"{var_name}_{region_name}_mean_timeseries.npz"

    np.savez_compressed(
        out_file,
        leads=leads,
        variable=var_name,
        var_idx=var_idx,
        region=region_name,
        units="kg/kg",
        region_cell_count=region_cell_count,
        pred_ts=np.array(pred_ts, dtype=np.float64),
        truth_ts=np.array(truth_ts, dtype=np.float64),
        rmse_ts=np.array(rmse_ts, dtype=np.float64),
        counts_by_day=np.array(counts_by_day, dtype=np.int64),
    )

    print(f"Saved: {out_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute AQcGAN regional mean time series and regional plot scales."
    )
    parser.add_argument(
        "--base",
        default="/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0",
        help="Directory containing prediction/truth npz files.",
    )
    parser.add_argument("--chkpt", type=int, default=100)
    parser.add_argument(
        "--mask-file",
        default=None,
        help="HadISST land/sea mask npz. Default: <base>/plot_metadata/hadisst_land_sea_mask_181x360.npz",
    )
    args = parser.parse_args()

    base = Path(args.base)
    out_dir = base / "leadmean_global_and_regional_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = base / "plot_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    mask_file = Path(args.mask_file) if args.mask_file else metadata_dir / "hadisst_land_sea_mask_181x360.npz"
    scale_file = metadata_dir / "leadmean_global_and_regional_timeseries_scales.npz"

    leads = np.arange(3, 241, 3)

    print_regions()
    print(f"\nUsing base      : {base}", flush=True)
    print(f"Using mask file : {mask_file}", flush=True)
    print(f"Output dir      : {out_dir}", flush=True)
    print(f"Scale file      : {scale_file}\n", flush=True)

    region_names = list(REGIONS.keys())
    n_regions = len(region_names)
    n_vars = len(VARS)

    pred_store = [[[] for _ in range(n_vars)] for _ in range(n_regions)]
    truth_store = [[[] for _ in range(n_vars)] for _ in range(n_regions)]
    rmse_store = [[[] for _ in range(n_vars)] for _ in range(n_regions)]

    counts_by_day = []

    masks = None
    region_cell_count = np.zeros(n_regions, dtype=np.int64)

    for day in range(1, 11):
        print("=" * 80, flush=True)
        print(f"Processing lead day {day}", flush=True)

        pred_file = base / f"test_ens_pred_stats_days{day}_chkpt{args.chkpt}.npz"
        truth_file = base / f"test_ens_stats_days{day}.npz"

        if not pred_file.exists() or not truth_file.exists():
            print(f"Missing day {day}, skipping.", flush=True)
            continue

        print(f"Prediction: {pred_file}", flush=True)
        print(f"Truth     : {truth_file}", flush=True)

        with np.load(pred_file) as pnpz, np.load(truth_file) as tnpz:
            pred_all = pnpz["ens_mean_pred_test_inv"]
            truth_all = tnpz["ens_mean_test_inv"]

            print(f"Shape: {pred_all.shape}", flush=True)
            counts_by_day.append(pred_all.shape[0])

            if masks is None:
                nlat, nlon = pred_all.shape[3], pred_all.shape[4]
                masks = [
                    make_region_mask(region_name, nlat, nlon, mask_file)
                    for region_name in region_names
                ]
                region_cell_count[:] = [mask.sum() for mask in masks]

                print("Region cell counts:", flush=True)
                for region_name, count in zip(region_names, region_cell_count):
                    print(f"  {region_name:15s}: {count}", flush=True)

            for var_idx, var_name in enumerate(VARS):
                print(f"  Variable {var_idx:02d}: {var_name}", flush=True)

                pred = pred_all[:, var_idx]
                truth = truth_all[:, var_idx]

                for region_idx, region_name in enumerate(region_names):
                    mask = masks[region_idx]

                    pred_day = masked_mean_over_samples_and_space(pred, mask)
                    truth_day = masked_mean_over_samples_and_space(truth, mask)
                    rmse_day = masked_rmse_over_samples_and_space(pred, truth, mask)

                    pred_store[region_idx][var_idx].extend(pred_day)
                    truth_store[region_idx][var_idx].extend(truth_day)
                    rmse_store[region_idx][var_idx].extend(rmse_day)

        print(f"Finished lead day {day}", flush=True)

    main_vmin = np.full((n_regions, n_vars), np.inf)
    main_vmax = np.full((n_regions, n_vars), -np.inf)
    rmse_vmax = np.full((n_regions, n_vars), -np.inf)

    for region_idx, region_name in enumerate(region_names):
        for var_idx, var_name in enumerate(VARS):
            pred_ts = np.array(pred_store[region_idx][var_idx], dtype=np.float64)
            truth_ts = np.array(truth_store[region_idx][var_idx], dtype=np.float64)
            rmse_ts = np.array(rmse_store[region_idx][var_idx], dtype=np.float64)

            if len(pred_ts) != len(leads):
                raise ValueError(
                    f"Lead mismatch for {region_name}/{var_name}: "
                    f"got {len(pred_ts)}, expected {len(leads)}"
                )

            main_vmin[region_idx, var_idx] = min(np.nanmin(pred_ts), np.nanmin(truth_ts))
            main_vmax[region_idx, var_idx] = max(np.nanmax(pred_ts), np.nanmax(truth_ts))
            rmse_vmax[region_idx, var_idx] = np.nanmax(rmse_ts)

            save_variable_region_file(
                out_dir=out_dir,
                var_name=var_name,
                var_idx=var_idx,
                region_name=region_name,
                leads=leads,
                pred_ts=pred_ts,
                truth_ts=truth_ts,
                rmse_ts=rmse_ts,
                region_cell_count=region_cell_count[region_idx],
                counts_by_day=counts_by_day,
            )

    main_ymin = np.empty_like(main_vmin)
    main_ymax = np.empty_like(main_vmax)
    rmse_ymin = np.zeros_like(rmse_vmax)
    rmse_ymax = np.empty_like(rmse_vmax)

    for region_idx in range(n_regions):
        for var_idx in range(n_vars):
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
        scale_file,
        regions=np.array(region_names),
        variables=np.array(VARS),
        species=np.array(VARS),
        units=UNITS,
        region_cell_count=region_cell_count,
        main_vmin=main_vmin,
        main_vmax=main_vmax,
        main_ymin=main_ymin,
        main_ymax=main_ymax,
        rmse_vmin=np.zeros_like(rmse_vmax),
        rmse_vmax=rmse_vmax,
        rmse_ymin=rmse_ymin,
        rmse_ymax=rmse_ymax,
        counts_by_day=np.array(counts_by_day, dtype=np.int64),
    )

    print("\nDone.", flush=True)
    print(f"Saved scale file: {scale_file}", flush=True)


if __name__ == "__main__":
    main()