#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot regional time series for a single AQcGAN v2.2.0 forecast initialization.

This script reads AQcGAN prediction and truth files for a selected lead day,
aerosol variable, forecast initialization time, and region. It computes the
regional mean prediction, truth, and RMSE for the eight 3-hour forecast frames
and generates a time-series plot.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
        test_ens_pred_stats_days{DAY}_chkpt100.npz
        test_ens_stats_days{DAY}.npz

The script uses:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/
        hadisst_land_sea_mask_181x360.npz
        startdate_global_and_regional_timeseries_scales.npz

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/startdate_global_and_regional_timeseries/
        {VARIABLE}_{REGION}_day{DAY}_{START_TIME}_pred_truth_rmse.png
        {VARIABLE}_{REGION}_day{DAY}_{START_TIME}_pred_truth_rmse.csv

Example:
    python plot_startdate_global_and_regional_timeseries.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for Discover/batch nodes

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0")

OUT_DIR = BASE / "startdate_global_and_regional_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_FILE = BASE / "plot_metadata" / "hadisst_land_sea_mask_181x360.npz"
REGION_SCALE_FILE = BASE / "plot_metadata" / "startdate_global_and_regional_timeseries_scales.npz"

SPECIES = [
    "BCPHILIC", "BCPHOBIC",
    "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a",
    "NO3an1", "NO3an2", "NO3an3",
    "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]

UNITS = {name: "kg/kg" for name in SPECIES}

DESCRIPTIONS = {
    "BCPHILIC": "Hydrophilic Black Carbon",
    "BCPHOBIC": "Hydrophobic Black Carbon",
    "OCPHILIC": "Hydrophilic Organic Carbon",
    "OCPHOBIC": "Hydrophobic Organic Carbon",
    "DU001": "Dust Mixing Ratio Bin 001",
    "DU002": "Dust Mixing Ratio Bin 002",
    "DU003": "Dust Mixing Ratio Bin 003",
    "DU004": "Dust Mixing Ratio Bin 004",
    "DU005": "Dust Mixing Ratio Bin 005",
    "NH4a": "Ammonium ion aerosol phase",
    "NO3an1": "Nitrate size bin 001",
    "NO3an2": "Nitrate size bin 002",
    "NO3an3": "Nitrate size bin 003",
    "SO4": "Sulphate aerosol",
    "SS001": "Sea Salt Mixing Ratio Bin 001",
    "SS002": "Sea Salt Mixing Ratio Bin 002",
    "SS003": "Sea Salt Mixing Ratio Bin 003",
    "SS004": "Sea Salt Mixing Ratio Bin 004",
    "SS005": "Sea Salt Mixing Ratio Bin 005",
}

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
    print("Available regions:")
    for name in REGIONS:
        print(f"  {name}")

def get_case_index(start_dt):
    first_start = datetime(2018, 6, 1, 0)
    delta_hours = int((start_dt - first_start).total_seconds() // 3600)

    if delta_hours < 0:
        raise ValueError("Start time is before 2018-06-01 00Z.")

    return delta_hours


def load_region_mask(region_name, nlat, nlon):
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


def masked_mean_3hour(data, mask):
    masked = np.where(mask[None, :, :], data, np.nan)
    return np.nanmean(masked, axis=(1, 2))


def masked_rmse_3hour(pred, truth, mask):
    sqerr = (pred - truth) ** 2
    masked = np.where(mask[None, :, :], sqerr, np.nan)
    return np.sqrt(np.nanmean(masked, axis=(1, 2)))


def padded_ylim(values, floor_zero=False):
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isclose(vmin, vmax):
        pad = abs(vmax) * 0.05 if vmax != 0 else 1.0
    else:
        pad = 0.08 * (vmax - vmin)

    ymin = vmin - pad
    ymax = vmax + pad

    if floor_zero:
        ymin = 0.0

    return ymin, ymax


def get_region_scales(region, variable):
    if not REGION_SCALE_FILE.exists():
        print(f"Scale file not found, using auto-scale: {REGION_SCALE_FILE}")
        return None, None

    data = np.load(REGION_SCALE_FILE, allow_pickle=True)

    regions = [str(x) for x in data["regions"]]
    species = [str(x) for x in data["species"]]

    if region not in regions:
        print(f"Region {region} not found in scale file, using auto-scale.")
        return None, None

    if variable not in species:
        print(f"Variable {variable} not found in scale file, using auto-scale.")
        return None, None

    region_idx = regions.index(region)
    var_idx = species.index(variable)

    main_ylim = (
        float(data["main_ymin"][region_idx, var_idx]),
        float(data["main_ymax"][region_idx, var_idx]),
    )
    rmse_ylim = (
        0.0,
        float(data["rmse_ymax"][region_idx, var_idx]),
    )

    print(f"Using fixed scales from: {REGION_SCALE_FILE}")
    print(f"Scale region: {region}")
    print(f"Scale variable: {variable}")

    return main_ylim, rmse_ylim


def main():
    print_regions()

    lead_day = int(input("\nLead day, 1-10: ").strip())
    variable = input("Variable/species, e.g. BCPHILIC: ").strip()
    start_text = input("Forecast start date/time, e.g. 2018-06-01 00: ").strip()
    region = input("Region, e.g. global, global_land, global_ocean, asia: ").strip()

    if lead_day < 1 or lead_day > 10:
        raise ValueError("Lead day must be 1 through 10.")

    if variable not in SPECIES:
        raise ValueError(f"Unknown variable: {variable}")

    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")

    start_dt = datetime.strptime(start_text, "%Y-%m-%d %H")
    case_idx = get_case_index(start_dt)
    var_idx = SPECIES.index(variable)

    pred_file = BASE / f"test_ens_pred_stats_days{lead_day}_chkpt100.npz"
    truth_file = BASE / f"test_ens_stats_days{lead_day}.npz"

    with np.load(pred_file) as pnpz, np.load(truth_file) as tnpz:
        pred_all = pnpz["ens_mean_pred_test_inv"]
        truth_all = tnpz["ens_mean_test_inv"]

        if case_idx >= pred_all.shape[0]:
            raise ValueError(
                f"Start time maps to case index {case_idx}, but file only has "
                f"{pred_all.shape[0]} cases."
            )

        pred = pred_all[case_idx, var_idx]
        truth = truth_all[case_idx, var_idx]

    nlat, nlon = pred.shape[1], pred.shape[2]
    mask = load_region_mask(region, nlat, nlon)

    pred_mean = masked_mean_3hour(pred, mask)
    truth_mean = masked_mean_3hour(truth, mask)
    rmse = masked_rmse_3hour(pred, truth, mask)

    valid_times = [
        start_dt + timedelta(days=lead_day, hours=3 * i)
        for i in range(8)
    ]

    unit = UNITS[variable]
    description = DESCRIPTIONS[variable]

    main_ylim, rmse_ylim = get_region_scales(region, variable)

    if main_ylim is None:
        main_ylim = padded_ylim(np.r_[pred_mean, truth_mean], floor_zero=False)

    if rmse_ylim is None:
        rmse_ylim = padded_ylim(rmse, floor_zero=True)

    df = pd.DataFrame({
        "valid_time_utc": [t.strftime("%Y-%m-%d %HZ") for t in valid_times],
        "lead_day": lead_day,
        "forecast_start_utc": start_dt.strftime("%Y-%m-%d %HZ"),
        "region": region,
        "region_cells_used": int(mask.sum()),
        "total_grid_cells": int(mask.size),
        "variable": variable,
        "description": description,
        "unit": unit,
        "predicted_regional_mean": pred_mean,
        "truth_regional_mean": truth_mean,
        "regional_rmse": rmse,
        "mean_ymin": main_ylim[0],
        "mean_ymax": main_ylim[1],
        "rmse_ymin": rmse_ylim[0],
        "rmse_ymax": rmse_ylim[1],
    })

    safe_start = start_dt.strftime("%Y%m%d_%H%M")
    stem = f"{variable}_{region}_day{lead_day}_{safe_start}_pred_truth_rmse"

    out_csv = OUT_DIR / f"{stem}.csv"
    out_png = OUT_DIR / f"{stem}.png"

    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(valid_times, truth_mean, marker="o", label="Truth")
    axes[0].plot(valid_times, pred_mean, marker="o", label="Prediction")
    axes[0].set_title(f"{variable} {region} Mean")
    axes[0].set_ylabel(unit)
    axes[0].set_ylim(*main_ylim)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(valid_times, rmse, marker="o", color="crimson")
    axes[1].set_title(f"{variable} {region} RMSE")
    axes[1].set_ylabel(unit)
    axes[1].set_ylim(*rmse_ylim)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Valid Time")
        ax.tick_params(axis="x", rotation=35)

    fig.suptitle(
        f"AQcGAN v2.2.0 | {description}\n"
        f"Forecast Initialization Time: {start_dt:%Y-%m-%d %HZ} | Lead Day {lead_day} | Region: {region}",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {out_png}")
    print(f"Saved values: {out_csv}")

if __name__ == "__main__":
    main()