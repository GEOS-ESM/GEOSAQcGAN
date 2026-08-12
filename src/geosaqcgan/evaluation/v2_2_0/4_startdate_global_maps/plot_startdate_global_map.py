#!/usr/bin/env python3

"""
Plot AQcGAN v2.2.0 global maps for a single forecast initialization.

This script reads AQcGAN prediction and truth files for a selected lead day,
aerosol variable, forecast initialization time, and 3-hour output frame. It
extracts the truth, prediction, and absolute-error maps and generates a
three-panel global diagnostic figure.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
        test_ens_pred_stats_days{DAY}_chkpt100.npz
        test_ens_stats_days{DAY}.npz

The script uses fixed color scales from:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/startdate_global_map_scales.npz

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/startdate_global_maps/
        {VARIABLE}_day{DAY}_{START_TIME}_{VALID_TIME}_maps.npz
        {VARIABLE}_day{DAY}_{START_TIME}_{VALID_TIME}_maps.png

Example:
    python plot_startdate_global_map.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for Discover/batch nodes

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except ImportError:
    HAS_BASEMAP = False


SPECIES = [
    "BCPHILIC", "BCPHOBIC", "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a", "NO3an1", "NO3an2", "NO3an3", "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]

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
    "SS001": "Sea Salt Mixing Ratio bin 001",
    "SS002": "Sea Salt Mixing Ratio bin 002",
    "SS003": "Sea Salt Mixing Ratio bin 003",
    "SS004": "Sea Salt Mixing Ratio bin 004",
    "SS005": "Sea Salt Mixing Ratio bin 005",
}

BASE_DIR = Path("/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0")
OUT_DIR = BASE_DIR / "startdate_global_maps"
SPLIT_START = datetime(2018, 6, 1, 0)
CHKPT_IDX = 100
UNITS = "kg/kg"


def parse_time(text):
    return datetime.strptime(text.strip(), "%Y-%m-%d %H")


def first_array(npz):
    key = list(npz.keys())[0]
    return key, npz[key]


def get_color_scales(variable):
    scale_file = BASE_DIR / "plot_metadata" / "startdate_global_map_scales.npz"

    if not scale_file.exists():
        raise FileNotFoundError(
            f"Missing color scale file: {scale_file}\n"
            "Run compute_startdate_global_map_scales.py first."
        )

    with np.load(scale_file) as scales:
        scale_species = list(scales["species"].astype(str))
        if variable not in scale_species:
            raise ValueError(f"{variable} not found in {scale_file}")

        scale_idx = scale_species.index(variable)

        vmin_main = float(scales["main_vmin"][scale_idx])
        vmax_main = float(scales["main_vmax"][scale_idx])
        vmin_ae = float(scales["ae_vmin"][scale_idx])
        vmax_ae = float(scales["ae_vmax"][scale_idx])

    return vmin_main, vmax_main, vmin_ae, vmax_ae


def plot_panel(ax, lon2d, lat2d, data, title, cmap, vmin, vmax, units):
    if HAS_BASEMAP:
        m = Basemap(projection="robin", lon_0=0, resolution="c", ax=ax)
        cs = m.pcolormesh(
            lon2d,
            lat2d,
            data,
            latlon=True,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        m.drawcoastlines(linewidth=0.6, color="black")
        m.drawcountries(linewidth=0.3, color="black")
        m.drawmeridians(
            np.arange(-180, 181, 60),
            color="gray",
            linewidth=0.4,
            dashes=[2, 2],
        )
        m.drawparallels(
            np.arange(-90, 91, 30),
            color="gray",
            linewidth=0.4,
            dashes=[2, 2],
        )
    else:
        cs = ax.pcolormesh(
            lon2d,
            lat2d,
            data,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.set_title(title, fontsize=13)
    cbar = plt.colorbar(cs, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label(units)
    cbar.ax.tick_params(labelsize=8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lead_day = int(input("Lead day, 1-10: ").strip())

    variable = input("Variable/species, e.g. BCPHILIC: ").strip()
    if variable not in SPECIES:
        raise ValueError(f"{variable} not found. Valid options: {', '.join(SPECIES)}")

    start_dt = parse_time(input("Forecast start date/time, e.g. 2018-06-01 00: "))

    valid_times = [
        start_dt + timedelta(days=lead_day, hours=3 * i)
        for i in range(8)
    ]

    print()
    print("Available output frames for this forecast:")
    for i, t in enumerate(valid_times):
        print(f"  {i}: {t:%Y-%m-%d %HZ}")

    frame_idx = int(input("Choose output frame index, 0-7: ").strip())

    if frame_idx < 0 or frame_idx >= len(valid_times):
        raise ValueError("Output frame index must be between 0 and 7")

    valid_time = valid_times[frame_idx]
    frame_offset_hours = 3 * frame_idx

    pred_file = BASE_DIR / f"test_ens_pred_stats_days{lead_day}_chkpt{CHKPT_IDX}.npz"
    truth_file = BASE_DIR / f"test_ens_stats_days{lead_day}.npz"

    if not pred_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")
    if not truth_file.exists():
        raise FileNotFoundError(f"Missing truth file: {truth_file}")

    case_idx = int((start_dt - SPLIT_START).total_seconds() // 3600)
    var_idx = SPECIES.index(variable)

    with np.load(pred_file) as pred_npz:
        pred_key, pred = first_array(pred_npz)

    with np.load(truth_file) as truth_npz:
        truth_key, truth = first_array(truth_npz)

    if pred.shape != truth.shape:
        raise ValueError(f"Prediction and truth shapes differ: {pred.shape} vs {truth.shape}")

    if case_idx < 0 or case_idx >= pred.shape[0]:
        last_start = SPLIT_START + timedelta(hours=pred.shape[0] - 1)
        raise IndexError(
            f"Start {start_dt:%Y-%m-%d %HZ} maps to case index {case_idx}, "
            f"but day-{lead_day} allows starts from "
            f"{SPLIT_START:%Y-%m-%d %HZ} to {last_start:%Y-%m-%d %HZ}."
        )

    pred_map = pred[case_idx, var_idx, frame_idx]
    truth_map = truth[case_idx, var_idx, frame_idx]
    ae_map = np.abs(pred_map - truth_map)

    nlat, nlon = pred_map.shape
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon, lat)

    vmin_main, vmax_main, vmin_ae, vmax_ae = get_color_scales(variable)

    stem = (
        f"{variable}_day{lead_day}_{start_dt:%Y%m%d_%H%M}_"
        f"valid{valid_time:%Y%m%d_%H%M}_maps"
    )
    npz_file = OUT_DIR / f"{stem}.npz"
    png_file = OUT_DIR / f"{stem}.png"

    np.savez_compressed(
        npz_file,
        variable=variable,
        description=DESCRIPTIONS[variable],
        units=UNITS,
        lead_day=lead_day,
        forecast_start=start_dt.strftime("%Y-%m-%d %HZ"),
        valid_time=valid_time.strftime("%Y-%m-%d %HZ"),
        case_idx=case_idx,
        frame_idx=frame_idx,
        frame_offset_hours=frame_offset_hours,
        pred_key=pred_key,
        truth_key=truth_key,
        color_scale_main_vmin=vmin_main,
        color_scale_main_vmax=vmax_main,
        color_scale_ae_vmin=vmin_ae,
        color_scale_ae_vmax=vmax_ae,
        lat=lat,
        lon=lon,
        pred_map=pred_map.astype(np.float32),
        truth_map=truth_map.astype(np.float32),
        ae_map=ae_map.astype(np.float32),
    )

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.25])

    ax_truth = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_ae = fig.add_subplot(gs[1, :])

    plot_panel(ax_truth, lon2d, lat2d, truth_map, "Truth", "RdPu", vmin_main, vmax_main, UNITS)
    plot_panel(ax_pred, lon2d, lat2d, pred_map, "Prediction", "RdPu", vmin_main, vmax_main, UNITS)
    plot_panel(ax_ae, lon2d, lat2d, ae_map, "Absolute Error", "Reds", vmin_ae, vmax_ae, UNITS)

    fig.suptitle(
        f"AQcGAN v2.2.0 aerosol forecast | {variable} ({DESCRIPTIONS[variable]})\n"
        f"Initialized {start_dt:%Y-%m-%d %HZ} | Lead day {lead_day} | "
        f"Valid {valid_time:%Y-%m-%d %HZ} | Units: {UNITS}",
        fontsize=16,
        y=0.97,
    )

    plt.tight_layout()
    plt.savefig(png_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved map data: {npz_file}")
    print(f"Saved plot: {png_file}")
    print(f"Main color scale: {vmin_main:.8e} to {vmax_main:.8e} {UNITS}")
    print(f"Absolute error color scale: {vmin_ae:.8e} to {vmax_ae:.8e} {UNITS}")


if __name__ == "__main__":
    main()