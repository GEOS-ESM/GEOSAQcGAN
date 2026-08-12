#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot global lead-time mean truth, prediction, and RMSE maps for AQcGAN v2.2.0.

This script reads lead-time mean statistics produced by
compute_leadmean_global_maps_data_and_scales.py and generates three-panel
diagnostic figures (Truth, Prediction, and RMSE) for selected forecast leads.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_maps/{VARIABLE}_lead_mean_maps.npz

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_maps/plots/{VARIABLE}_lead{LEAD}_truth_pred_rmse.png

Example:
    python plot_leadmean_global_maps.py --var BCPHILIC --leads 24 120 240

Plot all 3-hour leads:
    python plot_leadmean_global_maps.py --var BCPHILIC --leads $(seq 3 3 240)
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for Discover/batch nodes

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Plot AQcGAN global lead-mean truth, prediction, and RMSE maps."
    )
    parser.add_argument("--var", required=True, help="Example: BCPHILIC, SO4, SS005")
    parser.add_argument("--leads", type=int, nargs="+", required=True)
    parser.add_argument(
        "--base",
        default="/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0",
    )
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    base = Path(args.base)
    in_file = base / "leadmean_global_maps" / f"{args.var}_lead_mean_maps.npz"
    out_dir = base / "leadmean_global_maps" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Missing file: {in_file}")

    data = np.load(in_file)

    leads = data["leads"]
    truth_map = data["truth_map"]
    pred_map = data["pred_map"]
    rmse_map = data["rmse_map"]
    units = str(data["units"]) if "units" in data else "kg/kg"

    nlat = truth_map.shape[1]
    nlon = truth_map.shape[2]

    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-180, 180, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon, lat)

    vmin_main = float(data["main_vmin"])
    vmax_main = float(data["main_vmax"])

    vmin_rmse = float(data["rmse_vmin"])
    vmax_rmse = float(data["rmse_vmax"])

    print(f"Loaded: {in_file}")
    print(f"truth_map shape: {truth_map.shape}")
    print(f"pred_map shape : {pred_map.shape}")
    print(f"rmse_map shape : {rmse_map.shape}")
    print(f"Truth/pred color scale: {vmin_main:.8e} to {vmax_main:.8e} {units}")
    print(f"RMSE color scale      : {vmin_rmse:.8e} to {vmax_rmse:.8e} {units}")

    for lead in args.leads:
        lead_idx_arr = np.where(leads == lead)[0]

        if len(lead_idx_arr) == 0:
            print(f"Skipping lead {lead}. Not found.")
            continue

        lead_idx = int(lead_idx_arr[0])

        datasets = [
            truth_map[lead_idx],
            pred_map[lead_idx],
            rmse_map[lead_idx],
        ]

        titles = ["Truth", "Prediction", "RMSE"]
        cmaps = ["RdPu", "RdPu", "Reds"]
        vmins = [vmin_main, vmin_main, vmin_rmse]
        vmaxs = [vmax_main, vmax_main, vmax_rmse]

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.35])

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ]

        for ax, plot_data, title, cmap, vmin, vmax in zip(
            axes, datasets, titles, cmaps, vmins, vmaxs
        ):
            m = Basemap(projection="robin", lon_0=0, resolution="c", ax=ax)

            cs = m.pcolormesh(
                lon2d,
                lat2d,
                plot_data,
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
                linewidth=0.5,
                dashes=[2, 2],
            )
            m.drawparallels(
                np.arange(-90, 91, 30),
                color="gray",
                linewidth=0.5,
                dashes=[2, 2],
            )

            cbar = fig.colorbar(cs, ax=ax, shrink=0.75, pad=0.04)
            cbar.set_label(units)
            cbar.ax.tick_params(labelsize=8)

            ax.set_title(title, fontsize=15)

        plt.suptitle(
            f"AQcGAN v2.2.0 Aerosol Forecast\n"
            f"Lead Time = {lead} hours | Variable = {args.var}",
            fontsize=18,
            y=0.96,
        )

        plt.tight_layout()

        out_png = out_dir / f"{args.var}_lead{lead:03d}_truth_pred_rmse.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_png}")

        if args.show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    main()