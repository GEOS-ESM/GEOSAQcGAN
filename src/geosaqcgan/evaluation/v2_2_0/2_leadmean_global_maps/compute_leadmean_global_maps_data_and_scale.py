#!/usr/bin/env python3

"""
Compute global lead-time mean map data and color scales for AQcGAN v2.2.0.

This script reads prediction and truth files for lead days 1–10, averages
across forecast samples, and computes prediction, truth, mean absolute error,
and RMSE maps for each aerosol variable.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/
        test_ens_pred_stats_days{DAY}_chkpt100.npz
        test_ens_stats_days{DAY}.npz

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/leadmean_global_maps/{VARIABLE}_lead_mean_maps.npz

Each output contains 80 global maps for 3-hour leads from 3 to 240 hours.
The processed files are used by plot_leadmean_global_maps.py.

Example:
    python compute_leadmean_global_maps_data_and_scales.py
"""

import argparse
from pathlib import Path

import numpy as np


VAR_NAMES = [
    "BCPHILIC", "BCPHOBIC",
    "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a",
    "NO3an1", "NO3an2", "NO3an3",
    "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]


def main():
    parser = argparse.ArgumentParser(
        description="Create global lead-mean map data for AQcGAN predictions, truth, AE, and RMSE."
    )
    parser.add_argument(
        "--base",
        default="/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0",
        help="Directory containing test_ens_pred_stats_days*.npz and test_ens_stats_days*.npz.",
    )
    parser.add_argument("--chkpt", type=int, default=100)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=10)

    args = parser.parse_args()

    base = Path(args.base)
    out_dir = base / "leadmean_global_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    leads = np.arange(3, 241, 3)

    for var_idx, var_name in enumerate(VAR_NAMES):
        print("=" * 80, flush=True)
        print(f"Processing variable {var_idx}: {var_name}", flush=True)

        pred_maps_by_day = []
        truth_maps_by_day = []
        ae_maps_by_day = []
        rmse_maps_by_day = []
        counts_by_day = []

        for day in range(args.start_day, args.end_day + 1):
            print(f"{var_name}: loading lead day {day}...", flush=True)

            pred_file = base / f"test_ens_pred_stats_days{day}_chkpt{args.chkpt}.npz"
            truth_file = base / f"test_ens_stats_days{day}.npz"

            if not pred_file.exists() or not truth_file.exists():
                print(f"Missing day {day}, skipping.", flush=True)
                continue

            with np.load(pred_file) as pnpz, np.load(truth_file) as tnpz:
                pred = pnpz["ens_mean_pred_test_inv"][:, var_idx]
                truth = tnpz["ens_mean_test_inv"][:, var_idx]

                print(f"  raw shape after var select: {pred.shape}", flush=True)

                pred_mean = np.nanmean(pred, axis=0)
                truth_mean = np.nanmean(truth, axis=0)
                ae_mean = np.nanmean(np.abs(pred - truth), axis=0)
                rmse = np.sqrt(np.nanmean((pred - truth) ** 2, axis=0))

                pred_maps_by_day.append(pred_mean.astype(np.float32))
                truth_maps_by_day.append(truth_mean.astype(np.float32))
                ae_maps_by_day.append(ae_mean.astype(np.float32))
                rmse_maps_by_day.append(rmse.astype(np.float32))
                counts_by_day.append(pred.shape[0])

            del pred, truth

        if not pred_maps_by_day:
            print(f"No data found for {var_name}, skipping.", flush=True)
            continue

        pred_map = np.concatenate(pred_maps_by_day, axis=0)
        truth_map = np.concatenate(truth_maps_by_day, axis=0)
        ae_map = np.concatenate(ae_maps_by_day, axis=0)
        rmse_map = np.concatenate(rmse_maps_by_day, axis=0)

        if pred_map.shape[0] != len(leads):
            raise ValueError(
                f"Lead mismatch for {var_name}: got {pred_map.shape[0]}, expected {len(leads)}"
            )

        main_vmin = np.nanmin([np.nanmin(truth_map), np.nanmin(pred_map)])
        main_vmax = np.nanmax([np.nanmax(truth_map), np.nanmax(pred_map)])

        ae_vmin = 0.0
        ae_vmax = np.nanmax(ae_map)

        rmse_vmin = 0.0
        rmse_vmax = np.nanmax(rmse_map)

        out_file = out_dir / f"{var_name}_lead_mean_maps.npz"

        np.savez_compressed(
            out_file,
            leads=leads,
            var_idx=var_idx,
            var_name=var_name,
            units="kg/kg",
            pred_map=pred_map,
            truth_map=truth_map,
            ae_map=ae_map,
            rmse_map=rmse_map,
            counts_by_day=np.array(counts_by_day),
            main_vmin=np.float32(main_vmin),
            main_vmax=np.float32(main_vmax),
            ae_vmin=np.float32(ae_vmin),
            ae_vmax=np.float32(ae_vmax),
            rmse_vmin=np.float32(rmse_vmin),
            rmse_vmax=np.float32(rmse_vmax),
        )

        print(f"Saved: {out_file}", flush=True)
        print(f"Final map shape: {pred_map.shape}", flush=True)
        print(f"Truth/pred color scale: {main_vmin:.8e} to {main_vmax:.8e}", flush=True)
        print(f"AE color scale        : {ae_vmin:.8e} to {ae_vmax:.8e}", flush=True)
        print(f"RMSE color scale      : {rmse_vmin:.8e} to {rmse_vmax:.8e}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()