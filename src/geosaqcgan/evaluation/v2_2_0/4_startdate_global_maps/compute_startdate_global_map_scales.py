#!/usr/bin/env python3

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
    plot_startdate_global_map.py

This script computes plotting scales only and does not create figures.
"""

from pathlib import Path

import numpy as np


SPECIES = [
    "BCPHILIC", "BCPHOBIC", "OCPHILIC", "OCPHOBIC",
    "DU001", "DU002", "DU003", "DU004", "DU005",
    "NH4a", "NO3an1", "NO3an2", "NO3an3", "SO4",
    "SS001", "SS002", "SS003", "SS004", "SS005",
]

BASE_DIR = Path("/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0")
OUT_DIR = BASE_DIR / "plot_metadata"
OUT_FILE = OUT_DIR / "startdate_global_map_scales.npz"

CHKPT_IDX = 100
START_DAY = 1
END_DAY = 10
CHUNK_SIZE = 4


def first_array(npz):
    key = list(npz.keys())[0]
    return key, npz[key]


def save_scales(main_vmin, main_vmax, ae_vmin, ae_vmax, counts_by_day):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_FILE,
        species=np.array(SPECIES),
        units="kg/kg",
        main_vmin=main_vmin.astype(np.float32),
        main_vmax=main_vmax.astype(np.float32),
        ae_vmin=ae_vmin.astype(np.float32),
        ae_vmax=ae_vmax.astype(np.float32),
        counts_by_day=np.array(counts_by_day, dtype=np.int32),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_species = len(SPECIES)

    main_vmin = np.full(n_species, np.inf, dtype=np.float64)
    main_vmax = np.full(n_species, -np.inf, dtype=np.float64)

    ae_vmin = np.zeros(n_species, dtype=np.float64)
    ae_vmax = np.full(n_species, -np.inf, dtype=np.float64)

    counts_by_day = []

    for lead_day in range(START_DAY, END_DAY + 1):
        pred_file = BASE_DIR / f"test_ens_pred_stats_days{lead_day}_chkpt{CHKPT_IDX}.npz"
        truth_file = BASE_DIR / f"test_ens_stats_days{lead_day}.npz"

        if not pred_file.exists() or not truth_file.exists():
            print(f"Missing lead day {lead_day}, skipping.", flush=True)
            continue

        print("=" * 80, flush=True)
        print(f"Processing lead day {lead_day}", flush=True)
        print(f"Prediction: {pred_file}", flush=True)
        print(f"Truth     : {truth_file}", flush=True)

        with np.load(pred_file) as pred_npz, np.load(truth_file) as truth_npz:
            pred_key, pred = first_array(pred_npz)
            truth_key, truth = first_array(truth_npz)

            if pred.shape != truth.shape:
                raise ValueError(
                    f"Shape mismatch for day {lead_day}: {pred.shape} vs {truth.shape}"
                )

            n_cases = pred.shape[0]
            counts_by_day.append((lead_day, n_cases))

            print(f"Shape: {pred.shape}", flush=True)

            for var_idx, var_name in enumerate(SPECIES):
                print(f"  {var_idx:02d} {var_name}", flush=True)

                for start in range(0, n_cases, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_cases)

                    pred_chunk = pred[start:end, var_idx]
                    truth_chunk = truth[start:end, var_idx]

                    main_vmin[var_idx] = min(
                        main_vmin[var_idx],
                        float(np.nanmin(pred_chunk)),
                        float(np.nanmin(truth_chunk)),
                    )
                    main_vmax[var_idx] = max(
                        main_vmax[var_idx],
                        float(np.nanmax(pred_chunk)),
                        float(np.nanmax(truth_chunk)),
                    )

                    ae_chunk = np.abs(pred_chunk - truth_chunk)

                    ae_vmax[var_idx] = max(
                        ae_vmax[var_idx],
                        float(np.nanmax(ae_chunk)),
                    )

                    del pred_chunk, truth_chunk, ae_chunk

        save_scales(main_vmin, main_vmax, ae_vmin, ae_vmax, counts_by_day)
        print(f"Checkpoint saved: {OUT_FILE}", flush=True)

    save_scales(main_vmin, main_vmax, ae_vmin, ae_vmax, counts_by_day)

    print()
    print(f"Saved: {OUT_FILE}", flush=True)
    print("variable,main_vmin,main_vmax,ae_vmin,ae_vmax", flush=True)

    with np.load(OUT_FILE) as data:
        for i, sp in enumerate(data["species"].astype(str)):
            print(
                f"{sp},"
                f"{data['main_vmin'][i]:.8e},"
                f"{data['main_vmax'][i]:.8e},"
                f"{data['ae_vmin'][i]:.8e},"
                f"{data['ae_vmax'][i]:.8e}",
                flush=True,
            )


if __name__ == "__main__":
    main()