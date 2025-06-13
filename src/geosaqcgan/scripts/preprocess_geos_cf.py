# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC.
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

"""
This script will take already merged data across time from pickle files for a given ensemble member 
and turn it into a numpy file where the data has been normalized using the already calculated
mean and standard deviation
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from .read_geos_cf_datafiles import obtain_geos_cf_fields
from ..shared.gen_utils import read_pickle_file

DO_NOT_NORMALIZE = ["lon", "lat", "lev", "time_of_year_x", "time_of_year_y", "time_of_day_x", "time_of_day_y"]

def load_member_data(m_dir: Path, filename: str) -> dict[str, np.ndarray]:
    """
    Loads ensemble member data

    Parameters:
        m_dir (Path): Directory storing the data for the ensemble member
        filename (str): name of the pkl file

    Returns:
        m_dict (dict): Dictionary containing the ensemble member's data
    """
    data_dir = m_dir
    m_data = data_dir / f"{filename}.pkl"
    m_dict = read_pickle_file(m_data)

    return m_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_stats_file", type=str, help="Name of norm stats file.")
    parser.add_argument("--exp_dir", type=str, help="Data directory for experiment.")
    parser.add_argument("--geos_cf_yaml_file", type=str, help="Full path to the YAML file containing setting parameters.")
    parser.add_argument("--validation_file", action="store_true", help="Whether to write validation file or not")

    args = parser.parse_args()

    print("Reading in data...")
    m_dict = obtain_geos_cf_fields(args.geos_cf_yaml_file)

    if ( args.validation_file ):
        # Save data for validation first
        vars = [var for var in m_dict if 'SpeciesConc' in var]
        val_ds = xr.Dataset(
            data_vars={
                var_name: (['time','lat','lon'], m_dict[var_name][:,:,:]) 
                for i, var_name in enumerate(vars)
                },
            coords= {
                'time': m_dict['time'],
                'lat': m_dict['lat'],
                'lon': m_dict['lon'],
                }
            )
    else:
        val_ds = None

    exp_name = m_dict["exp_name"]
    del m_dict["exp_name"]

    beg_date = pd.Timestamp(m_dict["time"][0]).strftime("%Y%m%d_%Hz")
    end_date = pd.Timestamp(m_dict["time"][-1]).strftime("%Y%m%d_%Hz")
    time_init  = m_dict["time"].copy()
    del m_dict["time"]

    lat = m_dict["lat"].copy()
    lon = m_dict["lon"].copy()

    print("Normalizing and reshaping data...")
    norm_stats = np.load(args.norm_stats_file,allow_pickle=True)

    z_mean = norm_stats["z_mean"]
    z_std = norm_stats["z_std"]

    z_mean_sub = []; z_std_sub = []

    for i, k in enumerate(norm_stats['variables']):
        if k in m_dict:
            m_dict[k] = (m_dict[k] - z_mean[i]) / z_std[i]
            z_mean_sub.append(z_mean[i])
            z_std_sub.append(z_std[i])

    # make train/test array
    # member variables and time data (time of year, time of day)
    m_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in sorted(m_dict.keys()) if k not in DO_NOT_NORMALIZE], axis=0)
    time_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in DO_NOT_NORMALIZE[-4:]], axis=0)

    split = "val"

    # save off metadata (lat, lon, z_mean, z_std, z_vars)
    meta_save_dict = {
            "lat": lat, 
            "lon": lon,
            "time_init": time_init,
            "exp_name": exp_name,
            "z_mean": z_mean_sub,
            "z_std": z_std_sub,
            "z_vars": [k for k in sorted(m_dict.keys()) if k not in DO_NOT_NORMALIZE], 
            "time_vars": DO_NOT_NORMALIZE[-4:]
    }
    with open(Path(args.exp_dir) / f"{exp_name}.{beg_date}-{end_date}.meta.pkl", "wb") as fid:
        pickle.dump(meta_save_dict, fid, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved metadata dict.")

    # save train/test array for variables and time data
    split_dir = Path(args.exp_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"{exp_name}.{beg_date}-{end_date}.fields.npy", "wb") as fid:
        np.save(fid, m_array)

    print("Saved member data array.")
    with open(split_dir / f"{exp_name}.{beg_date}-{end_date}.time.npy", "wb") as fid:
        np.save(fid, time_array)
    print("Saved time data array.")


    norm_stats['n_timesteps'] = time_array.shape[1]
    with open(args.norm_stats_file, "wb") as fid:
        pickle.dump(norm_stats, fid, protocol=pickle.HIGHEST_PROTOCOL)
    print("Overwrite norm_stats n_timesteps.") 

    if ( args.validation_file ):
        # Write netcdf for validation data
        val_file = f"{split_dir}/{exp_name}.{beg_date}-{end_date}.val.nc4"
        if ( val_ds is not None ):
            val_ds.to_netcdf(val_file)
        print("Saved validation data file") 
