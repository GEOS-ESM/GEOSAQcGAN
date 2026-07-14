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
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from read_geos_cf_datafiles import obtain_geos_cf_fields

COORDS = ["lon", "lat", "lev", "time", "time_of_year_x", "time_of_year_y", "time_of_day_x", "time_of_day_y"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data directory for experiment.")
    parser.add_argument("--geos_cf_yaml_file", type=str, help="Full path to the YAML file containing setting parameters.")
    parser.add_argument("--aqcgan_config_file", type=str, help="Full path to the YAML file containing AQcGAN expt settings")
    parser.add_argument("--split", type=str, help="split (test or val)")
    parser.add_argument("--level", type=int, help="vertical level")

    args = parser.parse_args()

    print("Reading aqcgan config file...")
    with open(args.aqcgan_config_file, "r") as file:
        aqcgan_config = yaml.safe_load(file)

    print("Reading in data...")
    m_dict = obtain_geos_cf_fields(args.geos_cf_yaml_file)

    exp_name = m_dict["exp_name"]
    del m_dict["exp_name"]

    beg_date = pd.Timestamp(m_dict["time"][0]).strftime("%Y%m%d_%Hz")
    end_date = pd.Timestamp(m_dict["time"][-1]).strftime("%Y%m%d_%Hz")
    time_init  = m_dict["time"].copy()
    del m_dict["time"]

    lat = m_dict["lat"].copy()
    lon = m_dict["lon"].copy()

    z_vars = [k for k in sorted(m_dict.keys()) if k not in COORDS]
    
    # Make sure they match the feature names
    feat_names = aqcgan_config["data"]["feat_names"][args.level]
    if set(feat_names) == set(z_vars):
        z_vars = feat_names 
    time_vars = COORDS[-4:]

    # member variables and time data (time of year, time of day)
    m_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in z_vars], axis=0)
    time_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in time_vars], axis=0)

    # save variables and time data in val folder
    split_dir = Path(args.data_dir) / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"{exp_name}.{beg_date}-{end_date}.fields.npy", "wb") as fid:
        np.save(fid, m_array)

    print("Saved member data array.")
    with open(split_dir / f"{exp_name}.{beg_date}-{end_date}.time.npy", "wb") as fid:
        np.save(fid, time_array)
    print("Saved time data array.")    

    # save off metadata (lat, lon, z_mean, z_std, z_vars)
    meta_save_dict = {
            "lat": lat, 
            "lon": lon, 
            "time": time_init,
            "z_vars": z_vars, 
            "time_vars": time_vars,
    }
    with open(Path(args.data_dir) / "meta.pkl", "wb") as f:
        pickle.dump(meta_save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved metadata dict.")

        
    # Rearrange norm stats based on features specified in the config file
    if z_vars != feat_names:
        # Norm stats may have more variables than in the new GEOS data
        print("Overwrite norm_stats") 
        norm_stats_file = Path(args.data_dir) / "norm_stats.pkl"
        
        with open(norm_stats_file, "r") as file:
            norm_stats = np.load(norm_stats_file, allow_pickle=True)    
    
        indices = [feat_names.index(var) for var in z_vars]
        new_norm_stats = {}
        for key, values in norm_stats.items():
            if key == "n_timesteps":
                new_norm_stats[key] = time_array.shape[1]
            elif key == "boxcox_lambda":
                new_norm_stats[key] = norm_stats[key]
            else:
                if isinstance(values, list):
                    new_norm_stats[key] = [values[i] for i in indices]
                else:
                    new_norm_stats[key] = values[indices]

        
        with open(norm_stats_file, "wb") as fid:
            pickle.dump(new_norm_stats, fid, protocol=pickle.HIGHEST_PROTOCOL)
