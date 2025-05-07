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

from read_geos_cf_datafiles import obtain_geos_cf_fields

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
    with open(m_data, "rb") as f:
        m_dict = pickle.load(f)

    return m_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #KNJR parser.add_argument("perturb_type", help="Perturbation type.", choices=["emis_only", "met_only", "met_and_emis"])
    parser.add_argument("data_category", help="Category of data to merge.", choices=["bihlo", "gcc"])
    parser.add_argument("data_file", type=str, help="Name of data pkl file.")
    parser.add_argument("norm_stats_file", type=str, help="Name of norm stats file.")
    parser.add_argument("member_idx", type=int, help="Member idx to preprocess.")
    parser.add_argument("exp_dir", type=str, help="Data directory for experiment.")
    parser.add_argument("geos_cf_yaml_fname", type=str, help="Full path to the YAML file containing setting parameters.")
    parser.add_argument("--train_members", type=int, nargs="+", help="Indices for training ensemble members.")
    parser.add_argument("--val_members", type=int, nargs="+", help="Indices for validation ensemble members.")
    parser.add_argument("--test_members", type=int, nargs="+", help="Indices for test ensemble members.")
    parser.add_argument("--test_date_start_idx", "-t_start", type=int, help="Test set start date.")
    parser.add_argument("--test_date_end_idx", "-t_end", type=int, help="Test set end date.")
    parser.add_argument("--log_transform", "-log", action="store_true", help="Apply log transformation before normalization.")
    #KNJR parser.add_argument("--root_dir", help="Root data directory with perturbations as subdirectories.", required=True)

    args = parser.parse_args()

    #KNJR root_dir = Path(args.root_dir)
    #KNJR perturb_dir = root_dir / args.perturb_type
    #KNJR m_dir = sorted(list(perturb_dir.glob("mem*/")))[args.member_idx-1]

    print("Reading in data...")
    m_dict = obtain_geos_cf_fields(args.geos_cf_yaml_fname)
    #KNJR m_dict = load_member_data(m_dir, args.data_file)

    exp_name = m_dict["exp_name"]
    del m_dict["exp_name"]

    beg_date = np.datetime_as_string(m_dict["time"][0], unit="D")
    end_date = np.datetime_as_string(m_dict["time"][-1], unit="D")

    lat = m_dict["lat"].copy()
    lon = m_dict["lon"].copy()

    print("Normalizing and reshaping data...")
    with open(args.norm_stats_file, "rb") as f:
        norm_stats = pickle.load(f)

    z_mean = norm_stats["z_mean"]
    z_std = norm_stats["z_std"]
    for i, k in enumerate(norm_stats['variables']):
        if k in m_dict:
            m_dict[k] = (m_dict[k] - z_mean[i]) / z_std[i]

    # make train/test array
    # member variables and time data (time of year, time of day)
    m_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in sorted(m_dict.keys()) if k not in DO_NOT_NORMALIZE], axis=0)
    time_array = np.stack( [m_dict[k].data if isinstance(m_dict[k], np.ma.MaskedArray) else m_dict[k] for k in DO_NOT_NORMALIZE[-4:]], axis=0)

    # train/val set
    if args.member_idx in args.train_members or args.member_idx in args.val_members:
        if args.member_idx in args.train_members:
            split = "train"
        else:
            split = "val"

        # if splitting train/test based on time as well
        # e.g. train on jan -> may, sep -> dec, test on jun -> aug
        #KNJR if args.test_date_start_idx and args.test_date_end_idx:
        #KNJR     m_array = np.concatenate((m_array[:, :args.test_date_start_idx, :, :],
        #KNJR         m_array[:, args.test_date_end_idx+1:, :, :]), axis=1)
        #KNJR     time_array = np.concatenate((time_array[:, :args.test_date_start_idx],
        #KNJR         time_array[:, args.test_date_end_idx+1:]), axis=1)

    # test set
    elif args.member_idx in args.test_members:
        split = "test"

        # if splitting train/test based on time as well
        # e.g. train on jan -> may, sep -> dec, test on jun -> aug
        if args.test_date_start_idx and args.test_date_end_idx:
            m_array = m_array[:, args.test_date_start_idx:args.test_date_end_idx+1, :, :]
            time_array = time_array[:, args.test_date_start_idx:args.test_date_end_idx+1]

    # save off metadata (lat, lon, z_mean, z_std, z_vars)
    meta_save_dict = {
            "lat": lat, 
            "lon": lon, 
            "z_mean": z_mean,
            "z_std": z_std, 
            "z_vars": norm_stats['variables'], 
            "time_vars": DO_NOT_NORMALIZE[-4:]
    }
    #KNJR with open(Path(args.exp_dir) / f"meta.pkl", "wb") as f:
    with open(Path(args.exp_dir) / f"{exp_name}_{beg_date}_{end_date}_meta.pkl", "wb") as fid:
        pickle.dump(meta_save_dict, fid, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved metadata dict.")

    # save train/test array for variables and time data
    split_dir = Path(args.exp_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    #KNJR (Path(args.exp_dir) / split).mkdir(parents=True, exist_ok=True)
    #KNJR with open(Path(args.exp_dir) / split / f"{args.member_idx}.npy", "wb") as f:
    with open(split_dir / f"{exp_name}_{beg_date}_{end_date}.npy", "wb") as fid:
        np.save(fid, m_array)
    print("Saved member data array.")
    #KNJR with open(Path(args.exp_dir) / split / f"{args.member_idx}_time.npy", "wb") as f:
    with open(split_dir / f"{exp_name}_{beg_date}_{end_date}_time.npy", "wb") as fid:
        np.save(fid, time_array)
    print("Saved time data array.")


