#!/usr/bin/env python
"""
This script is meant to gather GEOS CF input data files.
It reads the YAML file containing the collections of interest.
In each collection, we list the fields we want to read and save.
"""

import yaml
import os
import sys
import glob
from copy import deepcopy
from pathlib import Path
import datetime as dttm
import xarray as xr
import numpy as np
import pandas as pd

from ..shared.gen_utils import read_yaml_file
from ..shared.gen_utils import get_list_files
from ..shared.gen_utils import create_list_dates


def obtain_geos_cf_fields(yaml_file_name: str) -> dict():
    """
    Read data files from different GEOS CF colections to
    extract specific fields that are stored in a dictionary.

    Parameters
    ----------
    yaml_file_name : str
       YAML file containing parameter settings.

    Returns
    -------
    ds_dict : dict
       Dictionary which keys are the field names and the values
       are the associated NumPy arrays.
    """

    # Read the parameter settings from the YAML file

    params = read_yaml_file(yaml_file_name)
    if not params:
        print(f"Was not able to read the content of {file_name}")
        sys.exit()

    exp_name = params["exp_name"]
    beg_date = params["beg_date"]
    end_date = params["end_date"]

    print('-'*70)
    print(f"... Ready to read data files")
    print(f"...    Experiment name: {exp_name}")
    print(f"...         Date range: from {beg_date}  to {end_date}")
    print('-'*70)
    # List of individual Xarrays Datasets, each one associated with a collection
    list_ds = list()

    # Loop over all the collections to extract time series fields
    for key in params["collection"]:
        print(f"... Read data files for collection: {key}")

        # Get parameters associated with collection
        col_params = params['collection'][key]

        # Check if level_id is provided
        if "level_id" in col_params:
            level_id = col_params["level_id"]
        else:
            level_id = -1

        # Gather data for the collection
        ds = read_geos_cf_collection(
                data_dir = col_params["data_dir"], 
                file_prefix = col_params["file_prefix"], 
                fields = col_params["fields"],
                beg_date = beg_date, 
                end_date = end_date, 
                fields_map = col_params["fields_map"], 
                level_id = level_id)

        list_ds.append(ds)
        print(f"... Done with collection: {key}")
        print()

    # Merge the individual Datasets into a unique one
    new_ds = xr.merge(list_ds)

    # Create a dictionary where the keys are the field names and
    # their values are the corresponding NumPy arrays.
    keys_list = ["lat", "lon", "time"] + list(new_ds.keys())
    ds_dict = dict()
    ds_dict["exp_name"] = exp_name
    for key in keys_list:
        ds_dict[key] = new_ds[key].to_numpy()

    return ds_dict

def read_geos_cf_collection(data_dir: str, file_prefix: str, 
                    fields: list, fields_map: list, 
                    beg_date: int, end_date: int, 
                    level_id: int=-1) -> xr.Dataset:
    """
    Given a range of dates, read in time series fields within the range only.
    We extract the field values at the specified vertical level.
    If necessary, we rename the fields.

    Parameters
    ----------
    data_dir : str
       Full path to the directory where the data files reside.
    file_prefix : str
       Prefix of the file to be read.
    fields : list
       List of fields to be extracted for the files.
    fields_map : list
       List of field names to match what is expected in AQcGAN. 
       This list should have a one to one match with names in fields_map.
       If the names fields_map should not be changed, this list should be
       the same as fields_map.
    beg_date : int
       Start date in the format YYYYMMDD
    end_date : int
       End date in the format YYYYMMDD
    level_id : int
       Vertical level of interest (1 to 72 where 72 is the surface).
       This only apply if dealing with multiple level data files.

    Returns
    -------
    xr_ds : xr.Dataset
       Xarray Dataset containing a time series collection of fields at
       the specified level index.
    """

    # Obtain the list of files to read
    list_files = get_list_files(data_dir, file_prefix, beg_date, end_date)
    print(f"   ... {len(list_files)} files to read.")

    # Compute the the time parameters
    date_list = list()
    time_list = list()
    time_of_year_x = list()
    time_of_year_y = list()
    time_of_day_x = list()
    time_of_day_y = list()
    for file in list_files:
        date_info = extract_date_from_file_name(file)
        mydate = date_info.split("_")

        date_list.append(date_info)
        time_list.append(mydate[1])

        time_year_day = comp_time_year_day(mydate[0], int(mydate[1]))
        time_of_year_x.append(time_year_day[0])
        time_of_year_y.append(time_year_day[1])
        time_of_day_x.append(time_year_day[2])
        time_of_day_y.append(time_year_day[3])

    # Open the files as a Xarray Dataset
    ds = xr.open_mfdataset(list_files)

    # Only select the fields of interest
    ds = ds[fields]

    # Select data at the specified level
    if level_id > 0:
        ds = ds.isel(lev=level_id-1)

    # Make sure that the variables have names expected by the AQcGAN tool
    if level_id > 0: 
        lev = str(level_id).zfill(2)
    else:
        lev = None
    map_vars = dict()
    for i, vname in enumerate(fields):
        new_name = fields_map[i]
        if lev:
            new_name = f"{new_name}_{lev}"
        map_vars[vname] = new_name
    ds = ds.rename(map_vars)

    # Change time values and attributes
    date_list.sort()
    time_list = list(set(time_list))
    time_list.sort()
    nrecs_per_day = len(time_list)
    freq_hours = 24 // nrecs_per_day
    ds['time'] = create_list_dates(date_list[0], date_list[-1], freq_hours)
    ds['time'].attrs['begin_date'] = date_list[0].split("_")[0]
    ds['time'].attrs['begin_time'] = f'{date_list[0].split("_")[1]}00'
    ds['time'].attrs['long_name'] = 'time'
    ds['time'].attrs['time_increment'] = freq_hours*10000

    # Add new variables
    ds['time_of_year_x'] = xr.DataArray(np.array(time_of_year_x), dims='time')
    ds['time_of_year_y'] = xr.DataArray(np.array(time_of_year_y), dims='time')
    ds['time_of_day_x'] = xr.DataArray(np.array(time_of_day_x), dims='time')
    ds['time_of_day_y'] = xr.DataArray(np.array(time_of_day_y), dims='time')

    # Write the dimensions and associated values in a netCDF file
    # for future use.
    tmp_ds = ds[['lat', 'lon', 'time']]
    tmp_ds.to_netcdf(path="tmp_dimensions.nc4")

    return ds

def extract_date_from_file_name(file_name: str) -> str:
    """
    Given a file name like:
       chm_inst_1hr_glo_L1440x721_v72/CF2_control.chm_inst_1hr_glo_L1440x721_v72.20240102_2100z.nc4
    we return:
       20240102_2100
    that represent the date and time record for the file.
    """
    file_stem = Path(file_name).stem
    mydate = file_stem.split(".")[-1]
    mydate = mydate[:-1]  # remove the z character

    # We make this change baecause we want to ensure that
    # all the data files we read have whole hours (no minutes).
    if mydate.endswith('30'):
        mydate = mydate[:-2]+"00"

    return mydate

def comp_time_year_day(beg_date: int, beg_time: int)-> tuple():
    """
    """
    # map to 0 - 2pi and project to unit circle
    begin_date_dt = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d')
    begin_time_dt = dttm.datetime.strptime(f'{beg_time:06d}', '%H%M%S')

    if begin_date_dt.year % 4 == 0 and begin_date_dt.year % 100 != 0:
        total_days = 366
    else:
        total_days = 365

    # time of year (toy) and time of day (tod)
    toy_2pi = begin_date_dt.timetuple().tm_yday / total_days * np.pi * 2
    tod_2pi = begin_time_dt.timetuple().tm_hour / 24 * np.pi * 2

    toy_x = np.cos(toy_2pi)
    toy_y = np.sin(toy_2pi)
    tod_x = np.cos(tod_2pi)
    tod_y = np.sin(tod_2pi)

    return toy_x, toy_y, tod_x, tod_y

