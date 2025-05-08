#!/usr/bin/env python
"""
This script is meant to gather GCC input data files.
It reads the YAML file settings_input_dir.yaml and
creates a directory structure containing requested files
(over the provided date range). It is important to note
that we do not copy files but use symbolic links.

To run this script, first edit the settings_input_dir.yaml file
to provide the full path to your local input dir (my_input_dir),
select the date range (beg_date and end_date) and provide the
list of indices of interest (list_member_ids)
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
    root_dir = params["root_dir"]
    beg_date = params["beg_date"]
    end_date = params["end_date"]

    # List of individual Xarrays Datasets, each one associated with a collection
    list_ds = list()

    # Loop over all the collections to extract time series fields
    for key in params["collection"]:
        print(f"... Read data files for collection: {key}")
        # Full path to the location of the data files
        data_dir = f'{root_dir}/{exp_name}/holding/{key}'

        # Get parameters associated with collection
        col_params = params['collection'][key]
        if "level_id" in col_params:
            level_id = col_params["level_id"]
        else:
            level_id = -1

        # Gather data for the collection
        ds = read_geos_cf_collection(
                data_dir = data_dir, 
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

def read_yaml_file(file_name) -> dict():
    """
    Read a YAML file and returns it content as a dictionary.
    """
    try:
        with open(file_name, 'r') as fid:
            return yaml.safe_load(fid)
    except FileNotFoundError:
        print(f"Error: File not found: {file_name}")
        return None
    except yaml.YAMLError as e:
         print(f"Error parsing YAML file: {e}")
         return None
    else:
        print(f"Successfully read the file: \n {file_name}")
        print()

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

    date_list = list()
    time_list = list()
    time_of_year_x = list()
    time_of_year_y = list()
    time_of_day_x = list()
    time_of_day_y = list()
    for file in list_files:
        date_info = extract_date_from_file_name(file)

        date_list.append(date_info[0])
        time_list.append(date_info[1])

        time_year_day = comp_time_year_day(date_info[0], date_info[1])
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
    ds['time'].attrs['begin_date'] = date_list[0]
    ds['time'].attrs['begin_time'] = str(time_list[0]).ljust(4, '0')
    ds['time'].attrs['long_name'] = 'time'
    ds['time'].attrs['time_increment'] = freq_hours*10000

    # Add new variables
    ds['time_of_year_x'] = xr.DataArray(np.array(time_of_year_x), dims='time')
    ds['time_of_year_y'] = xr.DataArray(np.array(time_of_year_y), dims='time')
    ds['time_of_day_x'] = xr.DataArray(np.array(time_of_day_x), dims='time')
    ds['time_of_day_y'] = xr.DataArray(np.array(time_of_day_y), dims='time')

    return ds


def get_list_files(data_dir: str, file_prefix: str, beg_date: int, end_date: int) -> list:
    """
    """

    # Convert the dates from string into datetime object
    beg_date = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d')
    end_date = dttm.datetime.strptime(f'{end_date}', '%Y%m%d')

    freq_dt = dttm.timedelta(days=1)

    list_files = list()
    cur_date = beg_date
    while cur_date <= end_date:
        files = f"{data_dir}/{file_prefix}.{cur_date.strftime('%Y%m%d')}*.nc4"
        list_files += sorted(glob.glob(files))
        cur_date += freq_dt

    return list_files

def extract_date_from_file_name(file_name: str):
    """
    Given a file name like:
       chm_inst_1hr_glo_L1440x721_v72/CF2_control.chm_inst_1hr_glo_L1440x721_v72.20240102_2100z.nc4
    we return:
       20240102 and 2100
    that represent the date and time record for the file.
    """
    file_stem = Path(file_name).stem
    mydate = file_stem.split(".")[-1]
    mydate = mydate.split("_")
    beg_date = mydate[0]
    beg_time = mydate[1][:4]
    if beg_time.endswith('30'):
        beg_time = f"{beg_time[0:2]}00"

    return int(beg_date), int(beg_time)

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

def create_list_dates(beg_date: int, end_date: int, freq_hours: int):
    """
    """
    sdate = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d')
    edate = dttm.datetime.strptime(f'{end_date}', '%Y%m%d')
    edate += dttm.timedelta(days=1)
    edate -= dttm.timedelta(minutes=1)

    return pd.date_range(sdate, edate, freq=f"{freq_hours}h")


