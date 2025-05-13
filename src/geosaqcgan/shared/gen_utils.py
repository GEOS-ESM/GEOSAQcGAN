#!/usr/bin/env python
"""
Utility functions:
    - read_yaml_file
    - read_pickle_file
    - get_list_files
    - create_list_dates
"""

import yaml
import os
import sys
import glob
from pathlib import Path
import datetime as dttm
import pandas as pd
import pickle

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

def read_pickle_file(filepath):
    """ 
    Reads a pickle file and returns the unpickled object.
 
    Args:
        filepath (str): The path to the pickle file.
 
    Returns:
        object: The unpickled object, or None if an error occurs.
    """
    try:
        with open(filepath, 'rb') as fid:
            data = pickle.load(fid)
            return data
    except FileNotFoundError:
        print(f"Error: {filepath} does not exist!")
        return None
    except EOFError:
         print(f"Error: End of file reached unexpectedly. The file might be empty or corrupted.")
         return None
    except pickle.UnpicklingError:
        print(f"Error: Unpickling error. The file might be corrupted or contain an invalid pickle format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_list_files(data_dir: str, file_prefix: str, beg_date: int, end_date: int) -> list:
    """
    Gather the list of files (with a specific prefix) within a date range
    that are located in a directory.
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

def create_list_dates(beg_date: int, end_date: int, 
                      freq_hours: int):
    """
    Given a starting date, ending date and a frequency (in hours),
    list all the dates (as datetime objects) in the date range and
    at the frequency. This function is used to gather all the data
    files produced between the two dates (end dates included).

    Parameters
    ----------
    beg_date : int
       Staring date
    end_date : int
       End date
    freq_hours : int
       Number of hours between two consecutive dates

    Returns
    -------
    dates : pd.DataFrame
       
    """
    sdate = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d')
    edate = dttm.datetime.strptime(f'{end_date}', '%Y%m%d')
    # Because we need to include
    edate += dttm.timedelta(days=1)
    edate -= dttm.timedelta(minutes=1)

    dates = pd.date_range(sdate, edate, freq=f"{freq_hours}h")

    return dates


