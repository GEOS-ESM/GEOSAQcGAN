#!/usr/bin/env python
"""
Utility functions:
    - read_yaml_file
    - read_pickle_file
    - get_list_files
    - create_list_dates
"""

import yaml
import glob
import datetime as dttm
import pandas as pd
import pickle
from typing import Union

def read_yaml_file(file_name) -> dict | None:
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


def get_list_files(data_dir: str, file_prefix: str, 
                   beg_date: str, end_date: str,
                   freq_nhours: int=1) -> list:
    """
    Gather the list of files (with a specific prefix) within a date range
    that are located in a directory. We will only take file at the
    freq_nhours frequency.

    Parameters
    ----------
    data_dir : str
       Full path to the directory where the data files reside.
    file_prefix : str
       Prefix of the file to be read.
    beg_date : str
       Start date in the format YYYYMMDD_HHz
    end_date : str
       End date in the format YYYYMMDD_HHz
    freq_nhours : int
       Frequency (in hours) for reading files.
    """

    # Convert the dates from string into datetime object
    date_s = dttm.datetime.strptime(beg_date, '%Y%m%d_%Hz')
    date_e = dttm.datetime.strptime(end_date, '%Y%m%d_%Hz')

    freq_dt = dttm.timedelta(hours=freq_nhours)

    list_files = list()
    cur_date = date_s
    while cur_date <= date_e:
        files = f"{data_dir}/{file_prefix}.{cur_date.strftime('%Y%m%d_%H')}*.nc4"
        list_files += sorted(glob.glob(files))
        cur_date += freq_dt

    return list_files

def create_list_dates(beg_date: str, end_date: str, 
                      freq_nhours: int=1):
    """
    Given a starting date, ending date and a frequency (in hours),
    list all the dates (as datetime objects) in the date range and
    at the frequency. This function is used to gather all the data
    files produced between the two dates (end dates included).

    Parameters
    ----------
    beg_date : str
       Staring date in the format YYYYMMDD_HH
    end_date : str
       End date in the format YYYYMMDD_HH
    freq_nhours : int
       Number of hours between two consecutive dates

    Returns
    -------
    dates : pd.DataFrame
       
    """
    sdate = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d_%H%M')
    edate = dttm.datetime.strptime(f'{end_date}', '%Y%m%d_%H%M')
    # Because we need to include
    #edate += dttm.timedelta(days=1)
    #edate -= dttm.timedelta(minutes=1)

    dates = pd.date_range(sdate, edate, freq=f"{freq_nhours}h")

    return dates

def calc_nhours_between_dates(beg_date: str, end_date: str) -> int:
    """
    Calculate the number of hours between two dates in the format YYYYMMDD_HH.

    Parameters
    ----------
    beg_date : str
       Staring date in the format YYYYMMDD_HH
    end_date : str
       End date in the format YYYYMMDD_HH

    Returns
    -------
    nhours : int
       Number of hours between two dates
    """
    sdate = dttm.datetime.strptime(f'{beg_date}', '%Y%m%d_%H%M')
    edate = dttm.datetime.strptime(f'{end_date}', '%Y%m%d_%H%M')

    nhours = (edate - sdate).total_seconds() / 3600

    return int(nhours)

