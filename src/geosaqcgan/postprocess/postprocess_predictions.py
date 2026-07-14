"""
This script has to be run from the root directory of the repository with the command
`python -m inference.postprocess_predictions`
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

try:
    from ..shared.gen_utils import read_yaml_file
    from ..shared.gen_utils import read_pickle_file
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared/')))
    from gen_utils import read_yaml_file
    from gen_utils import read_pickle_file


def write_val_file(ft, fcst_init_times, fcst_lead_times,
                    exp_dir, meta_data, exp_name,
                    predictions_mean, val_mean ):

    # Make aqcgan output folder if it doesn't exist
    output_path = os.path.join(exp_dir,"aqcgan_predictions")
    os.makedirs(output_path, exist_ok=True)

    # file name    
    ts = pd.Timestamp(fcst_init_times[ft]).strftime("%Y%m%d_%Hz")
    save_file = f"{output_path}/{exp_name}.aqcgan_prediction.{ts}.nc4"

    # variable encoding
    encoding_dict = {"dtype": "float32", "zlib": True, "complevel": 3, '_FillValue':np.nan}

    ds = xr.Dataset(
            data_vars={
                **{f"{var_name}_pred": (['time','lat','lon'], predictions_mean[ft,i,:,:,:]) 
                for i, var_name in enumerate(outvars)
                },
                **{f"{var_name}_truth": (['time','lat','lon'], val_mean[ft,i,:,:,:]) 
                for i, var_name in enumerate(outvars)
                },
                'fcst_lead_time': (['time'], fcst_lead_times )
                },
            coords= {
                'time': fcst_init_times[ft] + fcst_lead_times,
                'lat': meta_data['lat'],
                'lon': meta_data['lon'],
                }
            )

    ds.attrs['title'] = 'AQcGAN predictions'

    if os.path.exists(save_file):
        print(f'Appending {save_file}')
        with xr.open_dataset( save_file, 
                            decode_timedelta=True ) as ds_prev: 
            ds = ds_prev.combine_first(ds)

    ds.to_netcdf(save_file, engine='h5netcdf', mode="w", 
                    encoding={var: encoding_dict for var in ds.data_vars})

def write_pred_file(ft, fcst_init_times, fcst_lead_times,
                    exp_dir, meta_data, exp_name, 
                    predictions_mean ):

    # Make aqcgan output folder if it doesn't exist
    output_path = os.path.join(exp_dir,"aqcgan_predictions")
    os.makedirs(output_path, exist_ok=True)

    # variable encoding
    encoding_dict = {"dtype": "float32", "zlib": True, "complevel": 3, '_FillValue':np.nan}

    # file name    
    for lead_time in fcst_lead_times:
        init_time_str = pd.Timestamp(fcst_init_times[ft]).strftime("%Y%m%d_%Hz")
        fcst_time = fcst_init_times[ft] + fcst_lead_times
        fcst_time_str= pd.Timestamp(fcst_time).strftime("%Y%m%d_%Hz")
        save_file = f"{output_path}/{exp_name}.aqcgan_prediction.{init_time_str}+{fcst_time_str}+.nc4"

        ds = xr.Dataset(
                data_vars={
                    **{f"{var_name}_pred": (['time','lat','lon'], predictions_mean[ft,i,:,:,:]) 
                    for i, var_name in enumerate(outvars)
                        },
                    },
                coords= {
                    'time': fcst_time,
                    'lat': meta_data['lat'],
                    'lon': meta_data['lon'],
                    }
                )

        ds.attrs['title'] = 'AQcGAN predictions'

        ds.to_netcdf(save_file, engine='h5netcdf', mode="w", 
                        encoding={var: encoding_dict for var in ds.data_vars})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Full path to the experiment directory")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--config_filepath", type=str, help="Full path to the experiment config file")
    parser.add_argument("--meta_filepath", type=str, help="Full path to the Pickle file containing experiment data.")
    parser.add_argument("--n_passes", "-n", type=int, default=1, help="Number of forward passes to run.")
    parser.add_argument("--vertical_level", type=int, default=72, help="Vertical level to evaluate on")
    parser.add_argument("--split", type=str, default="val", help="AQcGAN split: val or test")
    parser.add_argument("--mode", type=str, default="validate", help="validate or forecast")
    args = parser.parse_args()

    # Open experiment config file
    config_filepath = Path(args.config_filepath)
    config          = read_yaml_file(config_filepath)
    if not config:
        sys.exit(1)

    # open predictions npz file
    pred_npz_file = f"{args.exp_dir}/{args.split}_ens_pred_stats_days{args.n_passes}_level{args.vertical_level}.npz"
    predictions_dict = np.load(pred_npz_file)

    predictions_mean = predictions_dict[f"ens_mean_pred_{args.split}"]

    # get meta data
    meta_data = read_pickle_file(args.meta_filepath)    
    if not meta_data:
        sys.exit(1)

    # Get some config values
    exp_name = args.exp_name
    times = np.array(meta_data['time'], dtype='datetime64')
    n_timesteps = len(times)
    window_size = config["data"]["n_frames"]*config["data"]["step_size"]
    time_span   = n_timesteps - window_size*(args.n_passes + 1) + 1
    
    # save predictions as nc4 files
    # filename has the form {exp_name}.aqcgan_prediction.{fcst init time}.nc4
    # fcst init time is the last timestamp of the GEOS species (only) data 
    # used for the forecast.
    # So if we use the data for 20230401 (0z-21z) as the initial condition, 
    # then the fcst init time is 20240402_21z
    #         
    # this is the last time of the GEOS-CF composition data used for a forecast
    fcst_init_times = times[window_size-1:
                            n_timesteps-window_size*args.n_passes]
    
    # Forecast lead times for the 8 n_frames
    fcst_lead_times = np.arange( 3 * window_size*(args.n_passes - 1) + 3,
                                3 * window_size*args.n_passes + 3, 
                                3, 
                                dtype='timedelta64[h]' )



    # output variables
    outvars = config["data"]["target_names"][args.vertical_level]

    if args.mode == "validate":
        # open predictions npz file
        val_npz_file = f"{args.exp_dir}/{args.split}_ens_stats_days{args.n_passes}_level{args.vertical_level}.npz"
        print(f"Reading AQcGAN output file: {val_npz_file}")
        val_dict = np.load(val_npz_file)

        val_mean = val_dict[f"ens_mean_{args.split}"]
        
        dummy = [write_val_file(
                    ft, fcst_init_times, fcst_lead_times, 
                    args.exp_dir, meta_data, exp_name,
                    predictions_mean, val_mean
                                ) 
                        for ft in range(len(fcst_init_times))]

    else:
        # get outfile name
        ts = str(meta_data['time_init'][0])[:13]
        dt = meta_data['time_init'][1] - meta_data['time_init'][0]
        dt = dt.astype('timedelta64[h]').astype(int)

        save_file = f"{exp_name}.aqcgan_prediction.{ts}.nc4"

        if os.path.exists(config.chkpt_dir / save_file):
            print(f'{save_file} already exists! Skipping')
            exit()
            
        ds = xr.Dataset(
        data_vars={
            'CO':  (['time', 'hour', 'lat', 'lon'], predictions_mean[:,0,:,:,:]),
            'NO':  (['time', 'hour', 'lat', 'lon'], predictions_mean[:,1,:,:,:]),
            'NO2': (['time', 'hour', 'lat', 'lon'], predictions_mean[:,2,:,:,:]),
            'O3':  (['time', 'hour', 'lat', 'lon'], predictions_mean[:,3,:,:,:])

        },
        coords={
            'time': meta_data['time_init'][0:time_span],
            'hour': range(window_size)*dt,
            'lat': meta_data['lat'],
            'lon': meta_data['lon']
        }
    )

        ds.attrs['title'] = 'AQcGAN predictions'
        ds.to_netcdf(config.chkpt_dir / save_file,engine='netcdf4')
