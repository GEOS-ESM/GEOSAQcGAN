"""
Create a HadISST-derived land/sea mask on the AQcGAN 181 x 360 latitude-longitude grid.

This script reads the HadISST2 sea-ice concentration NetCDF file and uses the
valid/missing data pattern to infer ocean versus land points. Valid sea-ice
grid cells are treated as ocean, and missing cells are treated as land.

The source HadISST longitude grid is converted to -180 to 180, then the mask is
remapped to the AQcGAN grid using nearest-neighbor matching.

Input:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/land_sea_mask_source/HadISST.2.2.2.0_sea_ice_concentration.nc

Output:
    /gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata/hadisst_land_sea_mask_181x360.npz

The output .npz file contains:
    lat
        AQcGAN latitude array, shape (181,)

    lon
        AQcGAN longitude array, shape (360,)

    land_mask
        Boolean mask where True means land, shape (181, 360)

    ocean_mask
        Boolean mask where True means ocean, shape (181, 360)

    land_sea_mask
        Integer mask where 1 means land and 0 means ocean, shape (181, 360)

    source_file
        Path to the HadISST2 NetCDF file used to create the mask.

    source_variable
        Name of the HadISST2 data variable used.

This mask is used later to calculate global_land and global_ocean regional
statistics for AQcGAN forecast evaluation.
"""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

SOURCE_FILE = Path(
    "/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/"
    "plot_metadata/land_sea_mask_source/"
    "HadISST.2.2.2.0_sea_ice_concentration.nc"
)

OUT_DIR = Path(
    "/gpfsm/dnb06/projects/p271/aqcgan_products/v2_2_0/plot_metadata"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "hadisst_land_sea_mask_181x360.npz"

aqcgan_lat = np.linspace(-90, 90, 181)
aqcgan_lon = np.linspace(-180, 180, 360, endpoint=False)

with Dataset(SOURCE_FILE) as ds:
    print("Variables:", list(ds.variables.keys()))

    lat_name = "latitude" if "latitude" in ds.variables else "lat"
    lon_name = "longitude" if "longitude" in ds.variables else "lon"

    src_lat = np.array(ds.variables[lat_name][:])
    src_lon = np.array(ds.variables[lon_name][:])

    data_var = None
    for name in ds.variables:
        if name not in {lat_name, lon_name, "time", "time_bnds"}:
            data_var = name
            break

    if data_var is None:
        raise ValueError("Could not find sea-ice data variable.")

    print("Using data variable:", data_var)

    sample = np.array(ds.variables[data_var][0])

    if np.ma.isMaskedArray(ds.variables[data_var][0]):
        valid_ocean = ~np.ma.getmaskarray(ds.variables[data_var][0])
    else:
        fill_value = getattr(ds.variables[data_var], "_FillValue", None)
        missing_value = getattr(ds.variables[data_var], "missing_value", None)

        valid_ocean = np.isfinite(sample)
        if fill_value is not None:
            valid_ocean &= sample != fill_value
        if missing_value is not None:
            valid_ocean &= sample != missing_value

# Normalize source longitude to -180..180
src_lon_180 = ((src_lon + 180) % 360) - 180
sort_idx = np.argsort(src_lon_180)
src_lon_180 = src_lon_180[sort_idx]
valid_ocean = valid_ocean[:, sort_idx]

# Nearest-neighbor remap to AQcGAN grid
lat_idx = np.abs(src_lat[:, None] - aqcgan_lat[None, :]).argmin(axis=0)
lon_idx = np.abs(src_lon_180[:, None] - aqcgan_lon[None, :]).argmin(axis=0)

ocean_mask = valid_ocean[np.ix_(lat_idx, lon_idx)].astype(bool)
land_mask = ~ocean_mask

np.savez_compressed(
    OUT_FILE,
    lat=aqcgan_lat,
    lon=aqcgan_lon,
    land_mask=land_mask,
    ocean_mask=ocean_mask,
    land_sea_mask=land_mask.astype(np.int8),
    source_file=str(SOURCE_FILE),
    source_variable=data_var,
    note="1=land, 0=ocean. Derived from HadISST2 valid/missing sea-ice mask and remapped nearest-neighbor to AQcGAN 181x360 grid.",
)

print(f"Saved: {OUT_FILE}")
print(f"Shape: {land_mask.shape}")
print(f"Land cells : {land_mask.sum()}")
print(f"Ocean cells: {ocean_mask.sum()}")