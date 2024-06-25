import xarray as xr
from tqdm import tqdm
import os
from pathlib import Path
import argparse

obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'

#climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr'

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='data/era5_240/full/', help='where to store outputs')
parser.add_argument('--year', default=1979, type=int, help='year to download')

args = parser.parse_args()

Path(args.folder).mkdir(parents=True, exist_ok=True)

for hour in (0, 6, 12, 18):
    fname = Path(args.folder)/f'era5_240_{args.year}_{hour}h.nc'
    if not Path(fname).exists() or not os.stat(fname).st_size in [4580704409, 4593249697]:
        obs_xarr = xr.open_zarr(obs_path)
    
        ds = obs_xarr.sel(time=obs_xarr.time.dt.year.isin([args.year]))
        ds2 = ds.sel(time=ds.time.dt.hour.isin([hour]))
    
        ds2.to_netcdf(fname)
            