import argparse
import numpy as np 
import xarray as xr
import pandas as pd
import geopandas
import rasterio
from shapely.geometry import Point
import matplotlib.pyplot as plt
from rasterio.plot import show
import rioxarray as rxr
import xarray_regrid

def regrid_and_save(mlhfi, dirtclim, year, output_dir=".", **kwargs):
    """
    Regrid MLHFI data to DIRTCLIM grid and save as tif file 
    
    Parameters:
    mlhfi_path : str
        Path to MLHFI TIF file
    dirtclim_path : str
        Path to DIRTCLIM TIF file
    year : int or str
        Year to include in output filename
    output_dir : str
        Directory to save output file
    **kwargs : dict
        Additional keyword arguments:
        - extra_coarsen : int, additional coarsening factor (default: 1)
        - compress : str, compression type for output (default: "LZW")
        - stat_method : str, aggregation method (default: "max")
        - output_crs : str, output CRS (default: "EPSG:4326")
    
    Returns:
    result : xarray.DataArray
        Regridded result
    """
    # Extract kwargs with defaults
    extra_coarsen = kwargs.get('extra_coarsen', 1)
    compress = kwargs.get('compress', 'LZW')
    stat_method = kwargs.get('stat_method', 'max')
    output_crs = kwargs.get('output_crs', 'EPSG:4326')
    
    dirtclim = xr.open_dataset(dirtclim)
    mlhfi = xr.open_dataset(mlhfi)
    # Reproject DIRTCLIM 
    dirtclim_reprojected = dirtclim.rio.reproject(output_crs)
    
    # Clean up (what is spatial ref??)
    dirtclim_clean = dirtclim_reprojected.drop_vars('spatial_ref', errors='ignore')
    mlhfi_clean = mlhfi.drop_vars('spatial_ref', errors='ignore')
    
    # Get variables and remove band dimension
    mlhfi_var = mlhfi_clean['band_data'].squeeze('band', drop=True)
    dirtclim_var = dirtclim_clean['band_data'].squeeze('band', drop=True)
    
    # make sure its not  upside down. if your data is sorted, this may not be necessary
    mlhfi_var = mlhfi_var.sortby('y')
    dirtclim_var = dirtclim_var.sortby('y')

    print("mlhfi size:", mlhfi_var.shape)
    print('dirtclim size:', dirtclim_var.shape)
    
    # Regrid
    print("Regridding...")
    result = mlhfi_var.regrid.stat(dirtclim_var, stat_method)
    print('regridded shape:', result.shape)
    
    # Set CRS
    result.rio.write_crs(output_crs, inplace=True)
    
    # Save
    output_path = f"{output_dir}/mlhfi_regridded_{year}.tif"
    print(f"Saving to {output_path}...")
    result.rio.to_raster(output_path, compress=compress, dtype="float32")
    
    print(f"Done! Saved to {output_path}")
    
    
    return result

# run this main function to run a bunch of years at a time
def main(mlhfi,dirtclim,file_id):
        years= [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
        year = years[file_id]
        regrid_and_save(mlhfi,dirtclim,year)

#run this main function to run one year at a time. best for debugging and testing small batches
def main(mlhfi,dirtclim,year):
        regrid_and_save(mlhfi,dirtclim,year)


if __name__=='__main__':
        parser = argparse.ArgumentParser(description='regrid data from mlhfi onto dirtclim')
        parser.add_argument('mlhfi', type=str, help='path to mlhfi data')
        parser.add_argument('dirtclim', type=str,  help='path to dirtclim data')
        parser.add_argument('year', type=int,  help='year of data')
        # parser.add_argument('file_id', type=int, help='year of the data you are regridding')
        args = parser.parse_args()
        main(args.mlhfi, args.dirtclim, args.year)
