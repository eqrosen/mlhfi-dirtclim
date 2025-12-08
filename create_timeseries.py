import xarray as xr
import rioxarray
from pathlib import Path
import numpy as np
import argparse

def load_tifs_to_dataset(file_pattern, time_values=None, time_dim='time'):
    """
    Load multiple TIF files as xarray DataArrays and merge into a Dataset along time dimension.
    
    Parameters:
    -----------
    file_pattern : str or Path
        Glob pattern to match TIF files (e.g., "data/*.tif" or list of file paths)
    time_values : list or array-like, optional
        Time values corresponding to each file. If None, uses integer indices.
    time_dim : str, default 'time'
        Name of the time dimension
    
    Returns:
    --------
    xr.Dataset
        Dataset with all TIF files merged along time dimension
    """
    if isinstance(file_pattern, (list, tuple)):
        files = [Path(f) for f in file_pattern]
    else:
        # Handle both relative and absolute paths
        pattern_path = Path(file_pattern)
        if '*' in file_pattern:
            # Extract directory and pattern
            parent = pattern_path.parent if pattern_path.parent != Path('.') else Path('.')
            pattern = pattern_path.name
            files = sorted(parent.glob(pattern))
        else:
            files = [pattern_path]
    
    
    if not files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    # Create time coordinate if not provided
    if time_values is None:
        time_values = np.arange(len(files))
    elif len(time_values) != len(files):
        raise ValueError(f"Number of time values ({len(time_values)}) must match number of files ({len(files)})")
    
    # Load each TIF and add time coordinate
    data_arrays = []
    for i, (file, time_val) in enumerate(zip(files, time_values)):
        # Load TIF with rioxarray
        da = rioxarray.open_rasterio(file, masked=True)
        
        # Remove band dimension if single band
        if 'band' in da.dims and len(da.band) == 1:
            da = da.squeeze('band', drop=True)
        
        # Add time coordinate
        da = da.expand_dims({time_dim: [time_val]})
        #da = da.rio.reproject("EPSG:4326")
        
        data_arrays.append(da)
        print(f"Loaded: {file.name}")
    
    # Concatenate along time dimension
    combined = xr.concat(data_arrays, dim=time_dim)
    
    # Convert to dataset
    ds = combined.to_dataset(name='data')
    
    return ds





if __name__=='__main__':
        ds1 = load_tifs_to_dataset("mlhfi_regridded/*.tif")
        ds1.to_netcdf("mlhfi_timeseries.nc")
        print('saved as netcdf')
