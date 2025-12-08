import numpy as np 
import xarray as xr
import pandas as pd
import geopandas
import rasterio
from shapely.geometry import Point
import matplotlib.pyplot as plt
from rasterio.plot import show
import rioxarray as rxr

#open timeseries data
mlhfi = xr.open_dataset('mlhfi_timeseries.nc')
dirtclim = xr.open_dataset('dirtclim_timeseries.nc')

#mlhfi linear regression
prediction = mlhfi.polyfit(dim='time', deg=1)
prediction = prediction.to_dataarray()
prediction = prediction.squeeze()


#dirclim linear regression
dirt_pred = dirtclim.polyfit(dim='time', deg=1)
dirt_pred = dirt_pred.to_dataarray()
dirt_pred = dirt_pred.squeeze()
dirt_pred = dirt_pred.sortby('y')

#save slopes to tif 
dirt_pred.rio.to_raster('dirtclim_regression.tif', compress='LZW', dtype="float32")
pred.rio.to_raster('mlhfi_regression.tif', compress='LZW', dtype='float32')
