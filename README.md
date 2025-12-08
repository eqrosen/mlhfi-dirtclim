# Geospatial Data Processing and Bivariate Analysis

This repository contains Python scripts for regridding geospatial data and creating bivariate maps to visualize trends in DIRTCLIM and MLHFI datasets.

## Overview

The workflow enables comparison of two different geospatial datasets by:
1. Regridding MLHFI data to match DIRTCLIM grid resolution
2. Creating time series from multiple TIF files
3. Computing linear regression trends for both datasets
4. Visualizing trends with bivariate maps

## Scripts

### 1. `regrid_mlhfi.py`
Regrids MLHFI data to match DIRTCLIM spatial resolution.

**Usage:**
```bash
python regrid_mlhfi.py <mlhfi_path> <dirtclim_path> <year>
```

**Arguments:**
- `mlhfi_path`: Path to MLHFI TIF file
- `dirtclim_path`: Path to DIRTCLIM TIF file  
- `year`: Year identifier for output filename

**Output:** `mlhfi_regridded_<year>.tif`

**Optional Parameters (in code):**
- `extra_coarsen`: Additional coarsening factor (default: 1)
- `compress`: Compression type (default: "LZW")
- `stat_method`: Aggregation method (default: "max")
- `output_crs`: Output CRS (default: "EPSG:4326")

---

### 2. `create_timeseries.py`
Loads multiple TIF files and combines them into a NetCDF time series dataset.

**Usage:**
```python
python create_timeseries.py
```

**Note:** Modify the file pattern in `__main__` section:
```python
ds1 = load_tifs_to_dataset("mlhfi_regridded/*.tif")
```

**Output:** `mlhfi_timeseries.nc`

---

### 3. `compute_trends.py`
Performs linear regression on time series data for both MLHFI and DIRTCLIM datasets.

**Usage:**
```python
python compute_trends.py
```

**Inputs (hardcoded):**
- `mlhfi_timeseries.nc`
- `dirtclim_timeseries.nc`

**Outputs:**
- `mlhfi_regression.tif` - Regression coefficients for MLHFI
- `dirtclim_regression.tif` - Regression coefficients for DIRTCLIM

---

### 4. `bivariate_map.py`
Creates bivariate choropleth maps showing trends in both datasets simultaneously.

**Usage:**
```bash
python bivariate_map.py <dirtclim_regression> <mlhfi_regression> <n_bins>
```

**Arguments:**
- `dirtclim_regression`: Path to DIRTCLIM regression TIF
- `mlhfi_regression`: Path to MLHFI regression TIF
- `n_bins`: Number of bins per dimension (e.g., 3 for 3×3 grid, 4 for 4×4)

**Output:** `dirtclim_mlhfi_regression.png`

**Color Scheme:**
- Blue: Low values in both datasets
- Red: High values in both datasets
- Orange: High DIRTCLIM, low MLHFI
- Green: Low DIRTCLIM, high MLHFI

---

## Typical Workflow

### Option A: Process One Year at a Time (Recommended for Testing)
```bash
# 1. Regrid MLHFI data for individual years
python regrid_mlhfi.py mlhfi_2000.tif dirtclim_2000.tif 2000
python regrid_mlhfi.py mlhfi_2001.tif dirtclim_2001.tif 2001
# ... repeat for all years

# 2. Create time series
python create_timeseries.py

# 3. Compute trends
python compute_trends.py

# 4. Create bivariate map
python bivariate_map.py dirtclim_regression.tif mlhfi_regression.tif 4
```

### Option B: Process Multiple Years Using File ID (Batch Processing)

For batch processing, modify `regrid_mlhfi.py` to use the batch `main()` function:
```python
# Uncomment this main function in regrid_mlhfi.py:
def main(mlhfi,dirtclim,file_id):
    years= [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
            2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
    year = years[file_id]
    regrid_and_save(mlhfi,dirtclim,year)
```

Then run with array job indexing:
```bash
# Process year 2000 (index 0)
python regrid_mlhfi.py mlhfi_2000.tif dirtclim_2000.tif 0

# Process year 2005 (index 5)
python regrid_mlhfi.py mlhfi_2005.tif dirtclim_2005.tif 5

# Or use a loop to process all years:
for i in {0..20}; do
    python regrid_mlhfi.py mlhfi_timeseries/$i.tif dirtclim_template.tif $i
done

# Or with SLURM array jobs:
# sbatch --array=0-20 regrid_job.sh
```

**Note:** The batch option is useful when:
- Processing many years simultaneously on HPC clusters
- Using SLURM/PBS array jobs
- The MLHFI files follow a predictable naming pattern

After regridding (either method), continue with steps 2-4 above.

---

## Dependencies
```
numpy
xarray
pandas
geopandas
rasterio
rioxarray
xarray-regrid
matplotlib
shapely
```

Install with:
```bash
pip install numpy xarray pandas geopandas rasterio rioxarray xarray-regrid matplotlib shapely
```

## Data Requirements

- **MLHFI data**: TIF files with geospatial fire data
- **DIRTCLIM data**: TIF files with climate data
- Both datasets should have spatial coordinates (x, y) and CRS metadata

## Notes

- The regridding uses the `stat` method from xarray-regrid (default: "max")
- Time series regression uses first-degree polynomial fitting
- Output TIFs use LZW compression and float32 dtype
- All coordinate systems are reprojected to EPSG:4326 (WGS84)
- The bivariate map uses quantile-based binning to create equal-frequency classes

## Troubleshooting

**Issue:** "No valid data points found for binning"
- Check that input regression files contain finite values
- Verify spatial alignment between datasets

**Issue:** Upside-down or misaligned data
- Check y-coordinate sorting in your input files
- Verify CRS metadata is correctly set

**Issue:** Memory errors with large datasets
- Consider using `extra_coarsen` parameter to reduce resolution
- Process data in chunks or tiles
