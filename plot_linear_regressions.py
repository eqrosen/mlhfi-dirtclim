import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr
import argparse
import pandas as pd
import geopandas
import rasterio
from shapely.geometry import Point
from rasterio.plot import show
import rioxarray as rxr
import xarray_regrid
 

def create_bivariate_map(dirtclim, mlhfi, 
                         n_bins, figsize=(15, 10)):
    """
    Create a bivariate map showing percent changes in both datasets.
    
    Parameters:
    -----------
    dirtclim_early : xr.DataArray
        Early period DIRTCLIM data
    dirtclim_late : xr.DataArray
        Late period DIRTCLIM data
    mlhfi_early : xr.DataArray
        Early period MLHFI regridded data
    mlhfi_late : xr.DataArray
        Late period MLHFI regridded data
    n_bins : int
        Number of bins for each dimension (3x3, 4x4, etc.)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    
    # Check original shapes
    print("Original shapes:")
    print(f"  DIRTCLIM early: {dirtclim.shape}")
    print(f"  MLHFI early: {mlhfi.shape}")

    #dirtclim =dirtclim.reindex(y=list(reversed(dirtclim.y)))

    
    
    # Handle infinities and very large values
    dirtclim = dirtclim.where(np.isfinite(dirtclim))
    mlhfi = mlhfi.where(np.isfinite(mlhfi))
    
    # Create mask for valid (non-NaN) values
    valid_mask = np.isfinite(dirtclim.values) & np.isfinite(mlhfi.values)
    
    # Create quantile bins for each variable (only from valid data)
    dirtclim_valid = dirtclim.values[valid_mask]
    mlhfi_valid = mlhfi.values[valid_mask]
    
    if len(dirtclim_valid) == 0 or len(mlhfi_valid) == 0:
        raise ValueError("No valid data points found for binning")
    
    dirtclim_quantiles = np.quantile(dirtclim_valid, np.linspace(0, 1, n_bins + 1))
    mlhfi_quantiles = np.quantile(mlhfi_valid, np.linspace(0, 1, n_bins + 1))
    
    # Bin the data
    dirtclim_binned = np.digitize(dirtclim.values, dirtclim_quantiles, right=True)
    mlhfi_binned = np.digitize(mlhfi.values, mlhfi_quantiles, right=True)

    # Create bivariate color scheme (using a diverging approach)
    # Colors go from blue (low-low) to red (high-high) with purple/pink for mixed
    bivariate_colors = create_bivariate_colormap(n_bins)
    
    # Map bins to colors
    combined = (dirtclim_binned - 1) * n_bins + (mlhfi_binned - 1)
    rgb_array = np.zeros((*combined.shape, 3))
    
    for i in range(n_bins):
        for j in range(n_bins):
            mask = combined == (i * n_bins + j)
            rgb_array[mask] = bivariate_colors[i, j]
    
    # Set NaN/invalid areas to white
    rgb_array[~valid_mask] = [1.0, 1.0, 1.0]  # White for NaN values
   
    # Create figure with two subplots: map and legend
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.3)
    
    # Main map
    ax_map = fig.add_subplot(gs[0])
    
 # Get coordinates for extent
    x = dirtclim.x.values
    y = dirtclim.y.values
    extent = [x.min(), x.max(), y.max(), y.min()]
    
    # Plot the bivariate map 
    ax_map.imshow(rgb_array, extent=extent, origin='lower', aspect='equal')
    ax_map.set_xlabel('Longitude', fontsize=12)
    ax_map.set_ylabel('Latitude', fontsize=12)
    ax_map.set_title('DIRTCLIM vs MLHFI Trends', 
                     fontsize=14, fontweight='bold')
    
    # Create 2D legend
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')
    
    # Create legend grid
    legend_size = 0.15
    for i in range(n_bins):
        for j in range(n_bins):
            rect = mpatches.Rectangle((j * legend_size, i * legend_size), 
                                       legend_size, legend_size,
                                       facecolor=bivariate_colors[i, j],
                                       edgecolor='white', linewidth=2)
            ax_legend.add_patch(rect)
    
    # Set legend limits and labels
    ax_legend.set_xlim(0, n_bins * legend_size)
    ax_legend.set_ylim(0, n_bins * legend_size)
    ax_legend.set_aspect('equal')
    
    # Add arrows and labels
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # MLHFI arrow (vertical)
    ax_legend.annotate('', xy=(n_bins * legend_size / 2, n_bins * legend_size + 0.05),
                       xytext=(n_bins * legend_size / 2, -0.05),
                       arrowprops=arrow_props)
    ax_legend.text(n_bins * legend_size / 2, n_bins * legend_size + 0.08, 
                   'MLHFI\nIncrease', ha='center', va='bottom', fontsize=10, 
                   fontweight='bold', rotation=0)
    ax_legend.text(n_bins * legend_size / 2, -0.08, 
                   'MLHFI\nDecrease', ha='center', va='top', fontsize=10, 
                   fontweight='bold', rotation=0)
    
    # DIRTCLIM arrow (horizontal)
    ax_legend.annotate('', xy=(n_bins * legend_size + 0.05, n_bins * legend_size / 2),
                       xytext=(-0.05, n_bins * legend_size / 2),
                       arrowprops=arrow_props)
    ax_legend.text(n_bins * legend_size + 0.08, n_bins * legend_size / 2, 
                   'DIRTCLIM\nIncrease', ha='left', va='center', fontsize=10, 
                   fontweight='bold', rotation=270)
    ax_legend.text(-0.08, n_bins * legend_size / 2, 
                   'DIRTCLIM\nDecrease', ha='right', va='center', fontsize=10, 
                   fontweight='bold', rotation=270)
    
    # Add quantile values as text
    info_text = f"DIRTCLIM range: {np.nanmin(dirtclim.values):.1f}% to {np.nanmax(dirtclim.values):.1f}%\n"
    info_text += f"MLHFI range: {np.nanmin(mlhfi.values):.1f}% to {np.nanmax(mlhfi.values):.1f}%"
    
    fig.text(0.7, 0.02, info_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dirtclim_mlhfi_regression.png', dpi=300, bbox_inches='tight')
    return fig, ax_map


def create_bivariate_colormap(n_bins):
    """
    Create a bivariate color scheme.
    Rows (i) = MLHFI axis (vertical in legend)
    Cols (j) = DIRTCLIM axis (horizontal in legend)
    """
    colors = np.zeros((n_bins, n_bins, 3))
    
    # Base colors for corners
    color_low_low = np.array([0.2, 0.3, 0.9])      # Blue (low both)
    color_high_low = np.array([0.9, 0.5, 0.2])     # Orange (high DIRTCLIM, low MLHFI)
    color_low_high = np.array([0.3, 0.8, 0.3])     # Green (low DIRTCLIM, high MLHFI)
    color_high_high = np.array([0.9, 0.2, 0.2])    # Red (high both)
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Normalize positions directly from bin indices
            x_norm = j / (n_bins - 1)  # DIRTCLIM axis (0 to 1)
            y_norm = i / (n_bins - 1)  # MLHFI axis (0 to 1)
            
            # Bilinear interpolation between corner colors
            color_bottom = (1 - x_norm) * color_low_low + x_norm * color_high_low
            color_top = (1 - x_norm) * color_low_high + x_norm * color_high_high
            color = (1 - y_norm) * color_bottom + y_norm * color_top
            
            colors[i, j] = color
    
    return colors


def main(dirtclim, mlhfi, n_bins):
    dirtclim_data = xr.open_dataarray(dirtclim).squeeze()

    mlhfi_data  = xr.open_dataarray(mlhfi).squeeze()
    
    dirtclim_slope = dirtclim_data[0]
    mlhfi_slope = mlhfi_data[0]

    fig, ax = create_bivariate_map(
    dirtclim_slope,  
    mlhfi_slope,     
    n_bins=n_bins,
)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='make bivariate trend map ')
    parser.add_argument('dirtclim', type=str,  help='path to dirtclim regression data')
    parser.add_argument('mlhfi', type=str, help='path to mlhfi regession data')

    parser.add_argument('n_bins', type=int, help='number of bins for color map')

    args = parser.parse_args()
    main(args.dirtclim, args.mlhfi, args.n_bins)
   
