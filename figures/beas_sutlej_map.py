import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

from cartopy.io import shapereader
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import numpy as np

import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa
from load import data_dir

# Data

# Topography data
top_ds = xr.open_dataset(data_dir + 'Elevation/GMTED2010_15n015_00625deg.nc')
top_ds = top_ds.assign_coords(
    {'nlat': top_ds.latitude, 'nlon': top_ds.longitude})
top_ds1 = top_ds.sel(nlat=slice(29, 35), nlon=slice(75, 84))

# Beas shapefile and projection
beas_path = data_dir + \
    "Shapefiles/beas-sutlej-shapefile/beas_sutlej_basins/beas_watershed.shp"
beas_shape = shapereader.Reader(beas_path)
beas_globe = ccrs.Globe(semimajor_axis=6377276.345,
                        inverse_flattening=300.8017)
beas_cranfield_crs = ccrs.LambertConformal(
    central_longitude=80, central_latitude=23, false_easting=0.0,
    false_northing=0.0, standard_parallels=[30, 30], globe=beas_globe)

# Sutlej shapefile and projection
stlj_path = data_dir + \
    "Shapefiles/beas-sutlej-shapefile/beas_sutlej_basins/sutlej_watershed.shp"
stlj_shape = shapereader.Reader(stlj_path)
stlj_globe = ccrs.Globe(semimajor_axis=6377276.345,
                        inverse_flattening=300.8017)
stlj_cranfield_crs = ccrs.LambertConformal(
    central_longitude=80, central_latitude=23, false_easting=0.0,
    false_northing=0.0, standard_parallels=[30, 30], globe=stlj_globe)


# Figure
proj = ccrs.PlateCarree()

# Main plot
fig, ax1 = plt.subplots(figsize=(25, 15), subplot_kw={'projection': proj},)
ax1.set_extent([75.5, 83, 30, 33.5])
gl = ax1.gridlines(draw_labels=True, linewidth=0.5,
                   linestyle='--', color='gray')

# Elevation
divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=6000.)
top_ds1.elevation.plot.contourf(cmap='gist_earth', levels=np.arange(0, 6000, 50),
                                norm=divnorm, rasterized=True,
                                cbar_kwargs={'label': 'Elevation [m]', 'location': 'bottom',
                                             'pad': 0.07, 'fraction': 0.05, 'aspect': 20})

# Gauges
gauges = ax1.scatter(df['Longitude (o)'], df['Latitude (o)'], s=180,
                     edgecolor='black', facecolor='white', label="Gauges", zorder=9, linewidths=1)

# Watershed boundaries
for rec in beas_shape.records():
    ax1.add_geometries(
        [rec.geometry],
        beas_cranfield_crs,
        edgecolor="black",
        facecolor="None",
        linestyle=(0, (10, 10)),
        linewidth=1)

for rec in stlj_shape.records():
    ax1.add_geometries(
        [rec.geometry],
        stlj_cranfield_crs,
        edgecolor="black",
        facecolor="None",
        linestyle=(0, (10, 10)),
        linewidth=1)

gl.top_labels = False
gl.right_labels = False
#gl.gridlines(draw_labels=True, linewidth=0.5)
#ax1.set_xticks(np.arange(76, 83, 1), ['76°E', '77°E', '78°E', '79°E', '80°E', '81°E', '82°E'])
#ax1.set_yticks(np.arange(30, 34, 0.5),['30°N', '30.5°N', '31°N', '31.5°N', '32°N', '32.5°N', '33°N', '33.5°N'])
ax1.set_ylabel(" ")
ax1.set_xlabel(" ")
ax1.text(81.6, 30.65, 'S')
ax1.text(77.12, 32.15, 'B')

# Legend
line = mlines.Line2D([], [], color='black', marker='None',
                     label='Watershed boundary', linestyle=(0, (5, 5)))
ax1.legend(handles=[gauges, line], loc='lower left')

# Inset plot
ax2 = fig.add_axes([0.652, 0.54, 0.25, 0.25], projection=proj)

# Map
land_50m = cf.NaturalEarthFeature(
    "physical", "land", "50m", edgecolor="darkgrey",
    facecolor='whitesmoke')
ax2.set_extent([60, 110, 5, 40])
ax2.add_feature(land_50m, zorder=0)
ax2.coastlines("50m", linewidth=0.4)

# High mountain contour
top_ds.elevation.plot.contourf(ax=ax2, colors=['None', '#c67fa8'],
                               levels=[2000, 10000], add_colorbar=False, rasterized=True)

# Rectangle
ax2.add_patch(Rectangle((75.5, 30), 7.5, 3.5,
                        edgecolor='black',
                        fill=False,
                        lw=0.4))

# Save and plot figure
plt.savefig('map_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()


def median_beas_sutlej_elev():
    """ Returns median elevation of the Beas and Sutlej basins """

    # Topography data
    top_ds = xr.open_dataset(
        data_dir + 'Elevation/GMTED2010_15n015_00625deg.nc')
    top_ds = top_ds.assign_coords(
        {'nlat': top_ds.latitude, 'nlon': top_ds.longitude})

    # Basin masks
    mask_filepath = data_dir + 'Masks/Beas_Sutlej_mask.nc'
    mask = xr.open_dataset(mask_filepath)
    mask = mask.rename({'latitude': 'nlat', 'longitude': 'nlon'})
    elv_da = top_ds.elevation.interp_like(mask)
    mask_da = mask.overlap
    masked_da = elv_da.where(mask_da > 0, drop=True)

    median_elev = masked_da.median(skipna=True).values
    print('Median elevation: ', median_elev, 'm')
    return median_elev
