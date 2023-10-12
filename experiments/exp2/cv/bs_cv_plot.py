import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as crf
import matplotlib.pyplot as plt
import seaborn as sns

from cartopy.io import shapereader

# custom modules
from load import beas_sutlej_gauges, data_dir

# Load data
recov_cluster_centres = np.load(
    'experiments/exp2/cv/exp2_cv_cluster_centers.npy')
recov_locs = np.load('experiments/exp2/cv/cv_locs.npy')

all_station_dict = pd.read_csv(
    data_dir + 'bs_gauges/gauge_info.csv', index_col='station').T
sta_list = list(all_station_dict)

df_list = []
minyear = '2000'
maxyear = '2004-12-31'

for station in sta_list:
    station_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=minyear, maxyear=maxyear)
    df_list.append(station_ds.to_dataframe().dropna().reset_index())
    sta_df = pd.concat(df_list)

gauge_df = sta_df[(sta_df['time'] > '2000') & (sta_df['time'] < '2005')]


# Load shapefiles and define projections

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

fig = plt.figure(figsize=(6, 5))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_patch(plt.Rectangle((75.7, 30.25), 0.25, 0.25,
             facecolor='white', edgecolor='black'))
ax.text(75.77, 30.43, "b", fontsize=14, va='top')


for rec in beas_shape.records():
    ax.add_geometries(
        [rec.geometry],
        beas_cranfield_crs,
        edgecolor="whitesmoke",
        facecolor="white",
    )

for rec in stlj_shape.records():
    ax.add_geometries(
        [rec.geometry],
        stlj_cranfield_crs,
        edgecolor="whitesmoke",
        facecolor="white",
    )

ax.set_xlim([75.7, 79.2])
ax.set_ylim([30.25, 33.2])

colormap = sns.color_palette("colorblind", len(recov_cluster_centres))
for i in range(len(recov_cluster_centres)):
    plt.scatter(recov_locs[i, :, 0],
                recov_locs[i, :, 1], label='fold ' + str(i+1), color=colormap[i], zorder=9)

plt.scatter(gauge_df['lon'], gauge_df['lat'], edgecolor='k',
            zorder=8, label='other stations', alpha=0.1)
plt.scatter(recov_cluster_centres[:, 0], recov_cluster_centres[:, 1],
            c='k', marker='*', label='cluster centres', zorder=10)

plt.xticks([76.0, 76.5, 77.0, 77.5, 78.0, 78.5, 79], [
           '76°E', '76.5°E', '77°E', '77.5°E', '78°E', '78.5°E', '79°E'])
plt.yticks([30.5, 31.0, 31.5, 32.0, 32.5, 33.0], [
           '30.5°N', '31°N', '31.5°N', '32°N', '32.5°N', '33°N'])

plt.legend(fontsize=8)
plt.savefig('exp2_cv_with_inset_colorblind.pdf', bbox_inches='tight', dpi=300)
