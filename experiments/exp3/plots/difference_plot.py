import os
import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import xarray as xr
import matplotlib.pyplot as plt
import GPy
import pandas as pd
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import scipy as sp

from load import aphrodite, data_dir

# MFGP output

# Load results
filepath = '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp3/outputs/preds_latlonpriors_mat52'
file_list = os.listdir(filepath)

x_plt_df = pd.DataFrame()

for i in range(10):
    df_temp = pd.read_csv(filepath + '/' + sorted(file_list)
                          [i]).drop(columns=['Unnamed: 0'])
    # Combine into dataframe
    hf_lambda = np.load(
        '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp3/outputs/lambdas.npy')[i]
    df_temp['y_pred'] = sp.special.inv_boxcox(
        df_temp['pred0'].values, hf_lambda)
    df_temp['95th'] = sp.special.inv_boxcox(
        df_temp['pred0'].values + 1.96 * np.sqrt(df_temp['y_var0'].values), hf_lambda)
    df_temp['5th'] = sp.special.inv_boxcox(
        df_temp['pred0'].values - 1.96 * np.sqrt(df_temp['y_var0'].values), hf_lambda)
    df_temp['CI'] = df_temp['95th'].fillna(0) - df_temp['5th'].fillna(0)
    x_plt_df = x_plt_df.append(df_temp, ignore_index=True)

# Fill NANs
x_plt_df[['y_pred', '95th', '5th', 'CI']] = x_plt_df[[
    'y_pred', '95th', '5th', 'CI']].fillna(0)

# To Data Array
df = x_plt_df.reset_index()
df = df.set_index(['time', 'lon', 'lat'])
ds = df.to_xarray()

print(ds.y_pred.std())
print(ds.y_pred.mean())


# APHRODITE

# Load data
aphro_ds = aphrodite.collect_APHRO('indus', 2000, 2010)

mask_filepath = data_dir + 'Masks/Beas_Sutlej_highres_mask.nc'
mask = xr.open_dataset(mask_filepath)
# kwargs={"fill_value": "extrapolate"})
aphro_interp_ds = aphro_ds.interp_like(mask, method='linear')
mask_da = mask.Overlap
aphro_interp_ds2 = aphro_interp_ds.where(mask_da > 0, drop=True)

print('APHRODITE mean = ', np.nanmean(aphro_interp_ds2.tp.values))
print('APHRODITE std dev = ', np.nanstd(aphro_interp_ds2.tp.values))

# Check
'''
plt.figure(figsize=(10, 5))
g = aphro_interp_ds2.isel(time=0).tp.plot(x='lon', y='lat', subplot_kws={
    "projection": ccrs.PlateCarree()}, cbar_kwargs={'label': 'Precipitation [mm/day]'})
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
plt.savefig('aphro_check.png', dpi=300)
'''


# Difference

ds['dif'] = ds.y_pred - \
    aphro_interp_ds2.tp.transpose('time', 'lon', 'lat').values

# Means
ds_annual_avg = ds.dif.mean(dim='time')

ds_jun = ds.dif[5::12]
ds_jul = ds.dif[6::12]
ds_aug = ds.dif[7::12]
ds_sep = ds.dif[8::12]
ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
ds_monsoon_avg = ds_monsoon.dif.mean(dim='time')

ds_dec = ds.dif[11::12]
ds_jan = ds.dif[0::12]
ds_feb = ds.dif[1::12]
ds_mar = ds.dif[2::12]
ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
ds_west_avg = ds_west.dif.mean(dim='time')

ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg],
                   pd.Index(["Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t'))

# Plot figure
g = ds_avg.plot(
    x="lon",
    y="lat",
    col="t",
    cbar_kwargs={"label": "Precipitation bias [mm/day]",
                 "pad": 0.03, "ticks": np.arange(-7.5, 8, 2.5)},
    size=5, aspect=2.1,
    subplot_kws={"projection": ccrs.PlateCarree()})

g.axs.flat[0].set_title("Annual", fontsize=22)
g.axs.flat[1].set_title("Monsoon (JJAS)", fontsize=22)
g.axs.flat[2].set_title("Winter (DJFM)", fontsize=22)

for ax in g.axes.flat:
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([75, 83.5, 29, 34])

plt.savefig('plots/difference_2000-2010_test.png',
            dpi=600, bbox_inches="tight")
