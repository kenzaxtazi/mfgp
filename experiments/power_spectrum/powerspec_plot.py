# # Power Spectrum Plots

import os
import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import scipy as sp
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import scipy.stats as stats
import matplotlib.pyplot as plt

# custom library
from load import era5, aphrodite, data_dir


# Load the data

# APHRODITE

aphro_ds = aphrodite.collect_APHRO(
    'indus', minyear='2000', maxyear='2009-12-31')
mask_filepath = data_dir + '/Masks/Beas_Sutlej_highres_mask.nc'
mask = xr.open_dataset(mask_filepath)
# kwargs={"fill_value": "extrapolate"})
aphro_interp_ds = aphro_ds.interp_like(mask, method='linear')
aphro_crop_ds = aphro_interp_ds.sel(lat=slice(31, 33), lon=slice(77, 79))

"""
plt.figure(figsize=(10, 5))
g = aphro_crop_ds.isel(time=0).tp.plot(x='lon', y='lat', subplot_kws={
    "projection": ccrs.PlateCarree()}, cbar_kwargs={'label': 'Precipitation [mm/day]'})
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
"""

# MFDGP results

filepath = '~/experiments/power_spectrum/outputs'
file_list = os.listdir(filepath)

x_plt_df = pd.DataFrame()

for i in range(10):
    df_temp = pd.read_csv(filepath + '/' + sorted(file_list)
                          [i]).drop(columns=['Unnamed: 0'])
    # Combine into dataframe
    hf_lambda = np.load(
        '~/experiments/exp3/outputs/lambdas.npy')[i]
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
ds_crop = ds.sel(lat=slice(31, 33), lon=slice(77, 79))

"""
plt.figure(figsize=(10, 5))
g = ds_crop.isel(time=0).y_pred.plot(x='lon', y='lat', subplot_kws={
    "projection": ccrs.PlateCarree()}, cbar_kwargs={'label': 'Precipitation [mm/day]'})
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
"""

# ERA5
era5_ds = era5.collect_ERA5('indus', minyear='2000', maxyear='2009-12-31')
era5_interp_ds = era5_ds.interp_like(mask, method='linear')
era5_crop_ds = era5_interp_ds.sel(lat=slice(31, 33), lon=slice(77, 79))
"""
plt.figure(figsize=(10, 5))
g = era5_crop_ds.isel(time=0).tp.plot(x='lon', y='lat', subplot_kws={
    "projection": ccrs.PlateCarree()}, cbar_kwargs={'label': 'Precipitation [mm/day]'})
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
"""

# Plots
plt.rcParams.update({'font.size': 22})

# One power spectrum

# Take the fourier transform
image = aphro_crop_ds.isel(time=0).to_array()
fourier_image = np.fft.fftn(image)
fourier_amplitudes = np.abs(fourier_image)**2

# checking for nan
# np.argwhere(np.isnan(image.values))
# print(image.shape)

# Constructing a wave vector array
npix = 32
kfreq = np.fft.fftfreq(npix) * npix
kfreq2D = np.meshgrid(kfreq, kfreq)
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
knrm = knrm.flatten()
fourier_amplitudes = fourier_amplitudes.flatten()

# Take the power spectrum
kbins = np.arange(0.5, npix//2+1, 1.)
kvals = 0.5 * (kbins[1:] + kbins[:-1])
Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic="mean",
                                     bins=kbins)
Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

"""
plt.figure(figsize=(10, 10))
plt.plot(kvals, Abins)
plt.ylabel('$P(k)$')
plt.xlabel('$k$')
"""

# Many power spectrums

# Constructing a wave vector array
npix = 32
kfreq = np.fft.fftfreq(npix) * npix
kfreq2D = np.meshgrid(kfreq, kfreq)
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
knrm = knrm.flatten()

kbins = np.arange(0.5, npix//2+1, 1.)
kvals = 0.5 * (kbins[1:] + kbins[:-1])

# Take the fourier transforms
APHRO_Abins_list = []
for i in range(len(aphro_crop_ds.time.values)):
    image = aphro_crop_ds.isel(time=i).to_array()
    image_scaled = (image - np.mean(image)) / np.std(image)
    fourier_image = np.fft.fftn(image_scaled)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = fourier_amplitudes.flatten()
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    APHRO_Abins_list.append(Abins)

ERA5_Abins_list = []
for i in range(len(era5_crop_ds.time.values)):
    image = era5_crop_ds.isel(time=i).tp
    image_scaled = (image - np.mean(image)) / np.std(image)
    fourier_image = np.fft.fftn(image_scaled)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = fourier_amplitudes.flatten()
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    ERA5_Abins_list.append(Abins)


MFDGP_Abins_list = []
for i in range(len(ds_crop.time.values)):
    image = ds_crop.isel(time=i).y_pred
    image_scaled = (image - np.mean(image)) / np.std(image)
    fourier_image = np.fft.fftn(image_scaled)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = fourier_amplitudes.flatten()
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    MFDGP_Abins_list.append(Abins)

kvals_tr = 1. / kvals

# Plotting
plt.figure(figsize=(10, 10))
plt.ylabel('$P(k)$')
plt.xlabel('$k^{-1} \quad [deg]$')

m1 = np.mean(APHRO_Abins_list, axis=0)
plt.fill_between(kvals_tr, y1=m1 + np.std(APHRO_Abins_list, axis=0),
                 y2=m1 - np.std(APHRO_Abins_list, axis=0), alpha=0.05)
plt.plot(kvals_tr, m1, label='APHRODITE')

m2 = np.mean(ERA5_Abins_list, axis=0)
plt.fill_between(kvals_tr, y1=m2 + np.std(ERA5_Abins_list, axis=0),
                 y2=m2 - np.std(ERA5_Abins_list, axis=0), alpha=0.05)
plt.plot(kvals_tr, m2, label='ERA5')

m3 = np.mean(MFDGP_Abins_list, axis=0)
plt.fill_between(kvals_tr, y1=m3 + np.std(MFDGP_Abins_list, axis=0),
                 y2=m3 - np.std(MFDGP_Abins_list, axis=0), alpha=0.1)
plt.plot(kvals_tr, m3, label='MFGP')
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
plt.yscale('log')
plt.savefig('powerspec_2deg_2000_2010_bs_test.pdf',
            bbox_inches='tight', dpi=300)
