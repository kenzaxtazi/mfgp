import os
import sys  # noqa
sys.path.append('/data/hpcdata/users/kenzi22')  # noqa

import scipy as sp
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from PIL import ImageColor
from generativepy.color import Color
from matplotlib.colors import rgb2hex
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from load import era5, data_dir, location_sel


def load_results(dataframe=False) -> xr.Dataset:

    filepath = 'experiments/exp3/outputs_2000_2010'
    file_list = os.listdir(filepath)

    x_plt_df = pd.DataFrame()

    for i in range(10):
        df_temp = pd.read_csv(filepath + '/' + sorted(file_list)
                              [i]).drop(columns=['Unnamed: 0'])
        # Combine into dataframe
        hf_lambda = np.load(
            'experiments/exp3/plots/lambdas.npy')[i]
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
    if dataframe == True:
        return df
    else:
        df = df.set_index(['time', 'lon', 'lat'])
        ds = df.to_xarray()
        return ds


def seasonal_means(da: xr.DataArray, var: str) -> xr.Dataset:

    ds_annual_avg = da.mean(dim='time')

    ds_jun = da[5::12]
    ds_jul = da[6::12]
    ds_aug = da[7::12]
    ds_sep = da[8::12]
    ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
    ds_monsoon_avg = ds_monsoon[var].mean(dim='time')

    ds_dec = da[11::12]
    ds_jan = da[0::12]
    ds_feb = da[1::12]
    ds_mar = da[2::12]
    ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
    ds_west_avg = ds_west[var].mean(dim='time')

    ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg],
                       pd.Index(["Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t'))

    return ds_avg


def plot_mean_posterior() -> xr.Dataset:
    """ Plot mean posterior figure """
    ds = load_results()
    ds_avg_ypred = seasonal_means(ds.y_pred, 'y_pred')

    # Plot
    g = ds_avg_ypred.plot(
        x="lon",
        y="lat",
        col="t",
        cbar_kwargs={"label": "MFGP precipitation [mm/day]", "pad": 0.03},
        size=5, aspect=2.1,
        subplot_kws={"projection": ccrs.PlateCarree()})

    '''
    g.axes.flat[0].set_title("Annual", fontsize=22)
    g.axes.flat[1].set_title("Monsoon (JJAS)", fontsize=22)
    g.axes.flat[2].set_title("Winter (DJFM)", fontsize=22)
    '''
    g.axes.flat[0].set_title(" ")
    g.axes.flat[1].set_title(" ")
    g.axes.flat[2].set_title(" ")

    for ax in g.axes.flat:
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([75, 83.5, 29, 34])
        ax.set_xlabel("Longitude")

    for artist in ax.get_children():
        if isinstance(artist, plt.Line2D):
            artist.set_antialiased(False)

    plt.savefig('experiments/exp3/plots/seasonal_2000-2010_test.png',
                dpi=600, bbox_inches="tight")
    
    return ds_avg_ypred


def plot_confidence_interval() -> xr.Dataset:
    """ Plot confidence interval figure """

    ds = load_results()
    ds_avg_CI = seasonal_means(ds.CI, 'CI')

    g = ds_avg_CI.plot(
        x="lon",
        y="lat",
        col="t",
        cmap='magma',
        cbar_kwargs={"label": "MFGP 95% CI [mm/day]", "pad": 0.03},
        size=5, aspect=2.1,
        subplot_kws={"projection": ccrs.PlateCarree()})

    g.axes.flat[0].set_title(" ")
    g.axes.flat[1].set_title(" ")
    g.axes.flat[2].set_title(" ")

    for ax in g.axes.flat:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([75, 83.5, 29, 34])
        ax.set_xlabel("Longitude")

    g.axes.flat[0].set_ylabel("Latitude")

    # Remove the lines from the SVG
    for artist in ax.get_children():
        if isinstance(artist, plt.Line2D):
            artist.set_edgecolor('none')

    plt.savefig('experiments/exp3/plots/seasonal_CI_2000-2010_test.png',
                dpi=600, bbox_inches="tight")
    return ds_avg_CI

def plot_bivariate_chloropleth(ds_avg_ypred: xr.Dataset, ds_avg_CI: xr.Dataset) -> None:
    """ Plot bivariate chloropleth plot """

    # Load data
    df_pred = ds_avg_ypred.to_dataframe().reset_index().dropna()
    df_ci = ds_avg_CI.to_dataframe().reset_index().dropna()
    values = df_pred['y_pred'].values/df_pred['y_pred'].max()*100
    ci = df_ci['CI'].values/df_ci['CI'].values.max()*100

    # Percentile bounds defining upper boundaries of color classes
    percentile_bounds = [25, 50, 75, 100]
    colorlist = bivariate_chloropleth_colorlist(percentile_bounds)

    # plot map based on bivariate choropleth
    df_pred['color_bivariate'] = [get_bivariate_choropleth_color(
        p1, p2, percentile_bounds) for p1, p2 in zip(values, ci)]
    da3 = df_pred.set_index(['t', 'lon', 'lat']).to_xarray()

    g = da3.color_bivariate.plot(x='lon', y='lat', col='t', colors=colorlist, add_colorbar=False, levels=np.arange(1, 17),
                                 size=5, aspect=1.7, subplot_kws={"projection": ccrs.PlateCarree()})

    for ax in g.axes.flat:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([75, 83.5, 29, 34])

    g.axes.flat[0].set_title(" ")
    g.axes.flat[1].set_title(" ")
    g.axes.flat[2].set_title(" ")

    # now create inset legend
    ax = ax.inset_axes([1.1, 0.3, 0.5, 0.5])
    ax.set_aspect('equal', adjustable='box')
    count = 0
    xticks = [0]
    yticks = [0]
    for i, percentile_bound_p1 in enumerate(percentile_bounds):
        for j, percentile_bound_p2 in enumerate(percentile_bounds):
            percentileboxes = [Rectangle((i, j), 1, 1)]
          
            pc = PatchCollection(
                percentileboxes, facecolor=colorlist[count], alpha=0.85)
            count += 1
            ax.add_collection(pc)
            if i == 0:
                yticks.append(percentile_bound_p2)
        xticks.append(percentile_bound_p1)

    _ = ax.set_xlim([0, len(percentile_bounds)])
    _ = ax.set_ylim([0, len(percentile_bounds)])
    _ = ax.set_xticks(list(range(len(percentile_bounds)+1)),
                      xticks, fontsize=12)
    _ = ax.set_xlabel('Mean value', fontsize=22)
    _ = ax.set_yticks(list(range(len(percentile_bounds)+1)),
                      yticks, fontsize=12)
    _ = ax.set_ylabel('Uncertainty', fontsize=22)

    plt.savefig('experiments/exp3/plots/seasonal_bivariate_2000_2010_test.png', dpi=600, bbox_inches="tight")


def bivariate_chloropleth_colorlist(percentile_bounds) -> list:
    """ Return list of colors for bivariate chloropleth plot """

    # Function to convert hex color to rgb to Color object (generativepy package)
    def hex_to_Color(hexcode):
        rgb = ImageColor.getcolor(hexcode, 'RGB')
        rgb = [v/256 for v in rgb]
        rgb = Color(*rgb)
        return rgb

    # get corner colors from https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
    c00 = hex_to_Color('#e8e8e8')
    c10 = hex_to_Color('#be64ac')
    c01 = hex_to_Color('#5ac8c8')
    c11 = hex_to_Color('#3b4994')

    # now create square grid of colors, using color interpolation from generativepy package
    num_grps = len(percentile_bounds)
    c00_to_c10 = []
    c01_to_c11 = []
    colorlist = []
    for i in range(num_grps):
        c00_to_c10.append(c00.lerp(c10, 1/(num_grps-1) * i))
        c01_to_c11.append(c01.lerp(c11, 1/(num_grps-1) * i))
    for i in range(num_grps):
        for j in range(num_grps):
            colorlist.append(c00_to_c10[i].lerp(
                c01_to_c11[i], 1/(num_grps-1) * j))

    # convert back to hex color
    colorlist = [rgb2hex([c.r, c.g, c.b]) for c in colorlist]
    return colorlist


def get_bivariate_choropleth_color(p1, p2, percentile_bounds):
    if p1 >= 0 and p2 >= 0:
        count = 0
        stop = False
        for percentile_bound_p1 in percentile_bounds:
            for percentile_bound_p2 in percentile_bounds:
                if (not stop) and (p1 <= percentile_bound_p1):
                    if (not stop) and (p2 <= percentile_bound_p2):
                        color = count
                        stop = True
                count += 1
    # else:
    #    color = [0.8,0.8,0.8,1]
    return color

def plot_era5()-> None :

    era5_ds= era5.collect_ERA5('indus', '2000', '2010')
    mask_filepath = data_dir + 'Masks/Beas_Sutlej_mask.nc'
    era5_interp_ds = location_sel.apply_mask(era5_ds, mask_filepath)

    ds_avg = seasonal_means(era5_interp_ds.tp, 'tp')

    # Plot map
    g = ds_avg.plot(
        x="lon",
        y="lat",
        col="t",
        cbar_kwargs={"label": "ERA5 precipitation [mm/day]", "pad": 0.03},
        size=5, aspect=2.1,
        subplot_kws={"projection": ccrs.PlateCarree()})

    g.axes.flat[0].set_title("Annual", fontsize=22)
    g.axes.flat[1].set_title("Monsoon (JJAS)", fontsize=22)
    g.axes.flat[2].set_title("Winter (DJFM)", fontsize=22)

    for ax in g.axes.flat:
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([75, 83.5, 29, 34])

    plt.savefig('experiments/exp3/plots/era5_seasonal_2000-2010_test.png', dpi=600, bbox_inches="tight")


if __name__ in '__main__':
    # plot_era5()
    ds_avg_ypred = plot_mean_posterior()
    ds_avg_CI = plot_confidence_interval()
    plot_bivariate_chloropleth(ds_avg_ypred, ds_avg_CI)

