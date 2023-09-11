#!/usr/bin/env python
# coding: utf-8

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
import emukit
import xarray as xr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import scipy as sp
import os
import GPy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from load import beas_sutlej_gauges, era5, data_dir
import sys
sys.path.append('/data/hpcdata/users/kenzi22')

#from utils.metrics import msll


# Load data

lambdas = []

for year in range(1980, 2010):
    minyear = year
    maxyear = year + 1

    all_station_dict = pd.read_csv(
        data_dir + 'bs_gauges/gauge_info.csv', index_col='station').T
    station_list = list(all_station_dict)

    hf_train_list = []
    for station in station_list:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.to_dataframe().dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    # era5.collect_ERA5('indus', minyear=minyear, maxyear=maxyear)
    era5_ds = era5.collect_ERA5('indus', minyear=minyear, maxyear=maxyear)
    era5_df = era5_ds.to_dataframe()

    lf_df = era5_df.dropna().reset_index()
    lf_df1 = lf_df[lf_df['lat'] <= 33.5]
    lf_df2 = lf_df1[lf_df1['lat'] >= 30.25]
    lf_df3 = lf_df2[lf_df2['lon'] >= 75.75]
    lf_train_df = lf_df3[lf_df3['lon'] <= 82.5]

    # Import GMTED2010 data
    gmted_ds = xr.open_dataset(data_dir + '/Elevation/GMTED2010_data.nc')
    gmted_ds = gmted_ds.rename({'nlat': 'lat', 'nlon': 'lon'})

    # Mask to beas and sutlej
    mask_filepath = data_dir + '/Masks/Beas_Sutlej_highres_mask.nc'
    mask = xr.open_dataset(mask_filepath)
    mask_da = mask.Overlap
    msk_hr_data_ds = gmted_ds.where(mask_da > 0, drop=True)

    times = era5_df.reset_index()['time'].drop_duplicates()
    msk_hr_data_ds = msk_hr_data_ds.assign_coords(time=times.values)
    hr_data_ds = msk_hr_data_ds.reindex({'time': times.values}, method='ffill')
    hr_data_df = hr_data_ds.to_dataframe().dropna().reset_index()
    hr_data_df = hr_data_df.rename(columns={'level_0': 'time'})
    hr_data_df = hr_data_df[['time', 'lon', 'lat', 'elevation']]

    # Prepare data

    # Transformations
    lf_train_df['tp_tr'], lf_lambda = sp.stats.boxcox(
        lf_train_df['tp'].values + 0.01)
    hf_train_df['tp_tr'] = sp.stats.boxcox(
        hf_train_df['tp'].values + 0.01, lmbda=lf_lambda)

    lambdas.append(lf_lambda)

pd.to_csv(
    '/data/hpcdata/users/kenzi22/mfdgp/experiments/exp3/lambdas_1980_2010', lambdas)
