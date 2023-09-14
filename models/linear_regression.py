from sklearn.linear_model import LinearRegression
from utils.metrics import rmses, msll
from sklearn.metrics import r2_score
import scipy as sp
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from load import beas_sutlej_gauges, era5
import sys
sys.path.append('/data/hpcdata/users/kenzi22/')
sys.path.append('/data/hpcdata/users/kenzi22/mfgp/')


if __name__ in "__main__":

   # Load data
    minyear = 2000
    maxyear = 2001

    train_stations = ['Banjar', 'Churah', 'Jogindernagar', 'Kalatop', 'Kangra', 'Sujanpur',
                      'Dadahu', 'Dhaula Kuan', 'Kandaghat', 'Nahan', 'Dehra',
                      'Pachhad', 'Paonta Sahib', 'Rakuna', 'Jubbal', 'Kothai',
                      'Mashobra', 'Rohru', 'Theog', 'Kalpa', 'Salooni', 'Hamirpur', 'Nadaun', ]
    hf_train_list = []
    for station in train_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.to_dataframe().dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj',
                    'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']
    val_list = []
    for station in val_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.to_dataframe().dropna().reset_index())
    val_df = pd.concat(val_list)

    era5_ds = era5.collect_ERA5('indus', minyear=minyear, maxyear=maxyear)
    #era5_df = era5.gauges_download(val_stations + train_stations, minyear=minyear, maxyear=maxyear)

    lf_df = era5_ds.to_dataframe().dropna().reset_index()
    lf_df1 = lf_df[lf_df['lat'] <= 33.5]
    lf_df2 = lf_df1[lf_df1['lat'] >= 30]
    lf_df3 = lf_df2[lf_df2['lon'] >= 75.5]
    lf_train_df = lf_df3[lf_df3['lon'] <= 83]

    # Prepare data

    # Transformations
    lf_train_df['tp_tr'], lf_lambda = sp.stats.boxcox(
        lf_train_df['tp'].values + 0.01)
    hf_train_df['tp_tr'] = sp.stats.boxcox(
        hf_train_df['tp'].values + 0.01, lmbda=lf_lambda)
    val_df['tp_tr'] = sp.stats.boxcox(
        val_df['tp'].values + 0.01, lmbda=lf_lambda)

    # Splitting
    x_train_lf = lf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_lf = lf_train_df['tp_tr'].values.reshape(-1)
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df[['tp_tr']].values.reshape(-1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1)

    # Scaling
    scaler = MinMaxScaler().fit(x_train_hf)
    x_train_hf1 = scaler.transform(x_train_hf)
    x_train_lf1 = scaler.transform(x_train_lf)
    x_val1 = scaler.transform(x_val)

    # Train and evaluate model

    linear_m = LinearRegression()
    linear_m.fit(x_train_lf1, y_train_hf)
    mu0 = linear_m.predict(x_val)

    # Metrics
    y_pred = sp.special.inv_boxcox(np.array(mu0), lf_lambda).reshape(-1)
    y_true = sp.special.inv_boxcox(y_val, lf_lambda).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse_all, rmse_p5, rmse_p95 = rmses(y_pred, y_true)

    print('Mean R2 = ', r2)
    print('Mean RMSE = ', rmse_all)
    print('5th RMSE = ', rmse_p5)
    print('95th RMSE = ', rmse_p95)
