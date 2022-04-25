import sys
sys.path.append("/home/users/ktazi")
sys.path.append("/home/users/kenzi22")

import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

# custom modules
from load import era5, value
from models import linear_mfdgp, nonlinear_mfdgp



######## Download and format data
train_stations = ['jackvik', 'leba', 'visby', 'corfu', 'klaipeda', 'arad', 'methoni', 
                 'siedlce', 'larissa', 'jokioinen-jokioisten', 'lazdijai', 
                 'kaunas', 'haparanda', 'sibiu', 'birzai', 'helsinki-kaisaniemi', 
                 'siikajoki-revonlahti', 'karasjok', 'jyvaskyla-lentoasema',
                 'bucuresti-baneasa', 'sodankyla', 'iasi', 'constanta', 'vardoe']

val_stations = ['brocken', 'verona-villafranca', 'zugspitze', 'hohenpeissenberg', 
                'innsbruck', 'regensburg', 'roma-ciampino', 'sonnblick',
                'salzburg', 'potsdam']

hf_train_list = []
for station in train_stations:
    station_df = value.gauge_download(station, minyear=1980, maxyear=2011)
    hf_train_list.append(station_df)
hf_train_df = pd.concat(hf_train_list)

hf_val_list = []
for station in val_stations:
    station_df = value.gauge_download(station, minyear=1980, maxyear=2011)
    hf_val_list.append(station_df)
hf_val_df = pd.concat(hf_val_list)

lf_train_stations = train_stations + val_stations 
lf_df = era5.value_gauge_download(lf_train_stations, minyear=1980, maxyear=2011)
lf_train_df = lf_df.reset_index()

# Transformations    
hf_train_df['tp_tr'], hf_lambda = sp.stats.boxcox(hf_train_df['tp'].values + 0.01)
lf_train_df['tp_tr'], lf_lambda = sp.stats.boxcox(lf_train_df['tp'].values + 0.01)

# To arrays
hf_x_train = hf_train_df[['time', 'lon', 'lat', 'alt']].values.reshape(-1,4)
hf_y_train = hf_train_df['tp_tr'].values.reshape(-1,1)
lf_x_train = lf_train_df[['time', 'lon', 'lat', 'z']].values.reshape(-1,4)
lf_y_train = lf_train_df['tp_tr'].values.reshape(-1,1)
x_val = hf_val_df[['time', 'lon', 'lat', 'alt']].values.reshape(-1,4)
y_val = hf_val_df['tp'].values.reshape(-1,1)

X_train, Y_train = convert_xy_lists_to_arrays([lf_x_train, hf_x_train,], [lf_y_train, hf_y_train])


####### Models
lm = linear_mfdgp(X_train, Y_train, dims=4)
lm.optimize()
nlm = nonlinear_mfdgp(X_train, Y_train, dims=4)
nlm.optimize()


###### Metrics
n = x_val.shape[0]

# Predictions for validation space
x_met = convert_x_list_to_array([x_val, x_val])
lm_l_y_pred, lm_l_std_pred = sp.special.inv_boxcox(lm.predict(x_met[:n]), lf_lambda)
lm_h_y_pred, lm_h_std_pred = sp.special.inv_boxcox(lm.predict(x_met[n:]), hf_lambda)

# Predictions for validation space
x_met = convert_x_list_to_array([x_val, x_val])
nlm_l_y_pred, nlm_l_std_pred = sp.special.inv_boxcox(nlm.predict(x_met[:n]), lf_lambda)
nlm_h_y_pred, nlm_h_std_pred = sp.special.inv_boxcox(nlm.predict(x_met[n:]), hf_lambda)


# R2
lm_h_r2 = r2_score(y_val, lm_h_y_pred)
lm_l_r2 = r2_score(y_val, lm_l_y_pred)
nlm_h_r2 = r2_score(y_val, nlm_h_y_pred)
nlm_l_r2 = r2_score(y_val, nlm_h_y_pred)

# RMSE
lm_h_rmse = mean_squared_error(y_val, lm_h_y_pred, squared=False)
lm_l_rmse = mean_squared_error(y_val, lm_l_y_pred, squared=False)
nlm_h_rmse = mean_squared_error(y_val, nlm_h_y_pred, squared=False)
nlm_l_rmse = mean_squared_error(y_val, nlm_l_y_pred, squared=False)

# Write to csv
with open('value_multiloc_test_alt.txt', 'w') as f:
    f.write('Nonlinear MFDGP high RMSE = ', nlm_h_rmse,
            'Nonlinear MFDGP low RMSE = ', nlm_l_rmse,
            'Linear MFDGP high R2 = ', lm_h_r2,
            'Linear MFDGP low R2 = ', lm_l_r2,
            'Nonlinear MFDGP high R2 = ', nlm_h_r2,
            'Nonlinear MFDGP low R2 = ', nlm_l_r2)