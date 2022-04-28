

import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

import GPy 

import emukit
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

# custom modules
from load import era5, value
from gp import data_prep as dp




stations = ['brocken', 'verona-villafranca', 'zugspitze', 'hohenpeissenberg', 
            'innsbruck', 'regensburg', 'roma-ciampino', 'sonnblick',
            'salzburg', 'potsdam']

lf_r2 = []
hf_r2 = []
lf_rmse = []
hf_rmse = []

for station in stations:
    
    hf_ds =  value.gauge_download(station, minyear=1980, maxyear=2011)
    lf_ds = era5.value_gauge_download(station, minyear=1980, maxyear=2011)
    
    # Transformation
    hf_ds['tp_tr'], hf_lambda = sp.stats.boxcox(hf_ds['tp'].values + 0.01)
    lf_ds['tp_tr'], lf_lambda = sp.stats.boxcox(lf_ds['tp'].values + 0.01)

    # High fidelity data needs to be smaller in length then low fidelity data
    x_train_l = lf_ds.time[:330].values.reshape(-1,1)
    x_train_h = hf_ds.time[:240].values.reshape(-1,1)
    y_train_l = lf_ds.tp_tr[:330].values.reshape(-1, 1)
    y_train_h = hf_ds.tp_tr[:240].values.reshape(-1,1)
    x_val = hf_ds.time[240:330].values.reshape(-1,1)
    y_val = hf_ds.tp_tr[240:330].values.reshape(-1,1)
    

    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

    # Model
    kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)

    # Train model
    lin_mf_model.optimize()
    
    # Metrics
    n = x_val.shape[0]
    y_true = sp.special.inv_boxcox(y_val, hf_lambda)
    
    # Predictions for validation space
    x_met = convert_x_list_to_array([x_val, x_val])
    lin_mf_l_y_pred, mf_l_y_std_pred = sp.special.inv_boxcox(lin_mf_model.predict(x_met[:n]), lf_lambda)
    lin_mf_h_y_pred, mf_h_y_std_pred = sp.special.inv_boxcox(lin_mf_model.predict(x_met[n:]), hf_lambda)
    
    # R2
    hf_r2.append(r2_score(y_true, lin_mf_h_y_pred))
    lf_r2.append(r2_score(y_true, lin_mf_l_y_pred))
    
    hf_rmse.append(mean_squared_error(y_true, lin_mf_h_y_pred, squared=False))
    lf_rmse.append(mean_squared_error(y_true, lin_mf_l_y_pred, squared=False))
    
    
print('lf r2 = ', np.mean(lf_r2))
print('hf r2 = ', np.mean(hf_r2))
print('lf rmse = ', np.mean(lf_rmse))
print('hf rmse = ', np.mean(hf_rmse))