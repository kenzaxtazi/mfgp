#!/usr/bin/env python
# coding: utf-8

# # Appendix C

# In[ ]:


import sys
sys.path.append('/data/hpcdata/users/kenzi22')

from load import era5, data_dir, value
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import GPy
from tqdm import tqdm
import scipy as sp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from mfdgp.utils.metrics import mll, r2_low_vs_high


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import emukit
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays


### Prepare data


# Load data
minyear = 1980
maxyear = 2010
'''
gauge_df = value.all_gauge_data(minyear, maxyear, monthly=True)
station_names = gauge_df.drop_duplicates('name')['name']

# Get CV scheme
cv_locs = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/exp1/cv/exp1_cv_locs.npy')
cv_locs = cv_locs.reshape(-1, 2)

station_list = []
for loc in cv_locs:
    station_row = gauge_df[(gauge_df['latitude'] == loc[1]) | (gauge_df['longitude'] == loc[0])].iloc[0]
    station_list.append(station_row['name'])
station_arr = np.array(station_list)

# Split indexes
kf = KFold(n_splits=5)

cv_train_list = []
cv_test_list = []

for train_index, test_index in kf.split(station_arr):
    hf_train, hf_test = station_arr[train_index], station_arr[test_index]
    cv_train_list.append(hf_train)
    cv_test_list.append(hf_test)

# Split data
cv_x_train_hf = []
cv_y_train_hf = []
cv_x_train_lf = []
cv_y_train_lf = []
cv_x_val = []
cv_y_val = []
lf_lambdas = []


for i in range(len(cv_train_list)):

    hf_train_list = []
    for station in cv_train_list[i]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    val_list = []
    for station in cv_test_list[i]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.dropna().reset_index())
    val_df = pd.concat(val_list)

    era5_df = era5.value_gauge_download(
        list(cv_test_list[i]) + list(cv_train_list[i]), minyear=minyear, maxyear=maxyear)
    
    hf_train_df.sort_values(by='time', inplace=True)
    val_df.sort_values(by='time', inplace=True)
    lf_train_df = era5_df.reset_index().sort_values(by='time')
    
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
    y_train_lf = lf_train_df['tp_tr'].values.reshape(-1, 1)
    x_train_hf = hf_train_df[['time', 'latitude', 'longitude', 'altitude']].values.reshape(-1, 4)
    y_train_hf = hf_train_df[['tp_tr']].values.reshape(-1, 1)
    x_val = val_df[['time', 'latitude', 'longitude', 'altitude']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1, 1)

    # Scaling
    scaler = StandardScaler().fit(x_train_hf)
    x_train_hf1 = scaler.transform(x_train_hf)
    x_train_lf1 = scaler.transform(x_train_lf)
    x_val1 = scaler.transform(x_val)
    
    cv_x_train_hf.append(x_train_hf1)
    cv_y_train_hf.append(y_train_hf)
    cv_x_train_lf.append(x_train_lf1)
    cv_y_train_lf.append(y_train_lf)
    cv_x_val.append(x_val1)
    cv_y_val.append(y_val)
    lf_lambdas.append(lf_lambda)


np.save('data/cv_x_train_hf_value_1980-2010.npy', cv_x_train_hf)
np.save('data/cv_y_train_hf_value_1980-2010.npy', cv_y_train_hf)
np.save('data/cv_x_train_lf_value_1980-2010.npy', cv_x_train_lf)
np.save('data/cv_y_train_lf_value_1980-2010.npy', cv_y_train_lf)
np.save('data/cv_y_val_value_1980-2010.npy', cv_y_val)
np.save('data/cv_x_val_value_1980-2010.npy', cv_x_val)
np.save('data/lf_lambda_1980-2010.npy', lf_lambdas)

'''

### Load data
cv_x_train_hf = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_x_train_hf_value_1980-2010.npy', allow_pickle=True)
cv_y_train_hf = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_y_train_hf_value_1980-2010.npy', allow_pickle=True)
cv_x_train_lf = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_x_train_lf_value_1980-2010.npy', allow_pickle=True)
cv_y_train_lf = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_y_train_lf_value_1980-2010.npy', allow_pickle=True)
cv_x_val = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_x_val_value_1980-2010.npy', allow_pickle=True)
cv_y_val = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/cv_y_val_value_1980-2010.npy', allow_pickle=True)
lf_lambdas = np.load('/data/hpcdata/users/kenzi22/mfdgp/experiments/app_c/data/lf_lambda_1980-2010.npy', allow_pickle=True)




### MFDGP

R2_all = []

for t in tqdm([2, 4, 6, 8, 10, 12, 14, 16]):

    R2 = []

    for i in range(len(cv_x_train_hf)):

        tr = 12 * t

        # Input data
        X_train, Y_train = convert_xy_lists_to_arrays([cv_x_train_lf[i][:tr*28], cv_x_train_hf[i][:tr*28]], [
                                                      cv_y_train_lf[i][:tr*28], cv_y_train_hf[i][:tr*28]])

        # Train and evaluate
        kern1 = GPy.kern.Matern52(input_dim=4, ARD=True)
        kernels = [kern1, GPy.kern.Matern52(input_dim=4, ARD=True)]
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
            kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, lin_mf_kernel, n_fidelities=2,)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
        lin_mf_model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, 2, n_optimization_restarts=5)
        lin_mf_model.optimize()

        # Load and prep test data
        x_val1, y_val = cv_x_val[i][:tr*7], cv_y_val[i][:tr*7]
        n = x_val1.shape[0]
        x_met = convert_x_list_to_array([x_val1, x_val1])

        # ALL
        y_pred0, y_var0 = lin_mf_model.predict(x_met[n:])
        y_pred_low0, y_var_low0 = lin_mf_model.predict(x_met[:n])

        y_pred = sp.special.inv_boxcox(y_pred0, lf_lambdas[i]).reshape(-1)
        y_true = sp.special.inv_boxcox(y_val, lf_lambdas[i]).reshape(-1)
        R2.append(r2_score(y_true, y_pred))

    print('Mean R2 = ', np.mean(R2), '±', np.std(R2))
    R2_all.append(R2)
    np.savetxt('app_c_value.npy', R2)


# In[14]:


plt.figure()
R2_arr = np.array(R2_all)
for i in range(len([2, 4, 6, 8, 10, 12, 14, 16])):
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16])[i]
    plt.scatter(np.ones(5)*y, R2_arr[i])
    #plt.errorbar([0, 2, 4, 8, 15, 30], R2_arr[:, 0], yerr=R2_arr[:, 1],capsize=5, marker='.', linestyle='none')£
#reg = LinearRegression().fit(np.array([0, 2, 4, 8, 15, 30]).reshape(-1, 1), R2_arr[:, 0])
#y = reg.predict(np.linspace(0, 30, 10).reshape(-1, 1))
#plt.plot(np.linspace(0, 30, 10).reshape(-1, 1), y, linestyle='--')
plt.ylabel('$R^2$')
plt.xlabel('Years')
plt.savefig('appendix_C_plot.png', dpi=300)

