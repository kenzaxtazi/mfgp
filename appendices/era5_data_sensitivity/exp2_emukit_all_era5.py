import sys
sys.path.append('/data/hpcdata/users/kenzi22')

from load import era5, data_dir, beas_sutlej_gauges
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


# Set seed
import random
random.seed(3)

# Date range
minyear = 2000
maxyear = 2005

# Load cross-validation scheme
all_station_dict = pd.read_csv(data_dir + 'bs_gauges/gauge_info.csv', index_col='station')
cv_locs = np.load('/data/hpcdata/users/kenzi22/mfgp/experiments/exp2/cv/cv_locs.npy')
cv_locs = cv_locs.reshape(-1, 2)

station_list = []
for loc in cv_locs:
    station_row = all_station_dict[(all_station_dict['lat'] == loc[1]) | (all_station_dict['lon'] == loc[0])]
    station_list.append(str(np.array(station_row.index[0])))

station_arr = np.array(station_list)

# Split into five chunks
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
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.to_dataframe().dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    val_list = []
    for station in cv_test_list[i]:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.to_dataframe().dropna().reset_index())
    val_df = pd.concat(val_list)

    # era5.collect_ERA5('indus', minyear=minyear, maxyear=maxyear)
    era5_df = era5.gauges_download(
        list(cv_test_list[i]) + list(cv_train_list[i]), minyear=minyear, maxyear=maxyear)

    lf_train_df = era5_df
    
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
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df[['tp_tr']].values.reshape(-1, 1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
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

np.save('data/cv_x_train_hf_bs_2000-2005.npy', cv_x_train_hf)
np.save('data/cv_y_train_hf_bs_2000-2005.npy', cv_y_train_hf)
np.save('data/cv_x_train_lf_bs_2000-2005.npy', cv_x_train_lf)
np.save('data/cv_y_train_lf_bs_2000-2005.npy', cv_y_train_lf)
np.save('data/cv_y_val_bs_2000-2005.npy', cv_y_val)
np.save('data/cv_x_val_bs_2000-2005.npy', cv_x_val)
np.save('data/lf_lambda_2000-2005.npy', lf_lambdas)


cv_x_train_hf = np.load('data/cv_x_train_hf_bs_2000-2005.npy', allow_pickle=True)
cv_y_train_hf = np.load('data/cv_y_train_hf_bs_2000-2005.npy', allow_pickle=True)
cv_x_train_lf = np.load('data/cv_x_train_lf_bs_2000-2005.npy', allow_pickle=True)
cv_y_train_lf = np.load('data/cv_y_train_lf_bs_2000-2005.npy', allow_pickle=True)
cv_x_val = np.load('data/cv_x_val_bs_2000-2005.npy', allow_pickle=True)
cv_y_val = np.load('data/cv_y_val_bs_2000-2005.npy', allow_pickle=True)
lf_lambdas = np.load('data/lf_lambda_2000-2005.npy', allow_pickle=True)

R2_all = []
RMSE_all = []
RMSE_p5 = []
RMSE_p95 = []
MSLL = []

R2_all_low = []
RMSE_all_low = []
RMSE_p5_low = []
RMSE_p95_low = []
MSLL_low = []

r2_high_list = []
r2_low_list = []

for i in range(len(cv_train_list)):

    # Input data
    X_train, Y_train = convert_xy_lists_to_arrays([cv_x_train_lf[i], cv_x_train_hf[i]], [cv_y_train_lf[i], cv_y_train_hf[i]])

    # Train and evaluate
    kern1 = GPy.kern.Matern52(input_dim=4, ARD=True)
    kernels = [kern1, GPy.kern.Matern52(input_dim=4, ARD=True)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2,)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()
    #print(gpy_lin_mf_model.multifidelity.rbf.lengthscale)
    
    #with open('model_instances/bs_mfdgp' + str(i) + '.pkl', 'wb') as file:
    #    pickle.dump(lin_mf_model, file)
    
    # Load and prep test data                                     
    x_val1, y_val = cv_x_val[i], cv_y_val[i]                                                                                   
    n = x_val1.shape[0]
    x_met = convert_x_list_to_array([x_val1, x_val1])
    
    # ALL
    y_pred0, y_var0 = lin_mf_model.predict(x_met[n:])
    y_pred_low0, y_var_low0 = lin_mf_model.predict(x_met[:n])
    
    y_pred = sp.special.inv_boxcox(y_pred0, lf_lambdas[i]).reshape(-1)
    y_true = sp.special.inv_boxcox(y_val, lf_lambdas[i]).reshape(-1)
    R2_all.append(r2_score(y_true, y_pred))
    RMSE_all.append(mean_squared_error(y_true, y_pred, squared=False))
    
    y_pred_low = sp.special.inv_boxcox(y_pred_low0, lf_lambdas[i]).reshape(-1)
    R2_all_low.append(r2_score(y_true, y_pred_low))
    RMSE_all_low.append(mean_squared_error(y_true, y_pred_low, squared=False))

    # 5th PERCENTILE
    p5 = np.percentile(y_true, 5.0)
    indx = [y_true <= p5][0]
    x_val_p5 = x_val1[indx, :]
    y_true_p5 = y_true[indx]
    y_pred_p5 = y_pred[indx]
    y_pred_p5_low = y_pred_low[indx]
    RMSE_p5.append(mean_squared_error(y_true_p5, y_pred_p5, squared=False))
    RMSE_p5_low.append(mean_squared_error(y_true_p5, y_pred_p5_low, squared=False))

    # 95th PERCENTILE
    p95 = np.percentile(y_true, 95.0)
    indx = [y_true >= p95][0]
    x_val_p95 = x_val1[indx]
    y_true_p95 = y_true[indx]
    y_pred_p95 = y_pred[indx]
    y_pred_p95_low = y_pred_low[indx]
    RMSE_p95.append(mean_squared_error(y_true_p95, y_pred_p95, squared=False))
    RMSE_p95_low.append(mean_squared_error(y_true_p95, y_pred_p95_low, squared=False))
                        
    # MSLL
    ll = mll(y_val, y_pred0, y_var0)
    ll_low = mll(y_val, y_pred_low0, y_var_low0)
    MSLL.append(ll)
    MSLL_low.append(ll_low)
    
    r2_high, r2_low = r2_low_vs_high(y_pred_low, y_pred, x_val1, y_true)
    r2_high_list.append(r2_high)
    r2_low_list.append(r2_low)

print('Mean RMSE = ', np.mean(RMSE_all), '±', np.std(RMSE_all))
print('Mean R2 = ', np.mean(R2_all), '±', np.std(R2_all))
print('5th RMSE = ', np.mean(RMSE_p5), '±', np.std(RMSE_p5))
print('95th RMSE = ', np.mean(RMSE_p95), '±', np.std(RMSE_p95))
print('MSLL= ', np.mean(MSLL), '±', np.std(MSLL))
                        
print('Mean RMSE = ', np.mean(RMSE_all_low), '±', np.std(RMSE_all_low))
print('Mean R2 = ', np.mean(R2_all_low), '±', np.std(R2_all_low))
print('5th RMSE = ', np.mean(RMSE_p5_low), '±', np.std(RMSE_p5_low))
print('95th RMSE = ', np.mean(RMSE_p95_low), '±', np.std(RMSE_p95_low))
print('MSLL= ', np.mean(MSLL_low), '±', np.std(MSLL_low))


np.save('exp2_ypred_lf_all_era5_r2_2000_2005.npy', np.array(r2_low_list))
np.save('exp2_ypred_hf_all_era5_r2_2000_2005.npy', np.array(r2_high_list))