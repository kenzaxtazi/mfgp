import sys
import torch
import gpytorch
import scipy as sp
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# custom libraries
from utils import metrics
from models.gpytorch_gp import GPRegressionModel
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa
from load import era5, value, data_dir  # noqa


# Load data

minyear = '2000'
maxyear = '2004-12-31'

cv_locs = np.load('experiments/exp1/cv/exp1_cv_locs.npy')
cv_locs = cv_locs.reshape(-1, 2)

gauge_df = value.all_gauge_data(minyear, maxyear, monthly=True)
station_names = gauge_df.drop_duplicates('name')['name']

station_list = []
for loc in cv_locs:
    station_row = gauge_df[(gauge_df['lat'] == loc[1]) | (
        gauge_df['lon'] == loc[0])].iloc[0]
    station_list.append(station_row['name'])
station_arr = np.array(station_list)

# Split into five chunks
kf = KFold(n_splits=5)

cv_train_list = []
cv_test_list = []

for train_index, test_index in kf.split(station_arr):
    [station_names.drop(station_names.loc[station_names == l].index,
                        inplace=True) for l in station_arr[test_index]]
    hf_train = station_names.values
    hf_test = station_arr[test_index]
    cv_train_list.append(hf_train)
    cv_test_list.append(hf_test)


# GP model loop

R2_all = []
RMSE_all = []
RMSE_p5 = []
RMSE_p95 = []
MLL = []

for j in range(len(cv_train_list)):

    hf_train_list = []
    for station in cv_train_list[j]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)
    hf_train_df['time'] = pd.to_datetime(hf_train_df['time'])
    hf_train_df['time'] = pd.to_numeric(hf_train_df['time'])

    val_list = []
    for station in cv_test_list[j]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.dropna().reset_index())
    val_df = pd.concat(val_list)
    val_df['time'] = pd.to_datetime(val_df['time'])
    val_df['time'] = pd.to_numeric(val_df['time'])

    era5_df = era5.value_gauge_download(
        list(cv_test_list[j]) + list(cv_train_list[j]), minyear=minyear, maxyear=maxyear)
    lf_train_df = era5_df.reset_index()
    lf_train_df['time'] = pd.to_datetime(lf_train_df['time'])
    lf_train_df['time'] = pd.to_numeric(lf_train_df['time'])

    # Prepare data

    # Transformations
    lf_train_df['tp_tr'], lf_lambda = sp.stats.boxcox(
        lf_train_df['tp'].values + 0.01)
    hf_train_df['tp_tr'] = sp.stats.boxcox(
        hf_train_df['tp'].values + 0.01, lmbda=lf_lambda)
    val_df['tp_tr'] = sp.stats.boxcox(
        val_df['tp'].values + 0.01, lmbda=lf_lambda)

    # Splitting
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df[['tp_tr']].values.reshape(-1, 1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1, 1)

    # Scaling
    scaler = StandardScaler().fit(x_train_hf)
    x_train_hf1 = scaler.transform(x_train_hf)
    x_val1 = scaler.transform(x_val)

    # Make tensors
    train_x_hf, train_y_hf = torch.Tensor(
        x_train_hf1), torch.Tensor(y_train_hf.reshape(-1))

    # GP training
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=torch.ones(len(train_x_hf)) * 0.01)
    model = GPRegressionModel(train_x_hf, train_y_hf, likelihood, kernel = 'custom')

    training_iter = 200
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x_hf)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y_hf)
        loss.backward()
        if i % 10 == 0:
            print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                # model.covar_module.base_kernel.lengthscale.item(),
                # model.likelihood.noise.item()
            ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        trained_pred_dist = likelihood(model(torch.Tensor(x_val1)))
        y_pred0 = trained_pred_dist.mean
        y_std0 = trained_pred_dist.stddev

    y_pred = sp.special.inv_boxcox(y_pred0, lf_lambda).reshape(-1)
    y_pred[np.isnan(y_pred)] = 0
    y_true = sp.special.inv_boxcox(y_val, lf_lambda).reshape(-1)
    R2_all.append(r2_score(y_true, y_pred))
    RMSE_all.append(mean_squared_error(y_true, y_pred, squared=False))

    # 5th PERCENTILE
    p5 = np.percentile(y_true, 5.0)
    indx = [y_true <= p5][0]
    x_val_p5 = x_val[indx, :]
    y_true_p5 = y_true[indx]
    y_pred_p5 = y_pred[indx]
    RMSE_p5.append(mean_squared_error(y_true_p5, y_pred_p5, squared=False))

    # 95th PERCENTILE
    p95 = np.percentile(y_true, 95.0)
    indx = [y_true >= p95][0]
    x_val_p95 = x_val[indx]
    y_true_p95 = y_true[indx]
    y_pred_p95 = y_pred[indx]
    RMSE_p95.append(mean_squared_error(y_true_p95, y_pred_p95, squared=False))

    # MLL
    ll = metrics.mll(y_val, y_pred0, y_std0)
    MLL.append(ll)

print('Mean RMSE = ', np.mean(RMSE_all), '±', np.std(RMSE_all))
print('Mean R2 = ', np.mean(R2_all), '±', np.std(R2_all))
print('5th RMSE = ', np.mean(RMSE_p5), '±', np.std(RMSE_p5))
print('95th RMSE = ', np.mean(RMSE_p95), '±', np.std(RMSE_p95))
print('MLL= ', np.mean(MLL), '±', np.std(MLL))

'''

### Linear regression loop

R2_all = []
RMSE_all = []
RMSE_p5 = []
RMSE_p95 = []

for i in range(len(cv_train_list)):

    hf_train_list = []
    for station in cv_train_list[i]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)
    hf_train_df['time'] = pd.to_datetime(hf_train_df['time'])
    hf_train_df['time'] = pd.to_numeric(hf_train_df['time'])

    val_list = []
    for station in cv_test_list[i]:
        station_ds = value.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.dropna().reset_index())
    val_df = pd.concat(val_list)
    val_df['time'] = pd.to_datetime(val_df['time'])
    val_df['time'] = pd.to_numeric(val_df['time'])

    era5_df = era5.value_gauge_download(
        list(cv_test_list[i]) + list(cv_train_list[i]), minyear=minyear, maxyear=maxyear)
    lf_train_df = era5_df.reset_index()
    lf_train_df['time'] = pd.to_datetime(lf_train_df['time'])
    lf_train_df['time'] = pd.to_numeric(lf_train_df['time'])

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
    scaler = MinMaxScaler().fit(x_train_hf)
    x_train_hf1 = scaler.transform(x_train_hf)
    x_train_lf1 = scaler.transform(x_train_lf)
    x_val1 = scaler.transform(x_val)

    linear_m = LinearRegression()
    linear_m.fit(x_train_lf1, y_train_lf)

    # ALL
    y_pred = sp.special.inv_boxcox(
        linear_m.predict(x_val1), lf_lambda).reshape(-1)
    y_true = sp.special.inv_boxcox(y_val, lf_lambda).reshape(-1)
    R2_all.append(r2_score(y_true, y_pred))
    RMSE_all.append(mean_squared_error(y_true, y_pred, squared=False))

    # 5th PERCENTILE
    p5 = np.percentile(y_true, 5.0)
    indx = [y_true <= p5][0]
    x_val_p5 = x_val[indx, :]
    y_true_p5 = y_true[indx]
    y_pred_p5 = y_pred[indx]
    RMSE_p5.append(mean_squared_error(y_true_p5, y_pred_p5, squared=False))

    # 95th PERCENTILE
    p95 = np.percentile(y_true, 95.0)
    indx = [y_true >= p95][0]
    x_val_p95 = x_val[indx]
    y_true_p95 = y_true[indx]
    y_pred_p95 = y_pred[indx]
    RMSE_p95.append(mean_squared_error(y_true_p95, y_pred_p95, squared=False))


print('Mean RMSE = ', np.mean(RMSE_all), '±', np.std(RMSE_all))
print('Mean R2 = ', np.mean(R2_all), '±', np.std(R2_all))
print('5th RMSE = ', np.mean(RMSE_p5), '±', np.std(RMSE_p5))
print('95th RMSE = ', np.mean(RMSE_p95), '±', np.std(RMSE_p95))
'''
