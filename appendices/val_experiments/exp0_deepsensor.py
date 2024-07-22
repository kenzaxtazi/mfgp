
import sys  # noqa
filepath = '/Users/kenzatazi/Documents/CDT/Code'  # noqa
sys.path.append(filepath)  # noqa

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from load import era5, data_dir, value
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import deepsensor.torch
from deepsensor.model import ConvNP
from deepsensor.train import Trainer
from deepsensor.data import DataProcessor, TaskLoader


### Prepare data

# Load data
minyear = '2000'
maxyear = '2004-12-31'

# Get CV scheme
cv_locs = np.load(
    filepath + '/mfgp/experiments/exp1/cv/exp1_cv_locs.npy')
cv_locs = cv_locs.reshape(-1, 2)

gauge_df = value.all_gauge_data(minyear, maxyear, monthly=True)
station_names = gauge_df.drop_duplicates('name')['name']

station_list = []
for loc in cv_locs:
    station_row = gauge_df[(gauge_df['lat'] == loc[1]) | (
        gauge_df['lon'] == loc[0])].iloc[0]
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

cv_train_hf = []
cv_train_lf = []
cv_val = []
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

    # Standardise time
    lf_train_df['time'] = pd.to_datetime(lf_train_df['time'])
    hf_train_df['time'] = pd.to_datetime(hf_train_df['time'])
    val_df['time'] = pd.to_datetime(val_df['time'])
    
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
    
    cv_train_hf.append(hf_train_df)
    cv_train_lf.append(lf_train_df)
    cv_val.append(val_df)
    lf_lambdas.append(lf_lambda)

    
    
### Training
def gen_tasks(dates, progress=True):
    tasks = []
    for date in tqdm(dates, disable=not progress):
        N_c = np.random.randint(0, 500)
        task = task_loader(date, context_sampling=["all", "all"], target_sampling="all")
        tasks.append(task)
    return tasks

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))

dates = pd.date_range(lf_train_df.time.values.min(), lf_train_df.time.values.max(), freq='MS')


# ConvCNP
R2_all = []
RMSE_all = []
RMSE_p5 = []
RMSE_p95 = []
MSLL = []

for i in range(5):

    # Prepare data
    lf_train_ds = cv_train_lf[i][['tp_tr', 'time', 'lat', 'lon',]].set_index(['time', 'lat', 'lon']).to_xarray()
    z_train_ds = cv_train_lf[i][['z', 'time', 'lat', 'lon',]].set_index(['time', 'lat', 'lon']).to_xarray()

    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    lf_ds = data_processor(lf_train_ds)
    hf_df = data_processor(cv_train_hf[i][['tp_tr', 'time', 'lat', 'lon',]].set_index(['time', 'lat', 'lon']))
    z_ds = data_processor(z_train_ds)

    task_loader = TaskLoader(context=[lf_ds, z_ds], target=hf_df)

    # Setup model
    model = ConvNP(data_processor, task_loader, internal_density=500)

    losses = []
    val_rmses = []
    val_tasks = gen_tasks(dates[1::2])

    # Train model
    val_rmse_best = np.inf
    trainer = Trainer(model, lr=5e-5)
    for epoch in tqdm(range(1)):
        train_tasks = gen_tasks(dates[::2])
        batch_losses = trainer(train_tasks)
        losses.append(np.mean(batch_losses))
        val_rmses.append(compute_val_rmse(model, val_tasks))
        if val_rmses[-1] < val_rmse_best:
            val_rmse_best = val_rmses[-1]
            model.save('convcnp_test_'+str(i))

    # Prepare loader
    test_tasks = task_loader(dates)
    xval = cv_val[i][['lat', 'lon']].drop_duplicates().values.transpose()
   
    # Predictions
    y_pred_df = model.predict(test_tasks, X_t=xval)
    y_pred0 = np.array(y_pred_df["tp_tr"].values[:, 0])
    y_var0 = np.array(y_pred_df["tp_tr"].values[:, 1])

    y_pred0 = y_pred0.astype(np.float64)
    y_var0 = y_var0.astype(np.float64)
    
    y_pred = sp.special.inv_boxcox(y_pred0, lf_lambda)
    y_true = y_val.reshape(-1)

    # R2 and RMSE
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

    # MSLL
    ll = mll(y_val, y_pred0, y_var0)
    MSLL.append(ll)

# Print metrics

print('Mean RMSE = ', np.mean(RMSE_all), '±', np.std(RMSE_all))
print('Mean R2 = ', np.mean(R2_all), '±', np.std(R2_all))
print('5th RMSE = ', np.mean(RMSE_p5), '±', np.std(RMSE_p5))
print('95th RMSE = ', np.mean(RMSE_p95), '±', np.std(RMSE_p95))
print('MSLL= ', np.mean(MSLL), '±', np.std(MSLL))
