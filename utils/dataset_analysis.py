# Dataset analyses

import sys
import torch
import gpytorch

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# custom libraries
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa
from load import beas_sutlej_gauges, era5, cru, value, beas_sutlej_wrf, gpm, aphrodite, data_dir
from models.gpytorch_gp import GPRegressionModel
from sklearn.preprocessing import StandardScaler


def dataset_stats(datasets, ref_ds=None, ret=False):
    """Print mean, standard deviations and slope for datasets."""

    r2_list = []
    rmse_list = []
    rmse_p5_list = []
    rmse_p95_list = []
    r2_p5_list = []
    r2_p95_list = []

    for ds in datasets:

        tp = ds.tp.values
        '''
        slope, _intercept, _r_value, _p_value, _std_err = stats.linregress(
            da.time.values, da.values)

        print(name)
        print('mean = ', np.mean(da.values), 'mm/day')
        print('std = ', np.std(da.values), 'mm/day')
        print('slope = ', slope, 'mm/day/year')
        '''
        if ref_ds is not None:
            tp_ref = ref_ds.tp.values
            # print(tp_ref.shape, tp.shape)
            df = pd.DataFrame({'tp_ref': tp_ref, 'tp': tp})
            df = df.dropna()

            y_true = df['tp_ref'].values
            y_pred = df['tp'].values

            # all values
            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            # 5th percentile
            p5 = np.percentile(y_true, 5.0)
            indx = [y_true <= p5][0]
            y_true_p5 = y_true[indx]
            y_pred_p5 = y_pred[indx]
            r2_p5 = r2_score(y_true_p5, y_pred_p5)
            rmse_p5 = mean_squared_error(y_true_p5, y_pred_p5, squared=False)

            # 95th percentile
            p95 = np.percentile(y_true, 95.0)
            indx = [y_true >= p95][0]
            y_true_p95 = y_true[indx]
            y_pred_p95 = y_pred[indx]
            r2_p95 = r2_score(y_true_p95, y_pred_p95)
            rmse_p95 = mean_squared_error(
                y_true_p95, y_pred_p95, squared=False)

            # Print and append
            '''
            print('R2 = ', r2)
            print('RMSE = ', rmse)
            print('R2 p5 = ', r2_p5)
            print('RMSE p5 = ', rmse_p5)
            print('R2 p95 = ', r2_p95)
            print('RMSE p95 = ', rmse_p95)
            '''
            r2_list.append(r2)
            rmse_list.append(rmse)
            rmse_p5_list .append(rmse_p5)
            rmse_p95_list.append(rmse_p95)
            r2_p5_list .append(r2_p5)
            r2_p95_list .append(r2_p95)

    if ret is True:
        return [r2_list, rmse_list, r2_p5_list, rmse_p5_list, r2_p95_list, rmse_p95_list]


def bs_gauge_stats_all(minyear: str, maxyear: str) -> tuple:
    """
    Statisitics for gauges in the Uppper Beas and Sutlej Basins

    Print mean, standard deviations and slope for datasets.
    As well as calculating R2 and average RMSE, 5th percentile
    RMSE and 95th percentile RMSE for each reference dataset.
    These are in order:
     - ERA5
     - GPM
     - APHRODITE
     - CRU
     - Bias-corrected WRF

    Args:
        minyear(str): minimum year to analyse (inclusive)
        maxyear(str): maximum year to analyse (exclusive)

    Returns:
        tuple: list of averages and standerd deviations for each metric
    """

    bs_station_df = pd.read_csv(
        data_dir + '/bs_gauges/bs_only_gauge_info.csv')
    bs_station_df = bs_station_df.set_index('Unnamed: 0')
    station_list = list(bs_station_df.T)

    r2_list = []
    rmse_list = []
    rmse_p5_list = []
    rmse_p95_list = []
    r2_p5_list = []
    r2_p95_list = []

    for s in tqdm(station_list):

        gauge_ds = beas_sutlej_gauges.gauge_download(
            s, minyear=minyear, maxyear=maxyear)

        location = bs_station_df.loc[s].values
        # print(location)

        aphro_ds = aphrodite.collect_APHRO(location, minyear, maxyear)
        cru_ds = cru.collect_CRU(location, minyear, maxyear)
        era5_ds = era5.collect_ERA5(location, minyear, maxyear)
        gpm_ds = gpm.collect_GPM(location,  minyear, maxyear)
        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear, maxyear)

        timeseries = [era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds]

        # print(era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds)
        # Function to calculate statistics for each dataset and print values
        r2s, rmses, r2_p5, rmses_p5, r2_p95, rmses_p95 = dataset_stats(
            timeseries, ref_ds=gauge_ds, ret=True)
        r2_list.append(r2s)
        rmse_list.append(rmses)
        rmse_p5_list.append(rmses_p5)
        rmse_p95_list.append(rmses_p95)
        r2_p5_list.append(r2_p5)
        r2_p95_list.append(r2_p95)

    avg_r2 = np.array(r2_list).mean(axis=0)
    avg_rmse = np.array(rmse_list).mean(axis=0)
    avg_r2_p5 = np.array(r2_p5_list).mean(axis=0)
    avg_rmse_p5 = np.array(rmse_p5_list).mean(axis=0)
    avg_r2_p95 = np.array(r2_p95_list).mean(axis=0)
    avg_rmse_p95 = np.array(rmse_p95_list).mean(axis=0)

    std_r2 = np.array(r2_list).std(axis=0)
    std_rmse = np.array(rmse_list).std(axis=0)
    std_r2_p5 = np.array(r2_p5_list).std(axis=0)
    std_rmse_p5 = np.array(rmse_p5_list).std(axis=0)
    std_r2_p95 = np.array(r2_p95_list).std(axis=0)
    std_rmse_p95 = np.array(rmse_p95_list).std(axis=0)

    avgs = [avg_r2, avg_rmse, avg_r2_p5, avg_rmse_p5, avg_r2_p95, avg_rmse_p95]
    stds = [std_r2, std_rmse, std_r2_p5, std_rmse_p5, std_r2_p95, std_rmse_p95]

    return avgs, stds


def bs_gauge_stats_cv(minyear: str, maxyear: str) -> tuple:
    """
    Statisitics for gauges in the Uppper Beas and Sutlej Basins

    Print mean, standard deviations and slope for datasets.
    As well as calculating R2 and average RMSE, 5th percentile
    RMSE and 95th percentile RMSE for each reference dataset.
    These are in order:
     - ERA5
     - GPM
     - APHRODITE
     - CRU
     - Bias-corrected WRF

    Args:
        minyear(str): minimum year to analyse (inclusive)
        maxyear(str): maximum year to analyse (exclusive)

    Returns:
        tuple: list of averages and standard deviations for each metric
    """
    all_station_dict = pd.read_csv(
        data_dir + 'bs_gauges/gauge_info.csv', index_col='station')

    # Test locations from CV
    cv_locs = np.load('experiments/exp2/cv/cv_locs.npy')
    cv_locs = cv_locs.reshape(-1, 2)

    station_list = []
    for loc in cv_locs:
        station_row = all_station_dict[(all_station_dict['lat'] == loc[1]) | (
            all_station_dict['lon'] == loc[0])]
        station_list.append(str(np.array(station_row.index[0])))

    r2_list = []
    rmse_list = []
    rmse_p5_list = []
    rmse_p95_list = []
    r2_p5_list = []
    r2_p95_list = []

    for s in tqdm(station_list):

        gauge_ds = beas_sutlej_gauges.gauge_download(
            s, minyear=minyear, maxyear=maxyear)

        location = all_station_dict.loc[s].values[:2]  # lat, lon only
        # print(location)

        aphro_ds = aphrodite.collect_APHRO(location, minyear, maxyear)
        cru_ds = cru.collect_CRU(location, minyear, maxyear)
        era5_ds = era5.collect_ERA5(location, minyear, maxyear)
        gpm_ds = gpm.collect_GPM(location,  minyear, maxyear)
        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear, maxyear)

        timeseries = [era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds]

        # print(era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds)
        # Function to calculate statistics for each dataset and print values
        r2s, rmses, r2_p5, rmses_p5, r2_p95, rmses_p95 = dataset_stats(
            timeseries, ref_ds=gauge_ds, ret=True)
        r2_list.append(r2s)
        rmse_list.append(rmses)
        rmse_p5_list.append(rmses_p5)
        rmse_p95_list.append(rmses_p95)
        r2_p5_list.append(r2_p5)
        r2_p95_list.append(r2_p95)

    avg_r2 = np.array(r2_list).mean(axis=0)
    avg_rmse = np.array(rmse_list).mean(axis=0)
    avg_r2_p5 = np.array(r2_p5_list).mean(axis=0)
    avg_rmse_p5 = np.array(rmse_p5_list).mean(axis=0)
    avg_r2_p95 = np.array(r2_p95_list).mean(axis=0)
    avg_rmse_p95 = np.array(rmse_p95_list).mean(axis=0)

    std_r2 = np.array(r2_list).std(axis=0)
    std_rmse = np.array(rmse_list).std(axis=0)
    std_r2_p5 = np.array(r2_p5_list).std(axis=0)
    std_rmse_p5 = np.array(rmse_p5_list).std(axis=0)
    std_r2_p95 = np.array(r2_p95_list).std(axis=0)
    std_rmse_p95 = np.array(rmse_p95_list).std(axis=0)

    avgs = [avg_r2, avg_rmse, avg_r2_p5, avg_rmse_p5, avg_r2_p95, avg_rmse_p95]
    stds = [std_r2, std_rmse, std_r2_p5, std_rmse_p5, std_r2_p95, std_rmse_p95]

    return avgs, stds


def europe_vs_bs_precip_stats(minyear: str, maxyear: str):
    """
    Comparison of precipiation statistics between Europe and
    the Upper Beas and Sutlej Basins.

    Args:
        minyear(str): minimum year to analyse (inclusive)
        maxyear(str): maximum year to analyse (exclusive)
    """

    value_gauge_df = value.all_gauge_data(minyear, maxyear, monthly=True)

    bs_gauge_ds = beas_sutlej_gauges.all_gauge_data(minyear, maxyear)
    bs_gauge_df = bs_gauge_ds.to_dataframe()
    bs_gauge_df = bs_gauge_df.stack().reset_index()
    bs_gauge_df = bs_gauge_df.rename(
        {"level_0": 'time', "level_1": "station_id", 0: "tp"}, axis=1)

    value_tp = value_gauge_df['tp'].values
    bs_tp = bs_gauge_df['tp'].values

    # 1st order stats
    print('VALUE')
    print('mean = ', np.mean(value_tp), 'mm/day')
    print('std = ', np.std(value_tp), 'mm/day')
    print('5th percentile = ', np.percentile(value_tp, 5), 'mm/day')
    print('95th percentile = ', np.percentile(value_tp, 95), 'mm/day')

    print('BS')
    print('mean = ', np.mean(bs_tp), 'mm/day')
    print('std = ', np.std(bs_tp), 'mm/day')
    print('5th percentile = ', np.percentile(bs_tp, 5), 'mm/day')
    print('95th percentile = ', np.percentile(bs_tp, 95), 'mm/day')


def europe_vs_bs_precip_lengthscales(minyear: str, maxyear: str):

    value_gauge_df = value.all_gauge_data(minyear, maxyear, monthly=True)
    value_gauge_df['time'] = pd.to_datetime(value_gauge_df['time'])
    value_gauge_df['time'] = pd.to_numeric(
        value_gauge_df['time'])  # / (365*24*60*60*1e9) + 1970 to get value in years

    all_station_dict = pd.read_csv(
        data_dir + 'bs_gauges/gauge_info.csv', index_col='station')
    bs_gauge_list = []
    for station in all_station_dict.index:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        bs_gauge_list.append(station_ds.to_dataframe().dropna().reset_index())
    bs_gauge_df = pd.concat(bs_gauge_list)
    bs_gauge_df['time'] = pd.to_datetime(bs_gauge_df['time'])
    bs_gauge_df['time'] = pd.to_numeric(
        bs_gauge_df['time'])

    datasets = [value_gauge_df, bs_gauge_df]
    training_iter = 200

    lengthscale_list = []

    for df in datasets:
        tp = df['tp'].values

        # VALUE
        # Prep data
        df['tp_tr'], _ = sp.stats.boxcox(tp + 0.01)
        x_value_0 = df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
        y_value = df['tp_tr'].values.reshape(-1)

        scaler = StandardScaler().fit(x_value_0)
        x_value = scaler.transform(x_value_0)

        # Make tensors
        train_x_value, train_y_value = torch.Tensor(
            x_value.reshape(-1, 4)), torch.Tensor(y_value.reshape(-1))

        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.ones(len(train_x_value)) * 0.01)
        model = GPRegressionModel(
            train_x_value, train_y_value, likelihood, custom=False)

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
            output = model(train_x_value)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y_value)
            loss.backward()
            if i % 10 == 0:
                print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    # model.covar_module.base_kernel.lengthscale.item(),
                    # model.likelihood.noise.item()
                ))
            optimizer.step()

        lengthscales_tr = model.covar_module.base_kernel.lengthscale.detach().numpy()
        print(lengthscales_tr)
        lengthscales = lengthscales_tr * \
            df[['time', 'lat', 'lon', 'z']].std().values
        lengthscale_list.append(lengthscales)

    return lengthscale_list
