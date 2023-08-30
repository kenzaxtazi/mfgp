# Trend comparison


import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction')  # noqa

from load import beas_sutlej_gauges, era5, cru, beas_sutlej_wrf, gpm, aphrodite, data_dir
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
import pandas as pd


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
            #print(tp_ref.shape, tp.shape)
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


def gauge_stats_all(minyear: str, maxyear: str) -> tuple:
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

        #print(era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds)
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


def gauge_stats_cv(minyear: str, maxyear: str) -> tuple:
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

        #print(era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds)
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
