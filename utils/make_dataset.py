import pandas as pd
import scipy as sp
import numpy as np
import glob
import xarray as xr


directory = 'experiments/exp3/outputs_all/*'
paths = glob.glob(directory

# Info for inverse transform           
scaling_df = pd.read_csv('exp3/lambdas_1980_2013.csv', index_col=0)
lambdas = scaling_df[['lambdas']]

i = 0

for path in paths:
    print(path)
    df = pd.read_csv(path, index_col=0)
    
    # Transformed mean
    df['pred_tr'] = sp.special.inv_boxcox(df['pred0'], lambdas.iloc[i].values)
    i =+ 1
    
    # Transformed upper confidence interval
    pred_CI_upper = df['pred0'] + 1.96 * np.sqrt(df['y_var0'])
    df['pred_tr_CI_upper'] = sp.special.inv_boxcox(pred_CI_upper, lambdas.iloc[0].values)
    
    # Transformed lower confidence interval
    pred_CI_lower = df['pred0'] - 1.96 * np.sqrt(df['y_var0'])
    df['pred_tr_CI_lower'] = sp.special.inv_boxcox(pred_CI_lower, lambdas.iloc[0].values)
    
    p = path.split('/')
    df.to_csv(p[0] + '/tr_'+ p[1] + '/' + p[2])bdas.iloc[0].values)


tr_directory = 'experiments/exp3/outputs/tr_outputs_all'
tr_paths = glob.glob(tr_directory)
tr_df_all = pd.concat(map(pd.read_csv, tr_paths))
tr_df_all.drop(columns=['Unnamed: 0'], inplace=True)
tr_df_all.rename(columns={'elevation':'elev'}, inplace=True)
tr_df_all['date'] = pd.to_datetime(tr_df_all['time']).astype("datetime64[MS]")
tr_df_all.drop(columns='time', inplace=True)
tr_df_all.to_csv('experiments/exp3/mfgp_predictions_1980_2010.csv')

# to xr.Dataset
tr_df_all.set_index(['date', 'lat', 'lon'], inplace=True)
ds = tr_df_all.to_xarray()
ds.attrs['title'] = 'Downscaled ERA5 monthly precipitation data using Multi-Fidelity Gaussian Processes between 1980 and 2010 for the Upper Beas and Sutlej Basins, Himalayas'
ds.attrs['institutions'] = 'University of Cambridge, British Antactic Survey'
ds.to_netcdf('dataset_w_metadata/mfgp_predictions_1980_2010.nc')