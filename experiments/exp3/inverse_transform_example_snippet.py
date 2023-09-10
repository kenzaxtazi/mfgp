import pandas as pd
import numpy as np
import scipy as sp

# Read in the data
df = pd.read_csv('mfgp_predictions_1980_2010.csv', index_col=0)

# Read in the scaling factors
scaling_df = pd.read_csv('lambdas_1980_2010.csv', index_col=0)

# For 1980
df_subset = df.set_index('time')['1980-01-01':'1981-01-01']
lambda_1980 = scaling_df[scaling_df['year'] == 1980]['lambdas'].values[0]

# Mean
df_subset['pred_tr'] = sp.special.inv_boxcox(df_subset['pred0'], lambda_1980)

# Upper 95% confidence interval bound
pred_CI_upper = df_subset['pred0'] + 1.96 * np.sqrt(df_subset['y_var0'])
df_subset['pred_tr_CI_upper'] = sp.special.inv_boxcox(
    pred_CI_upper, lambda_1980)

# Lower 95% confidence interval bound
pred_CI_lower = df_subset['pred0'] - 1.96 * np.sqrt(df_subset['y_var0'])
df_subset['pred_tr_CI_lower'] = sp.special.inv_boxcox(
    pred_CI_lower, lambda_1980)
