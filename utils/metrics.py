import scipy as sp
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def rmses(y_pred, y_true):
    rmse_all = mean_squared_error(y_true, y_pred, squared=False)

    # 5th PERCENTILE
    p5 = np.percentile(y_true, 5.0)
    indx = [y_true <= p5][0]
    y_true_p5 = y_true[indx]
    y_pred_p5 = y_pred[indx]
    rmse_p5 = mean_squared_error(y_true_p5, y_pred_p5, squared=False)

    # 95th PERCENTILE
    p95 = np.percentile(y_true, 95.0)
    indx = [y_true >= p95][0]
    y_true_p95 = y_true[indx]
    y_pred_p95 = y_pred[indx]
    rmse_p95 = mean_squared_error(y_true_p95, y_pred_p95, squared=False)

    return rmse_all, rmse_p5, rmse_p95


def r2_low_vs_high(mu0: np.array, mu2: np.array, x_val1: np.array, lmbda: float):
    """ 
    Calculate R2 values for different locations in validation set.
    
    mu0 : mean posterior distribution for low fidelity
    mu2 : mean posterior distribution for high fidelity
    x_val1 : normalised x values
    lmbda : target variable scaling factor
    """

    val_df1 = pd.DataFrame(x_val1, columns=['time', 'lon', 'lat', 'z'])
    val_df1['mu2'] = mu2
    val_df1['mu0'] = mu0
    val_df1['tp_tr'] = y_val
    val_df1s = [x for _, x in val_df1.groupby(['lon', 'lat', 'z'])]

    R2_hf = []
    R2_lf = []

    for df in val_df1s:
        x_val = df[['time', 'lon', 'lat', 'z']].values.reshape(-1,4)
        y_val = df['tp_tr'].values.reshape(-1)
        y_pred0_lf = df['mu0'].values.reshape(-1)
        y_pred0_hf = df['mu2'].values.reshape(-1)
        # ALL
        y_pred_lf = sp.special.inv_boxcox(y_pred0_lf, lmbda).reshape(-1)
        y_pred_hf = sp.special.inv_boxcox(y_pred0_hf, lmbda).reshape(-1)
        y_true = sp.special.inv_boxcox(y_val, lmbda).reshape(-1)
        R2_hf.append(r2_score(y_true, y_pred_hf))
        R2_lf.append(r2_score(y_true, y_pred_lf))

    np.savetxt('table3_ypred_lf_r2_2000-2010.csv', R2_lf)
    np.savetxt('table3_ypred_hf_r2_2000-2010.csv', R2_hf)
