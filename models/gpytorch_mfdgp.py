from load import beas_sutlej_gauges, era5
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import gpytorch
import torch
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics import r2_score

import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')

base_kernel = gpytorch.kernels.RBFKernel(
    ard_num_dims=4, active_dims=[0, 1, 2, 3])


class LF_gp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LF_gp, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_nonlin_gp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(HF_nonlin_gp, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(1, active_dims=[4])
                                                         * base_kernel + base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_lin_gpmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(HF_lin_gpmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(1, active_dims=[4])
                                                         + base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_custom_gpmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(HF_custom_gpmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(1, active_dims=[4])
                                                         * base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_first_lvl(x_train_lf, y_train_lf, training_iter):

    likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
    m1 = LF_gp(x_train_lf, y_train_lf, likelihood1)

    # Find optimal model hyperparameters
    m1.train()
    likelihood1.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer1 = torch.optim.Adam(m1.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood1, m1)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer1.zero_grad()
        # Output from model
        output1 = m1(x_train_lf)
        # Calc loss and backprop gradients
        loss1 = -mll1(output1, y_train_lf)
        loss1.backward()
        if i % 10 == 1:
            print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss1.item(),
                # model.covar_module.base_kernel.lengthscale.item(),
                # model.likelihood.noise.item()
            ))
        optimizer1.step()

    return m1, likelihood1


def evaluate_first_lvl(m1, likelihood1, x_train_hf):
    # Get into evaluation (predictive posterior) mode
    m1.eval()
    likelihood1.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        trained_pred_dist1 = likelihood1(m1(x_train_hf))
        mu1 = trained_pred_dist1.mean
        v1 = trained_pred_dist1.variance
    return mu1, v1


def train_second_lvl(x_train_hf, y_train_hf, training_iter):

    XX = torch.Tensor(
        np.hstack([np.array(x_train_hf), np.array(mu1).reshape(-1, 1)]))
    likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
    m2 = HF_nonlin_gp(XX, y_train_hf, likelihood2)

    # Find optimal model hyperparameters
    m2.train()
    likelihood2.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer2 = torch.optim.Adam(m2.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2, m2)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer2.zero_grad()
        # Output from model
        output2 = m2(XX)
        # Calc loss and backprop gradients
        loss = -mll2(output2, y_train_hf)
        loss.backward()
        if i % 10 == 0:
            print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                # model.covar_module.base_kernel.lengthscale.item(),
                # model.likelihood.noise.item()
            ))
        optimizer2.step()

    return m2, likelihood2


def evaluate_second_lvl(m1, likelihood1, m2, likelihood2, x_val, nsamples=1000):

    # Get into evaluation (predictive posterior) mode
    m2.eval()
    likelihood2.eval()

    # Predict at validation points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        ntest = x_val.shape[0]
        trained_pred_dist0 = likelihood1(m1(torch.Tensor(x_val)))
        mu0 = trained_pred_dist0.mean
        v0 = trained_pred_dist0.variance
        C0 = trained_pred_dist0.covariance_matrix

        Z = np.random.multivariate_normal(mu0.flatten(), C0, nsamples)
        tmp_m = np.zeros((nsamples, ntest))
        tmp_v = np.zeros((nsamples, ntest))

        # Push samples through f_2
        for i in range(0, nsamples):
            XXX = torch.Tensor(np.hstack([x_val, np.array(Z)[i, :][:, None]]))
            trained_pred_dist2 = likelihood2(m2(XXX))
            mu2 = trained_pred_dist2.mean
            v2 = trained_pred_dist2.variance
            tmp_m[i, :] = mu2.flatten()
            tmp_v[i, :] = v2.flatten()

        # get mean and variance at X3
        mu3 = np.mean(tmp_m, axis=0)
        v3 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
        mu3 = mu2[:, None]
        v3 = np.abs(v3[:, None])

    return mu0, v0, mu3, v3


if __name__ in "__main__":

   # Load data
    minyear = 2000
    maxyear = 2001

    train_stations = ['Bharmaur', 'Churah', 'Jogindernagar', 'Kalatop', 'Kangra',
                      'Palampur', 'Salooni', 'Dehra', 'Hamirpur', 'Nadaun',
                      'Sujanpur', 'Dadahu', 'Dhaula Kuan', 'Kandaghat', 'Nahan',
                      'Pachhad', 'Paonta Sahib', 'Rakuna', 'Jubbal', 'Kothai',
                      'Mashobra', 'Rohru', 'Theog', 'Kalpa']
    hf_train_list = []
    for station in train_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.to_dataframe().dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj', 'Bhakra',
                    'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']
    val_list = []
    for station in val_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.to_dataframe().dropna().reset_index())
    val_df = pd.concat(val_list)

    era5_df = era5.gauges_download(
        train_stations + val_stations, minyear=minyear, maxyear=maxyear)

    # Prepare data

    # Transformations
    era5_df['tp_tr'], lf_lambda = sp.stats.boxcox(era5_df['tp'].values + 0.01)
    hf_train_df['tp_tr'], hf_lambda = sp.stats.boxcox(
        hf_train_df['tp'].values + 0.01)
    val_df['tp_tr'] = sp.stats.boxcox(
        val_df['tp'].values + 0.01, lmbda=hf_lambda)

    # Splitting
    x_train_lf = era5_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_lf = era5_df['tp_tr'].values.reshape(-1, 1)
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df['tp_tr'].values.reshape(-1, 1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler().fit(x_train_hf)
    x_train_hf = scaler.transform(x_train_hf)
    x_train_lf = scaler.transform(x_train_lf)
    x_val = scaler.transform(x_val)

    # Make tensors
    x_train_lf, y_train_lf = torch.Tensor(
        x_train_lf), torch.Tensor(y_train_lf.reshape(-1))
    x_train_hf, y_train_hf = torch.Tensor(
        x_train_hf), torch.Tensor(y_train_hf.reshape(-1))

    # Train and evaluate model
    training_iter = 200
    m1, likelihood1 = train_first_lvl(x_train_lf, y_train_lf, training_iter)
    mu1, v1 = evaluate_first_lvl(m1, likelihood1, x_train_hf)
    m2, likelihood2 = train_second_lvl(x_train_hf, y_train_hf, training_iter)
    mu0, v0, mu2, v2 = evaluate_second_lvl(
        m1, likelihood1, m2, likelihood2, x_val)

    plt.figure()
    plt.scatter(x_val[:10, 0], y_val[:10])
    plt.scatter(x_val[:10, 0], mu2[:10])
    plt.savefig('gpytorch_mfdgp_example_output.png')


def r2_low_vs_high(val_df, mu0, mu2):

    val_df['mu0'] = mu0
    val_df['mu2'] = mu2

    val_dfs = [x for _, x in val_df.groupby(['lon', 'lat', 'z'])]
    R2_hf = []
    R2_lf = []

    for df in val_dfs:
        xval_ = df[['time', 'lat', 'lon', 'z']
                   ].values.reshape(-1, 4)  # 'slope'
        yval_ = df['tp_tr'].values.reshape(-1, 1)

        # ALL
        y_pred_lf = np.nan_to_num(
            sp.special.inv_boxcox(df['mu0'].values, lf_lambda).reshape(-1))
        y_pred_hf = np.nan_to_num(
            sp.special.inv_boxcox(df['mu2'].values, hf_lambda).reshape(-1))
        y_true_ = sp.special.inv_boxcox(yval_, hf_lambda).reshape(-1)
        R2_hf.append(r2_score(y_true_, y_pred_hf))
        R2_lf.append(r2_score(y_true_, y_pred_lf))

    np.savetxt('table3_ypred_lf_r2_2000-2010.csv', R2_lf)
    np.savetxt('table3_ypred_hf_r2_2000-2010.csv', R2_hf)
