from utils.metrics import rmses, msll
from sklearn.metrics import r2_score
import scipy as sp
import numpy as np
import pandas as pd
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from load import beas_sutlej_gauges, era5
import sys
sys.path.append('/data/hpcdata/users/kenzi22/')
sys.path.append('/data/hpcdata/users/kenzi22/mfdgp/')


class LF_gp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LF_gp, self).__init__(train_x, train_y, likelihood)

        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        dim = train_x.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5,
                                                    ard_num_dims=dim, active_dims=np.arange(dim))
        # gpytorch.kernels.GridInterpolationKernel(grid_size=grid_size, num_dims=dim)
        self.covar_module = base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_nonlin_gp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super(HF_nonlin_gp, self).__init__(
            train_x, train_y, likelihood)

        map_kernel = gpytorch.kernels.RBFKernel(1, active_dims=[4]) * gpytorch.kernels.RBFKernel(
            ard_num_dims=4, active_dims=[0, 1, 2, 3])  # outputscale_prior=gpytorch.priors.NormalPrior(1, 1))
        bias_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=4, active_dims=[0, 1, 2, 3])  # outputscale_prior=gpytorch.priors.NormalPrior(0.01, 0.01))

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            map_kernel + bias_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_lin_gp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super(HF_lin_gp, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[4])) + base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HF_custom_gpmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super(HF_custom_gpmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_first_lvl(train_x_lf, train_y_lf, training_iter):

    likelihood1 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=torch.ones(len(train_x_lf)) * 0.01)
    m1 = LF_gp(train_x_lf, train_y_lf, likelihood1)

    if torch.cuda.is_available():
        m1 = m1.cuda()
        likelihood1 = likelihood1.cuda()

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
        output1 = m1(train_x_lf)
        # Calc loss and backprop gradients
        loss1 = -mll1(output1, train_y_lf)
        loss1.backward()
        if i % 10 == 0:
            print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss1.item(),
                # model.covar_module.base_kernel.lengthscale.item(),
                # model.likelihood.noise.item()
            ))
        optimizer1.step()

    return m1, likelihood1


def evaluate_first_lvl(m1, likelihood1, train_x_hf):
    # Get into evaluation (predictive posterior) mode
    m1.eval()
    likelihood1.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        trained_pred_dist1 = likelihood1(m1(train_x_hf))
        mu1 = trained_pred_dist1.mean
        v1 = trained_pred_dist1.variance

    return mu1, v1


def train_second_lvl(train_x_hf, train_y_hf, mu1, training_iter, base_kernel):

    x_arr = np.array(train_x_hf.cpu())
    mu1_arr = np.array(mu1.cpu()).reshape(-1, 1)
    XX = torch.Tensor(np.hstack([x_arr, mu1_arr]))

    if torch.cuda.is_available():
        XX = XX.cuda()

    likelihood2 = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=torch.ones(len(train_x_lf)) * 0.01)
    m2 = HF_lin_gp(XX, train_y_hf, likelihood2, base_kernel)

    if torch.cuda.is_available():
        m2 = m2.cuda()
        likelihood2 = likelihood2.cuda()

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
        loss = -mll2(output2, train_y_hf)
        loss.backward()
        if i % 10 == 0:
            print('Iter %d/%d - Loss: %.3f' % (  # '  noise: %.3f # lengthscale: %.3f   % (
                i + 1, training_iter, loss.item(),))
            # model.covar_module.base_kernel.lengthscale.item(),
            # m2.likelihood.noise.item()))
        optimizer2.step()

    return m2, likelihood2


def evaluate_second_lvl(m1, likelihood1, m2, likelihood2, val_x, nsamples=1000):

    # Get into evaluation (predictive posterior) mode
    m2.eval()
    likelihood2.eval()

    # Predict at validation points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        val_arr = np.array(val_x.cpu())
        ntest = val_arr.shape[0]

        if torch.cuda.is_available():
            val_x = val_x.cuda()

        trained_pred_dist0 = likelihood1(m1(val_x))
        mu0 = trained_pred_dist0.mean.cpu()
        v0 = trained_pred_dist0.variance.cpu()
        C0 = trained_pred_dist0.covariance_matrix.cpu()

        Z = np.random.multivariate_normal(mu0.flatten(), C0, nsamples)
        tmp_m = np.zeros((nsamples, ntest))
        tmp_v = np.zeros((nsamples, ntest))

        # Push samples through f_2
        for i in range(0, nsamples):
            XXX = torch.Tensor(
                np.hstack([val_arr, np.array(Z)[i, :][:, None]]))
            if torch.cuda.is_available():
                XXX = XXX.cuda()

            trained_pred_dist2 = likelihood2(m2(XXX))
            mu2 = trained_pred_dist2.mean.cpu()
            v2 = trained_pred_dist2.variance.cpu()
            tmp_m[i, :] = mu2.flatten()
            tmp_v[i, :] = v2.flatten()

        # get mean and variance at X3
        mu3 = np.mean(tmp_m, axis=0)
        v3 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
        mu3 = mu3[:, None]
        v3 = np.abs(v3[:, None])

    return mu0, v0, mu3, v3


if __name__ in "__main__":

   # Load data
    minyear = 2000
    maxyear = 2001

    train_stations = ['Banjar', 'Churah', 'Jogindernagar', 'Kalatop', 'Kangra', 'Sujanpur',
                      'Dadahu', 'Dhaula Kuan', 'Kandaghat', 'Nahan', 'Dehra',
                      'Pachhad', 'Paonta Sahib', 'Rakuna', 'Jubbal', 'Kothai',
                      'Mashobra', 'Rohru', 'Theog', 'Kalpa', 'Salooni', 'Hamirpur', 'Nadaun', ]
    hf_train_list = []
    for station in train_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        hf_train_list.append(station_ds.to_dataframe().dropna().reset_index())
    hf_train_df = pd.concat(hf_train_list)

    val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj',
                    'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']
    val_list = []
    for station in val_stations:
        station_ds = beas_sutlej_gauges.gauge_download(
            station, minyear=minyear, maxyear=maxyear)
        val_list.append(station_ds.to_dataframe().dropna().reset_index())
    val_df = pd.concat(val_list)

    era5_ds = era5.collect_ERA5('indus', minyear=minyear, maxyear=maxyear)
    #era5_df = era5.gauges_download(val_stations + train_stations, minyear=minyear, maxyear=maxyear)

    lf_df = era5_ds.to_dataframe().dropna().reset_index()
    lf_df1 = lf_df[lf_df['lat'] <= 33.5]
    lf_df2 = lf_df1[lf_df1['lat'] >= 30]
    lf_df3 = lf_df2[lf_df2['lon'] >= 75.5]
    lf_train_df = lf_df3[lf_df3['lon'] <= 83]

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
    y_train_lf = lf_train_df['tp_tr'].values.reshape(-1)
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df[['tp_tr']].values.reshape(-1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1)

    # Scaling
    scaler = MinMaxScaler().fit(x_train_hf)
    x_train_hf1 = scaler.transform(x_train_hf)
    x_train_lf1 = scaler.transform(x_train_lf)
    x_val1 = scaler.transform(x_val)

    # Make tensors
    train_x_lf, train_y_lf = torch.Tensor(
        x_train_lf1), torch.Tensor(y_train_lf)
    train_x_hf, train_y_hf = torch.Tensor(
        x_train_hf1), torch.Tensor(y_train_hf)
    val_x, val_y = torch.Tensor(x_val1), torch.Tensor(y_val)

    # Set to CUDA
    if torch.cuda.is_available():
        train_x_hf, train_y_hf = train_x_hf.cuda(), train_y_hf.cuda()
        train_x_lf, train_y_lf = train_x_lf.cuda(), train_y_lf.cuda(),
        val_x, val_y = val_x.cuda(), val_y.cuda()

    # Train and evaluate model

    training_iter = 400
    base_kernel = gpytorch.kernels.MaternKernel(
        ard_num_dims=4, active_dims=[0, 1, 2, 3])

    m1, likelihood1 = train_first_lvl(
        train_x_lf, train_y_lf, 200)
    mu1, v1 = evaluate_first_lvl(m1, likelihood1, train_x_hf)
    m2, likelihood2 = train_second_lvl(
        train_x_hf, train_y_hf, mu1, 1000, base_kernel)
    mu0, v0, mu2, v2 = evaluate_second_lvl(m1, likelihood1, m2,
                                           likelihood2, val_x, nsamples=100)

    # Metrics
    y_pred = sp.special.inv_boxcox(np.array(mu0), lf_lambda).reshape(-1)
    y_true = sp.special.inv_boxcox(y_val, lf_lambda).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse_all, rmse_p5, rmse_p95 = rmses(y_pred, y_true)
    log_loss = msll(y_val, mu2, v2)

    print('Mean R2 = ', r2)
    print('Mean RMSE = ', rmse_all)
    print('5th RMSE = ', rmse_p5)
    print('95th RMSE = ', rmse_p95)
    print('MSLL = ', log_loss)
