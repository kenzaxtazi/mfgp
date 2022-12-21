import sys
sys.path.append('/data/hpcdata/users/kenzi22/')

from load import beas_sutlej_gauges
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import gpytorch
import torch
import pandas as pd
import numpy as np
import scipy as sp

class GPRegressionModel(gpytorch.models.ExactGP):
    """ GPyTorch GP regression class"""

    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=4))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gpytorch_gp(train_x, train_y, training_iter):
    """ Create model instance and train """

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    training_iter = training_iter

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (  # lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            # model.covar_module.base_kernel.lengthscale.item(),
            # model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood


def model_eval(model, likelihood, val_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        trained_pred_dist = likelihood(model(val_x))
        y_pred = trained_pred_dist.mean
        y_std = trained_pred_dist.stddev

    return y_pred, y_std


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

    # Prepare data

    # Transformations
    hf_train_df['tp_tr'], hf_lambda = sp.stats.boxcox(
        hf_train_df['tp'].values + 0.1)
    val_df['tp_tr'] = sp.stats.boxcox(
        val_df['tp'].values + 0.1, lmbda=hf_lambda)

    # Splitting
    x_train_hf = hf_train_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_train_hf = hf_train_df['tp_tr'].values.reshape(-1)
    x_val = val_df[['time', 'lat', 'lon', 'z']].values.reshape(-1, 4)
    y_val = val_df['tp_tr'].values.reshape(-1)

    # Scaling
    scaler = MinMaxScaler().fit(x_train_hf)
    x_train_hf = scaler.transform(x_train_hf)
    x_val = scaler.transform(x_val)

    # Make tensors
    train_x, train_y = torch.Tensor(
        x_train_hf), torch.Tensor(y_train_hf)
    val_x, val_y = torch.Tensor(
        x_val), torch.Tensor(y_val)

    if torch.cuda.is_available():
        train_x, train_y, val_x, val_y = train_x.cuda(), train_y.cuda(), val_x.cuda(), val_x.cuda()

    # Train and evaluate model
    training_iter = 30
    model, likelihood = gpytorch_gp(train_x, train_y, training_iter)
    y_pred, y_std = model_eval(model, likelihood, val_x)

    # Metrics
    y_pred = sp.special.inv_boxcox(np.array(y_pred), lf_lambda).reshape(-1)
    y_true = sp.special.inv_boxcox(y_val, lf_lambda).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse_all, rmse_p5, rmse_p95 =  rmses(y_pred, y_true)

    print('Mean R2 = ', r2)
    print('Mean RMSE = ', rmse_all)
    print('5th RMSE = ', rmse_p5)
    print('95th RMSE = ', rmse_p95)
