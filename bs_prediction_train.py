# Train model for Beas Sutlej Basin

import sys
sys.path.append("/home/users/ktazi")
sys.path.append("/data/hpcdata/users/kenzi22")

import numpy as np
import pandas as pd
import scipy as sp

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array

# custom modules
import models as m
from pwd import pwd

# Import data
lf_df = pd.read_csv(pwd + 'data/lf_beasut_data.csv')
hf_df = pd.read_csv(pwd + 'data/hf_beasut_data.csv')
plot_df = pd.read_csv(pwd + 'data/plot_beasut_data.csv')

# Transformations    
hf_df['tp_tr'], hf_lambda = sp.stats.boxcox(hf_df['tp'].values + 0.01)
lf_df['tp_tr'], lf_lambda = sp.stats.boxcox(lf_df['tp'].values + 0.01)

# Split
x_train_lf = lf_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)
y_train_lf = lf_df[['tp_tr']].values.reshape(-1,1)
x_train_hf = hf_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)
y_train_hf = hf_df[['tp_tr']].values.reshape(-1,1)
x_plot = plot_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)

# Format
X_train = convert_x_list_to_array([x_train_lf, x_train_hf])
Y_train = convert_x_list_to_array([y_train_lf, y_train_hf])
X_plot = convert_x_list_to_array([x_plot, x_plot])

# Initialise and train model
model = m.linear_mfdgp(X_train, Y_train)
model.optimize()

# Save model
# TODO

# Make predictions
n = x_plot.shape[0]
y_predh, std_predh = sp.special.inv_boxcox(model.predict(X_plot[n:]), hf_lambda)
y_predl, std_predl = sp.special.inv_boxcox(model.predict(X_plot[:n]), lf_lambda)

# save predictions
np.savetxt(pwd + "data/y_predh.csv", y_predh, delimiter=",")
np.savetxt(pwd + "data/y_predl.csv", y_predl, delimiter=",")
np.savetxt(pwd + "data/std_predh.csv", std_predh, delimiter=",")
np.savetxt(pwd + "data/std_predl.csv", std_predl, delimiter=",")

