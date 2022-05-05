# Train model for Beas Sutlej Basin

import sys
sys.path.append("/home/users/ktazi")
sys.path.append("/data/hpcdata/users/kenzi22")

import numpy as np
import pandas as pd
import scipy as sp

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

# custom modules
import models as m

# Import data
lf_df = pd.read_csv('~/data/lf_beasut_data3.csv')
hf_df = pd.read_csv('~/data/hf_beasut_data3.csv')
plot_df = pd.read_csv('~/data/plot_beasut_data3.csv')

# Transformations    
hf_df['tp_tr'], hf_lambda = sp.stats.boxcox(hf_df['tp'].values + 0.01)
lf_df['tp_tr'], lf_lambda = sp.stats.boxcox(lf_df['tp'].values + 0.01)
lf_df['z'] /= 9.81 

# Split
x_train_lf = lf_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)
y_train_lf = lf_df[['tp_tr']].values.reshape(-1,1)
x_train_hf = hf_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)
y_train_hf = hf_df[['tp_tr']].values.reshape(-1,1)
x_plot = plot_df[['time', 'lat', 'lon', 'z']].values.reshape(-1,4)

# Format
X_train, Y_train = convert_xy_lists_to_arrays([x_train_lf, x_train_hf], [y_train_lf, y_train_hf])
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
np.savetxt('y_predh3.csv', y_predh, delimiter=",")
np.savetxt('y_predl3.csv', y_predl, delimiter=",")
np.savetxt('std_predh3.csv', std_predh, delimiter=",")
np.savetxt('std_predl3.csv', std_predl, delimiter=",")

