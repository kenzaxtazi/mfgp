#!/usr/bin/env python
# coding: utf-8

# High low plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                   mark_inset)


# Europe

hf_arr = np.load(
    '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp1/ouputs/exp1_ypred_hf_r2_2000-2005.npy')
lf_arr = np.load(
    '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp1/ouputs/exp1_ypred_lf_r2_2000-2005.npy')

fig, ax1 = plt.subplots(figsize=(5, 5))

for i in range(5):
    ax1.scatter(lf_arr[i], hf_arr[i], label='fold ' + str(i+1))
ax1.plot([-10.5, 1], [-10.5, 1], linestyle='--')
ax1.set_xlim([-10.5, 1])
ax1.set_ylim([-10.5, 1])
ax1.set_ylabel('High Fidelity $\mathregular{R^{2}}$')
ax1.set_xlabel('Low Fidelity $\mathregular{R^{2}}$')
ax1.text(-10.15, -9.7, "a", fontsize=14, va='top', zorder=4)
ax1.add_patch(plt.Rectangle((-10.5, -10.5), 1, 1,
              facecolor='white', edgecolor='black', zorder=3))
plt.legend()

ax2 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.5, 0.2, 0.4, 0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

for i in range(5):
    ax2.scatter(lf_arr[i], hf_arr[i])
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.savefig('value_high_low_plot_2000_2005.pdf', bbox_inches='tight')


## Beas and Sutlej

hf_arr = np.load(
    '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp2/outputs/exp2_ypred_hf_r2_2000_2005.npy')
lf_arr = np.load(
    '/Users/kenzatazi/Documents/CDT/Code/mfdgp/experiments/exp2/outputs/exp2_ypred_lf_r2_2000_2005.npy', allow_pickle=True)

fig, ax1 = plt.subplots(figsize=(5, 5))

for i in range(5):
    ax1.scatter(lf_arr[i], hf_arr[i], label='fold ' + str(i+1))
ax1.plot([-10.5, 1], [-10.5, 1], linestyle='--')
ax1.set_xlim([-10.5, 1])
ax1.set_ylim([-10.5, 1])
ax1.set_ylabel('High Fidelity $\mathregular{R^{2}}$')
ax1.set_xlabel('Low Fidelity $\mathregular{R^{2}}$')
ax1.text(-10.2, -9.75, "b", fontsize=14, va='top', zorder=4)
ax1.add_patch(plt.Rectangle((-10.5, -10.5), 1, 1,
              facecolor='white', edgecolor='black', zorder=3))
plt.legend()

ax2 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.5, 0.2, 0.4, 0.4])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

for i in range(5):
    ax2.scatter(lf_arr[i], hf_arr[i])
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.savefig('bs_high_low_plot_2000_2005.pdf', bbox_inches='tight')
