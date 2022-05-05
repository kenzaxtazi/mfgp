import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

y_predh =np.genfromtxt('y_predh.csv', delimiter=",")
'''
y_predl_df = pd.read_csv('y_predl2.csv', delimiter=",")
std_predh_df = pd.read_csv('std_predh2.csv', delimiter=",")
std_predl_df = pd.read_csv('std_predl2.csv', delimiter=",")
plot_df = pd.read_csv('~/data/plot_beasut_data2.csv')
'''
plot_df['tp'] = y_predh
plot_df = plot_df.set_index(['lat','lon'])
plot_ds = plot_df.to_xarray()

plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([75, 83, 29, 35])
g = plot_ds["tp"].plot(cmap="magma_r", vmin=0.001, cbar_kwargs={
    "label": "Average precipitation [mm/day]",
    "extend": "neither", "pad": 0.10})
g.cmap.set_under("white")
# ax.add_feature(cf.BORDERS)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Beas and Sutlej Basin ")
plt.show()
