import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as crf
import matplotlib.pyplot as plt

# custom modules
from load import value

# Load data
recov_cluster_centres = np.load('exp1_cv_cluster_centers.npy')
recov_locs = np.load('exp1_cv_locs.npy')

gauge_df = value.all_gauge_data(2000, 2005, monthly=True)
all_station_df = gauge_df.groupby('station_id').mean()


# Figure

plt.figure(figsize=(6, 5))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(crf.COASTLINE, alpha=0.3)
plt.xticks([-10, 0, 10, 20, 30], ['-10°E', '0°E', '10°E', '20°E', '30°E'])
plt.yticks([30, 40, 50, 60, 70], ['30°N', '40°N', '50°N', '60°N', '70°N'])
ax.set_xlim([-15, 35])
ax.set_ylim([30, 75])
ax.text(-13.6, 32.9, "a", fontsize=14, va='top', zorder=4)
ax.add_patch(plt.Rectangle((-15, 30), 4, 4,
                           facecolor='white', edgecolor='black', zorder=3))

for i in range(len(recov_cluster_centres)):
    plt.scatter(recov_locs[i, :, 0],
                recov_locs[i, :, 1], label='fold ' + str(i+1))

plt.scatter(all_station_df['longitude'], all_station_df['latitude'], edgecolor='k', alpha=0.1,
            zorder=8, label='other stations')
plt.scatter(recov_cluster_centres[:, 0], recov_cluster_centres[:, 1],
            c='k', marker='*', label='cluster centres')

plt.legend(fontsize=8)
plt.savefig('exp1_cv_with_inset_test.pdf', bbox_inches='tight', dpi=300)
