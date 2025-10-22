import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# custom modules
from load import beas_sutlej_gauges, data_dir


# Load data
all_station_dict = pd.read_csv(
    data_dir + 'bs_gauges/gauge_info.csv', index_col='station').T
sta_list = list(all_station_dict)

df_list = []
minyear = '1980'
maxyear = '2009-12-31'

for station in sta_list:
    station_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=minyear, maxyear=maxyear)
    df_list.append(station_ds.to_dataframe().dropna().reset_index())
    sta_df = pd.concat(df_list)

# Identify time period with most data
counts = sta_df.groupby('time').count()
counts.reset_index(inplace=True)

'''
plt.figure(figsize=(20, 5))
plt.scatter(counts['time'], counts['lat'])
plt.ylabel('Number of active stations')
plt.xlabel('Year')
plt.savefig('stations_count_vs_time.pdf', bbox_inches='tight', dpi=300)
'''
# visually we identify 2000-2005 as a period with approximately the most data
gauge_df = sta_df[(sta_df['time'] > 2000) & (sta_df['time'] < 2005)]

# Apply K-means clustering
cv_range_df1 = gauge_df.groupby('z').mean()
kmeans = KMeans(n_clusters=5, random_state=101).fit(
    cv_range_df1.values[:, 2:4])
unique, counts = np.unique(kmeans.labels_, return_counts=True)
#print(unique, counts)
# print(kmeans.cluster_centers_)
cv_range_df1['fold'] = kmeans.labels_

# Plot initial groups
"""
cv_range_df1.plot.scatter(
    x='longitude', y='latitude', c=cv_range_df1['fold'], cmap='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], c='k')
"""

# Keep seven closest stations to centroids

# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
closest_pt_idx = []
for iclust in range(kmeans.n_clusters):
    # get all points assigned to each cluster:
    cluster_pts = cv_range_df1[cv_range_df1['fold'] == iclust]
    # get all indices of points assigned to this cluster:
    cluster_pts_indices = np.where(cluster_pts['fold'] == iclust)[0]

    cluster_cen = kmeans.cluster_centers_[iclust]
    cluster_pts['e_dist'] = [euclidean(
        cluster_pts.iloc[idx].values[2:4], cluster_cen) for idx in cluster_pts_indices]
    cluster_pts.sort_values('e_dist', ignore_index=True, inplace=True)
    closest_pt_idx.append(cluster_pts[['longitude', 'latitude']].iloc[:7])

# Convert data to arrays and save as .npy
'''
cv_arr = np.array(closest_pt_idx)
print(cv_arr.shape)
np.save('exp2_cv_locs_test.npy', cv_arr)

centers_arr = np.array(kmeans.clusters_centers_)
np.save('exp2_cv_cluster_centers.npy', cv_arr)
'''
