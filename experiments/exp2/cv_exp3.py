import sys
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

sys.path.append('/Users/kenzatazi/Documents/CDT/Code')
from load import beas_sutlej_gauges, era5, data_dir


## Load data
all_station_dict = pd.read_csv(
    data_dir + 'bs_gauges/gauge_info.csv', index_col='station').T
sta_list = list(all_station_dict)
minyear=1980
maxyear=2010

df_list = []
for station in sta_list:
    station_ds = beas_sutlej_gauges.gauge_download(station, minyear=minyear, maxyear=maxyear)
    df_list.append(station_ds.to_dataframe().dropna().reset_index())
    sta_df = pd.concat(df_list)


## Find times with the most stations
counts = sta_df.groupby('time').count()
counts.reset_index(inplace=True)

'''
plt.figure(figsize=(20,5))
plt.scatter(counts['time'], counts['lat'])
plt.xlim()
'''

# On inspection the period between 2000 and 2005 contains
# the largest number of active stations 
cv_range_df = sta_df[(sta_df['time']>2000) & (sta_df['time']<2005)]


## Apply k-means to coordinates

# Group by elevation to get one entry per location
cv_range_df1 = cv_range_df.groupby('z').mean()

# K-means
kmeans = KMeans(n_clusters=5).fit(cv_range_df1[['lon', 'lat']].values)
cv_range_df1['fold'] = kmeans.labels_

'''
cv_range_df1.plot.scatter(
    x='lon', y='lat', c=cv_range_df1['fold'], cmap='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], c='k')
'''

# check the smallest number of stations per cluster
unique, counts = np.unique(kmeans.labels_, return_counts=True)
print('Cluster: ', unique)
print('Counts per cluster: ', counts)


## Keep stations closest to cluster centers

closest_pt_idx = []

# Loop over all clusters
for iclust in range(kmeans.n_clusters):
    # get all points assigned to each cluster:
    cluster_pts = cv_range_df1[cv_range_df1['fold']== iclust]
    # get all indices of points assigned to this cluster:
    cluster_pts_indices = np.where(cluster_pts['fold'] == iclust)[0]

    cluster_cen = kmeans.cluster_centers_[iclust]
    cluster_pts['e_dist'] = [euclidean(
        cluster_pts.iloc[idx].values[2:4], cluster_cen) for idx in cluster_pts_indices]
    cluster_pts.sort_values('e_dist', ignore_index=True, inplace=True)
    closest_pt_idx.append(cluster_pts[['lon', 'lat']].iloc[:7])


'''
plt.figure(figsize=(6,5))
for i in range(len(closest_pt_idx)):
    plt.scatter(closest_pt_idx[i]['lon'],closest_pt_idx[i]['lat'], label='fold ' + str(i+1))
plt.scatter(cv_range_df1['lon'], cv_range_df1['lat'], edgecolor='k', alpha=0.1,
            zorder=8, label='other stations')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='k', marker='*', label='cluster centres')
plt.xlim([75.7, 79.2])
plt.xlabel('Longitude °E')
plt.ylabel('Latitude °N')
plt.legend(fontsize=8)
plt.savefig('Experiment3_CV.png', dpi=300)
'''

## Save coordinates for each CV fold

# Convert to 3D array and save as .npy
cv_arr = np.array(closest_pt_idx)
print(cv_arr.shape)
np.save('cv_locs.npy', cv_arr)