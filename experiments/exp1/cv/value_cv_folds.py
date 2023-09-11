import sys  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# custom modules
from load import value


# Load data
gauge_df = value.all_gauge_data(2000, 2005, monthly=True)
cv_range_df1 = gauge_df.groupby('station_id').mean()
data = cv_range_df1[['longitude', 'latitude']].values

# K-means
kmeans = KMeans(n_clusters=5).fit(data)
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

# Keep stations closest to centroids

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
np.save('exp1_cv_locs_test.npy', cv_arr)

centers_arr = np.array(kmeans.clusters_centers_)
np.save('exp1_cv_cluster_centers.npy', cv_arr)
'''
