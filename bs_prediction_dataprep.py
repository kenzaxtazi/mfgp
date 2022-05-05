# Script for generating data for prediction map

import sys
sys.path.append("/home/users/ktazi")
sys.path.append("/data/hpcdata/users/kenzi22")

import xarray as xr
import numpy as np
import pandas as pd

# custom modules
from load import beas_sutlej_gauges, era5


########### Data

### HF data

# Import gauge data
station_dict = {'Arki':[31.154, 76.964, 1176], 'Banjar': [31.65, 77.34, 1914], 'Banjar IMD':[31.637, 77.344, 1427],  
                'Berthin':[31.471, 76.622, 657], 'Bhakra':[31.424, 76.417, 518], 'Barantargh': [31.087, 76.608, 285], 
                'Bharmaur': [32.45, 76.533, 1867], 'Bhoranj':[31.648, 76.698, 834], 'Bhuntar': [31.88, 77.15, 1100], 
                'Churah': [32.833, 76.167, 1358], 'Dadahu':[30.599, 77.437, 635], 'Daslehra': [31.4, 76.55, 561], 
                'Dehra': [31.885, 76.218, 472], 'Dhaula Kuan': [30.517, 77.479, 443], 'Ganguwal': [31.25, 76.486, 345], 
                'Ghanauli': [30.994, 76.527, 284], 'Ghumarwin': [31.436, 76.708, 640], 'Hamirpur': [31.684, 76.519, 763], 
                'Janjehl': [31.52, 77.22, 2071], 'Jogindernagar': [32.036, 76.734, 1442], 'Jubbal':[31.12, 77.67, 2135], 
                'Kalatop': [32.552, 76.018, 2376], 'Kalpa': [31.54, 78.258, 2439], 'Kandaghat': [30.965, 77.119, 1339], 
                'Kangra': [32.103, 76.271, 1318], 'Karsog': [31.383, 77.2, 1417], 'Kasol': [31.357, 76.878, 662], 
                'Kaza': [32.225, 78.072, 3639], 'Kotata': [31.233, 76.534, 320], 'Kothai': [31.119, 77.485, 1531],
                'Kumarsain': [31.317, 77.45, 1617], 'Larji': [31.80, 77.19, 975], 'Lohard': [31.204, 76.561, 290], 
                'Mashobra': [31.13, 77.229, 2240], 'Nadaun': [31.783, 76.35, 480], 'Nahan': [30.559, 77.289, 874], 
                'Naina Devi': [31.279, 76.554, 680], 'Nangal': [31.368, 76.404, 354], 'Olinda': [31.401, 76.385, 363],
                'Pachhad': [30.777, 77.164, 1257], 'Palampur': [32.107, 76.543, 1281], 'Pandoh':[31.67,77.06, 899], 
                'Paonta Sahib': [30.47, 77.625, 433], 'Rakuna': [30.605, 77.473, 688], 'Rampur': [31.454,77.644, 976],
                'Rampur IMD': [31.452, 77.633, 972], 'Rohru':[31.204, 77.751, 1565], 'Sadar-Bilarspur':[31.348, 76.762, 576], 
                'Sadar-Mandi': [31.712, 76.933, 761], 'Sainj': [31.77, 77.31, 1280] , 'Salooni':[32.728, 76.034, 1785],
                'Sarkaghat': [31.704, 76.812, 1155], 'Sujanpur':[31.832, 76.503, 557], 'Sundernargar': [31.534, 76.905, 889], 
                'Suni':[31.238,77.108, 655], 'Suni IMD':[31.23, 77.164, 765], 'Swaghat': [31.713, 76.746, 991], 
                'Theog': [31.124, 77.347, 2101]}

station_df = pd.DataFrame.from_dict(station_dict, orient='index', columns=['lat', 'lon', 'elv'])
station_df = station_df.reset_index()

# Define training set
hf_train_df1 = station_df[(station_df['lon']< 77.0) & (station_df['lat']> 32)]
hf_train_df2 = station_df[(station_df['lon']< 76.60) & ((station_df['lat']< 32) & (station_df['lat']> 31.6))]
hf_train_df3 = station_df[(station_df['lon']> 77.0) & (station_df['lat']< 31)]
hf_train_df4 = station_df[(station_df['lon']< 78.0) & (station_df['lon']> 77.0) & (station_df['lat']> 31) & (station_df['lat']< 31.23)]
hf_train_df5 = station_df[(station_df['lon']> 78.2)]
# hf_train_stations = list(hf_train_df1['index'].values)# + list(hf_train_df2['index'].values) + list(hf_train_df3['index'].values) + list(hf_train_df4['index'].values) + list(hf_train_df5['index'].values)
station_list =  ['Banjar', 'Bhakra', 'Larji', 'Kasol', 'Sainj', 'Suni', 'Pandoh', 'Janjehl', 'Bhuntar', 'Rampur']

# Format training data
hf_train_list = []
for station in station_list:
    station_ds = beas_sutlej_gauges.gauge_download(station, minyear=1998, maxyear=2004)
    station_ds['z'] = station_dict[station][2]
    #station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    #station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1,}) # 'slope':1})
    hf_train_list.append(station_ds)
hf_train_ds = xr.merge(hf_train_list)
hf_train_df = hf_train_ds.to_dataframe().dropna().reset_index()

### Plot data
srtm_ds = xr.open_dataset('~/data/SRTM_data.nc')
srtm_ds = srtm_ds.rename({'nlat': 'lat', 'nlon': 'lon', 'elevation': 'z'})

# Mask to beas and sutlej
mask_filepath = '~/data/Masks/Beas_Sutlej_highres_mask.nc'
mask = xr.open_dataset(mask_filepath)
mask_da = mask.Overlap
msk_srtm_ds = srtm_ds.where(mask_da > 0, drop=True)

msk_srtm_df = msk_srtm_ds.to_dataframe().dropna().reset_index()
plot_df = msk_srtm_df.drop(['slope', 'aspect', 'latitude', 'longitude'], axis=1)

plot_df1 = plot_df[plot_df['lat']<= 32]
plot_df2 = plot_df1[plot_df1['lat']>= 31]
plot_df3 = plot_df2[plot_df2['lon']>= 77]
plot_df4 = plot_df3[plot_df3['lon']<= 78]

# add to date predict at
n = plot_df.shape[0]
t = np.ones(n) * (1998 + (1./24.) * 23)
plot_df['time'] = t

### LF data
era5_ds =  era5.collect_ERA5('beas_sutlej', minyear=1998, maxyear=2004)
#era5_ds1 = era5_ds.interp_like(msk_srtm_ds, 'nearest')

lf_train_df = era5_ds.to_dataframe().dropna().reset_index()
lf_train_df = lf_train_df.drop(['expver', 'd2m', 'anor', 'slor', 'tcwv', 'N34'], axis=1)

lf_train_df1 = lf_train_df[lf_train_df['lat']<= 32]
lf_train_df2 = lf_train_df1[lf_train_df1['lat']>= 31]
lf_train_df3 = lf_train_df2[lf_train_df2['lon']>= 77]
lf_train_df4 = lf_train_df3[lf_train_df3['lon']<= 78]

# Save to csv
lf_train_df.to_csv('~/data/lf_beasut_data3.csv')
hf_train_df.to_csv('~/data/hf_beasut_data3.csv')
plot_df.to_csv('~/data/plot_beasut_data3.csv')