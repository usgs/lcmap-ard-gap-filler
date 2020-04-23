Disclaimer
==========

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.

## Description
This is the source code of "Zhou Q, Xian G, Shi H. Gap Fill of Land Surface Temperature and Reflectance Products in Landsat Analysis Ready Data. Remote Sensing. 2020 Jan;12(7):1192."

The recently released Landsat analysis ready data (ARD) over the United States provides the opportunity to investigate landscape dynamics using dense time series observations at 30-m resolution. However, the dataset often contains data gaps (or missing data) because of cloud contamination or data acquisition strategy, which result in different capabilities for seasonality modeling. We present a new algorithm that focuses on data gap filling using clear observations from orbit overlap regions. Multiple linear regression models were established for each pixel time series to estimate stable predictions and uncertainties. The model's training data came from stratified random samples based on the time series similarity between the pixel and data from the overlap regions.


## Usage
A typical workflow is 1) download the time series image stack, 2) generate training data, 3) fill time series with data gaps.

### Generate training data
The primary inputs are acquisition dates of images and the image stack as 3d array (n_rows, n_columns, n_timesteps).

The outputs are .model file that stores the cluster model and .pkl file that contains a set of training data.
```python
>>> import gap_fill_ard as gfa
>>> 
>>> import glob
>>> import os
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from datetime import datetime
>>> import pickle
>>> 
>>> acq_datelist = np.load(r'test/resources/dates.npy', allow_pickle=True)
>>> file_list = []
>>> for name in glob.glob('test/resources/data/*.npy'):
>>> 	file_list.append(name)
>>> temp_arr = np.load(file_list[0])
>>> img_stack = np.zeros((len(file_list), temp_arr.shape[0], temp_arr.shape[1]))
>>> file_list.sort()
>>> for i_row, f in enumerate(file_list):
>>> 	img_stack[i_row, :, :] = np.load(f)
>>> gfa.gather_training(acq_datelist, img_stack, outDir=r'test/', total_size=200, save_slice=False)
```

### Fill single pixel time series
```python
>>> n_cpu = 1
>>> dates_list = np.load(r'test/resources/dates.npy', allow_pickle=True)
>>> outname = os.path.join(work_dir, 'training_Atlanta')
>>> training_data_path = '{}.pkl'.format(outname)
>>> cluster_model_path = '{}.model'.format(outname)
>>> training_data = pd.read_pickle(training_data_path)
>>> date_idx = training_data.index.isin(dates_list)
>>> training_data = training_data[date_idx].values.T
>>> cluster_model = pickle.load(open(cluster_model_path, 'rb'))
>>> labels = cluster_model.labels_
>>> centroids = cluster_model.cluster_centers_
>>> centroids = centroids[:, date_idx]
>>> ts_org = np.load(r'test/resources/data/clear_full_2591.npy')
>>> ts_y = ts_org[800, :]
>>> gap_filled, gap_filled_std = gf.gap_fill_pixel(ts_y, training_data, labels, centroids)
```


## Installing
System requirements
* python3-dev
* python-virtualenv

It's highly recommended to do all your development & testing in anaconda virtual environment.

Required modules
* numpy>=1.10.0
* scikit-learn>=0.18
* gdal>=3.0.1
* multiprocessing>=0.70.8
