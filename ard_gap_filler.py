import numpy as np
import sys, os
import pandas as pd

from sklearn import linear_model
from sklearn.cluster import KMeans

import pickle

from osgeo import gdal
from multiprocessing import Process

import warnings
warnings.filterwarnings("ignore")

import logging
log = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(processName)s: %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.DEBUG)

def coefficient_matrix(dates, avg_days_yr=365.25, num_coefficients=8):
    """
    Fourier transform function to be used for the matrix of inputs for
    model fitting
    Args:
        dates: list of ordinal dates
        num_coefficients: how many coefficients to use to build the matrix
    Returns:
        Populated numpy array with coefficient values
        
    Original author: Kelcy Smith
    """
    w = 2 * np.pi / avg_days_yr

    matrix = np.zeros(shape=(len(dates), 7), order='F')

    # lookup optimizations
    # Before optimization - 12.53% of total runtime
    # After optimization  - 10.57% of total runtime
    cos = np.cos
    sin = np.sin

    w12 = w * dates
    matrix[:, 0] = dates
    matrix[:, 1] = cos(w12)
    matrix[:, 2] = sin(w12)

    if num_coefficients >= 6:
        w34 = 2 * w12
        matrix[:, 3] = cos(w34)
        matrix[:, 4] = sin(w34)

    if num_coefficients >= 8:
        w56 = 3 * w12
        matrix[:, 5] = cos(w56)
        matrix[:, 6] = sin(w56)

    return matrix


def lasso_fill(dates, X, avg_days_yr=365.25):
    #Date: n_features
    #X: (n_samples, n_features), non valid as 0
    coef_matrix = coefficient_matrix(dates, avg_days_yr) #(n_feature, 7)
    lasso = linear_model.Lasso()
    X_valid = (X != 0) * np.isfinite(X)
    X_invalid = ~X_valid
    for i_row in range(X.shape[0]):
        # print('shape :', X_valid[i_row, :].shape, coef_matrix[X_valid[i_row, :], :].shape, X[i_row, :].shape, X[i_row, :][X_valid[i_row, :]].shape)
        model = lasso.fit(coef_matrix[X_valid[i_row, :], :], X[i_row, :][X_valid[i_row, :]])
        X[i_row, :][X_invalid[i_row, :]] = model.predict(coef_matrix[X_invalid[i_row, :], :])
    return X

def ridge_fill(dates, X, avg_days_yr=365.25):
    #Date: n_features
    #X: (n_samples, n_features), non valid as 0
    coef_matrix = coefficient_matrix(dates, avg_days_yr) #(n_feature, 7)
    lasso = linear_model.Ridge()
    X_valid = (X != 0) * np.isfinite(X)
    X_invalid = ~X_valid
    for i_row in range(X.shape[0]):
        # print('shape :', X_valid[i_row, :].shape, coef_matrix[X_valid[i_row, :], :].shape, X[i_row, :].shape, X[i_row, :][X_valid[i_row, :]].shape)
        model = lasso.fit(coef_matrix[X_valid[i_row, :], :], X[i_row, :][X_valid[i_row, :]])
        X[i_row, :][X_invalid[i_row, :]] = model.predict(coef_matrix[X_invalid[i_row, :], :])
    return X

def huber_fill(dates, X, avg_days_yr=365.25):
    #Date: n_features
    #X: (n_samples, n_features), non valid as 0
    coef_matrix = coefficient_matrix(dates, avg_days_yr) #(n_feature, 7)
    huber = linear_model.HuberRegressor()
    X_valid = (X != 0)
    X_invalid = (X == 0)
    for i_row in range(X.shape[0]):
        print('shape :', X_valid[i_row, :].shape, coef_matrix[X_valid[i_row, :], :].shape, X[i_row, :].shape, X[i_row, :][X_valid[i_row, :]].shape)
        model = huber.fit(coef_matrix[X_valid[i_row, :], :], X[i_row, :][X_valid[i_row, :]])
        X[i_row, :][X_invalid[i_row, :]] = model.predict(coef_matrix[X_invalid[i_row, :], :])
    return X


def find_lastValley(arr_1d):
    # https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    min_idx = (np.diff(np.sign(np.diff(arr_1d))) > 0).nonzero()[0] + 1
    if (len(min_idx) == 0): # if there is only one peak
        return 21 # at least 21 clear obs for the harmonic model
    else:
        return np.argmin(abs(arr_1d - arr_1d[min_idx][-1]))


def save_cluster(ts, out_name, n_clusters=20, n_cpu=-1, method='KMean'):
    if method == 'KMean':
        cls = KMeans(n_clusters, n_jobs=n_cpu)
        labels = cls.fit_predict(ts)
    else:
        print('Not implemented!')
        return False
    pickle.dump(cls, open(out_name, 'wb'))
    return True


def gather_training(acq_datelist, img_stack, outDir=None, base_name='', total_size=200000, save_slice=True):
    """
    This function generates training data from image stacks.

    Parameters
    ----------

    acq_datelist: list
            list of acquisition dates (n_timesteps)
    img_stack: 3d ndarray
            A 3d array contains one year of image time series (n_rows, n_columns, n_timesteps)
            cloud contaminated value should be set as 0 or negative.
    outDir: String
            Specification of output location.
    base_name: String
            Specification of prefix of output file. The output file will look like: basename_data.pkl and basename_model.model
    total_size: int
            For time cost optimization, use only subset of overlap time sereies as training data.
            Set -1 to use all.
    save_slice: bool, optional
            If tree, the image stack will be saved as slice files.
            This is also an input of gap filling function
    """

    obs_clear_count = np.sum(img_stack > 0, axis=2)
    hist, bins = np.histogram(obs_clear_count, bins=range(np.max(obs_clear_count)))

    overlap_thesh = bins[find_lastValley(hist)]
    print('Valley threshold: ', overlap_thesh)
    if max(bins) < overlap_thesh:
        print('Can not find enough clear observations (> 21)')
        return None

    overlap_idx = (obs_clear_count > overlap_thesh)

    obs_clear_overlap = img_stack[overlap_idx]
    obs_clear_overlap_samp = obs_clear_overlap[np.random.permutation(np.sum(overlap_idx))[:total_size], :].T
    del obs_clear_overlap

    training_data = pd.DataFrame(obs_clear_overlap_samp, index=acq_datelist)

    acq_datelist = training_data.index.values
    training_data_fill = lasso_fill(training_data.index.values, training_data.values.T)
    training_data = pd.DataFrame(training_data_fill.T, index=acq_datelist)

    outname = os.path.join(outDir, base_name)
    training_data.to_pickle('{}_data.pkl'.format(outname))
    save_cluster(training_data.values.T, '{}_model.model'.format(outname))

    if save_slice:
        outDir = os.path.join(outDir, 'ts_slice')
        for i_row in range(img_stack.shape[0]):
            obs_clear_path = os.path.join(outDir, '{}.npy'.format(i_row))
            np.save(obs_clear_path, img_stack[i_row, :, :])



def cluster_obs_dis(X, centroids):
    """
    This function calculate distances from time series X to centroids.
    Only valid observations (X >0) will be used in the calculation.

    Parameters
    ----------
    X: ndarray
            Pixel time series to be filled
    centroids: array (n_classes, n_timesteps)
            Center time series of each cluster classes.

    Returns
    -------
    Index of selected samples
    """

    dis = np.zeros((X.shape[0], centroids.shape[0]))
    invalid = (X == 0)
    for i_cls in range(centroids.shape[0]):
        dif = X - np.tile(centroids[i_cls, :], X.shape[0]).reshape(X.shape)
        dif[invalid] = 0
        dis[:, i_cls] = np.sqrt(np.sum(dif**2, axis=1)/(centroids.shape[1]-np.sum(invalid, axis=1)))
    return dis


def huber_impute(ts_y, ts_x, replace=False):
    """
    This function predicts gap data in a time series using Huber regression method.

    Parameters
    ----------
    ts_y: ndarray (n_timesteps)
            Pixel time series to be filled
    ts_x: ndarray (n_samples, n_timesteps)
            Training data for gap filling
    replace: bool optional
            If true, the original clear observations will also be replace by model predictions.
    Returns
    -------
    gap filled time series
    """

    cls = linear_model.HuberRegressor()
    y_valid = (ts_y != 0)
    y_invalid = ~y_valid
    if np.sum(y_valid) > 0:
        model = cls.fit(ts_x[:, y_valid].T, ts_y[y_valid])
        if replace:
            ts_y = model.predict(ts_x.T)
        else:
            ts_y[y_invalid] = model.predict(ts_x[:, y_invalid].T)
        return ts_y
    else:
        return ts_y.fill(np.nan)

def lasso_impute(ts_y, ts_x, replace=False):
    """
    This function predicts gap data in a time series using LASSO regression method.

    Parameters
    ----------
    ts_y: ndarray (n_timesteps)
            Pixel time series to be filled
    ts_x: ndarray (n_samples, n_timesteps)
            Training data for gap filling
    replace: bool optional
            If true, the original clear observations will also be replace by model predictions.
    Returns
    -------
    gap filled time series
    """

    cls = linear_model.Lasso()
    y_valid = (ts_y != 0)
    y_invalid = ~y_valid
    if np.sum(y_valid) > 0:
        model = cls.fit(ts_x[:, y_valid].T, ts_y[y_valid])
        if replace:
            ts_y = model.predict(ts_x.T)
        else:
            ts_y[y_invalid] = model.predict(ts_x[:, y_invalid].T)
        return ts_y
    else:
        return ts_y.fill(np.nan)

def ridge_impute(ts_y, ts_x, replace=False):
    """
    This function predicts gap data in a time series using Ridge regression method.

    Parameters
    ----------
    ts_y: ndarray (n_timesteps)
            Pixel time series to be filled
    ts_x: ndarray (n_samples, n_timesteps)
            Training data for gap filling
    replace: bool optional
            If true, the original clear observations will also be replace by model predictions.
    Returns
    -------
    gap filled time series
    """

    cls = linear_model.Ridge()
    y_valid = (ts_y != 0)
    y_invalid = ~y_valid
    if np.sum(y_valid) > 0:
        model = cls.fit(ts_x[:, y_valid].T, ts_y[y_valid])
        if replace:
            ts_y = model.predict(ts_x.T)
        else:
            ts_y[y_invalid] = model.predict(ts_x[:, y_invalid].T)
        return ts_y
    else:
        return ts_y.fill(np.nan)


def gap_fill_pixel(ts_y, ts_x, labels, centroids, iter_num=10, reg_method='LASSO', sample_size=100):
    """
    This function fills gaps for a pixel time sereis.

    Parameters
    ----------
    ts_y: ndarray
            Pixel time series to be filled
    ts_x: ndarray or matrix (n_sampes, n_timesteps)
            Training data for gap filling
    labels: array (n_classes)
            Label of cluster classes.
    iter_num: int, optional
            Number of predictions to generate for each observation
    reg_method: string, optional
            Regression method for gap filling.
    sample_size: int, optional
            Number of sampled training data for gap filling.
    Returns
    -------
    impute_y : gap filled ts_y time sereis
    impute_y_std : standard deviation of predictions for each gap filled observation
    """
    impute_y = np.zeros_like(ts_y, dtype=np.int)
    impute_y_std = np.zeros_like(ts_y, dtype=np.float)
    if np.all(ts_y > 0): #if no missing data
        return ts_y, impute_y_std
    elif np.all(ts_y == 0): #if no valid data
        return impute_y, impute_y_std

    ts_temp = []
    for i_ter in range(iter_num):
        sample_idx = sampling_strata(ts_y, centroids, labels, sample_size=sample_size)
        if reg_method == 'LASSO':
            ts_temp.append(lasso_impute(np.copy(ts_y), ts_x[sample_idx, :], replace=False))
        elif reg_method == 'Ridge':
            ts_temp.append(ridge_impute(np.copy(ts_y), ts_x[sample_idx, :],replace=True))
        elif reg_method == 'Huber':
            ts_temp.append(huber_impute(np.copy(ts_y), ts_x[sample_idx, :]))
        else:
            print('Can not find the sepcified regression method! Please use LASSO, Ridge, or Huber.')

    ts_temp = np.array(ts_temp)
    impute_y = np.nanmedian(ts_temp, axis=0).astype(int)
    impute_y_std = np.nanstd(ts_temp, axis=0)

    return impute_y, impute_y_std

def gap_fill_slice(outDir, ts_path, ts_x, labels, centroids, reg_kws):
    """
    This function fills gaps for a slice of time series.
    For example, in a (nRow, nCol, nTimeSteps) image stack, a slice of time series means (nCol, nTimeSteps) ndarray

    Parameters
    ----------
    outDir: String
            Specification of output location.
    ts_path: string
            Specification of slice time series location.
    ts_x: ndarray or matrix (n_samples, n_timesteps)
            Training data for gap filling
    labels: array (n_samples)
            Label of cluster classes.
    centroids: array (n_classes, n_timesteps)
            Center time series of each cluster classes.
    reg_kws: dict, optional
            Keyword arguments for :func:`gap_fill_pixel`.
    """
    if reg_kws:
        iter_num = reg_kws["iter_num"]
        reg_method = reg_kws["reg_method"]
        sample_size = reg_kws["sample_size"]
    else:
        iter_num = 10
        reg_method = 'LASSO'
        sample_size = 100

    name, ext = os.path.splitext(os.path.basename(ts_path))
    ts_y_line = np.load(ts_path)
    impute_y = np.apply_along_axis(gap_fill_pixel, -1, ts_y_line, ts_x, labels, centroids, iter_num=iter_num,
                                   reg_method=reg_method, sample_size=sample_size)

    np.save(os.path.join(outDir, '{}_{}.npy'.format(name, 'gapfilled')), impute_y[:, 0])
    np.save(os.path.join(outDir, '{}_{}.npy'.format(name, 'gapfilled_STD')), impute_y[:, 1])

def sampling_strata(ts_y, centroids, labels, sample_size=100):
    """
    This function stratify samples of training data based on target pixel time sereis.

    Parameters
    ----------
    ts_y: ndarray
            Pixel time series to be filled
    centroids: array (n_classes, n_timesteps)
            Center time series of each cluster classes.
    labels: array (n_classes)
            Label of cluster classes.
    sample_size : int
            Number of sampled training data for gap filling

    Returns
    -------
    Index of selected samples
    """

    weight = 1/cluster_obs_dis(ts_y[np.newaxis, :], centroids)
    weight = weight / np.nanmax(weight)
    #if ts_y is at one of the centroids
    weight_idx = np.isfinite(weight)
    # weight[~weight_idx] = 2*np.nanmax(weight)
    weight[~weight_idx] = 1.0

    prob = np.take(weight, labels)
    if np.sum(prob) > 0:
        prob = prob / np.sum(prob)
        idx = np.random.choice(np.asarray(range(len(labels))), sample_size, replace=False, p=prob)
        return idx
    else:
        print('diag: ts_y, weight_idx, weight, prob', ts_y, weight_idx, weight, prob)
        print('total probability equal to 0!')
        return None

def fill_gaps_batch(slice_list, acq_datelist, training_data_path, cluster_model_path, outDir=None, cpu=20, reg_kws=None):
    """
    This function fills gaps for a slice of time series.

    Parameters
    ----------

    slice_list: list
            Specification of list of slice time series file.
            The slice time series file should be save in npy format.
            Each file saves a slice of time series (n_pixels, n_timesteps)
    acq_datelist: list
            list of acquisition dates (n_timesteps)
    training_data_path: string
            Collection of no-gap time series data as training data
    cluster_model_path: string
            Cluster model based on the training data
    outDir: String
            Specification of output location.
    cpu:
            Number of cpu
    reg_kws: dict, optional
            Keyword arguments for :func:`gap_fill_pixel`.

        Examples
    --------
    n_cpu = 20
    dates_list = np.load(os.path.join(work_dir, 'dates.npy'))
    outname = os.path.join(work_dir, 'training')
    training_data_path = '{}.pkl'.format(outname)
    cluster_model_path = '{}.model'.format(outname)
    fill_gaps(file_list, dates_list, training_data_path, cluster_model_path, cpu=n_cpu)
    """

    reg_kws = {} if reg_kws is None else reg_kws.copy()
    if outDir is None:
        outDir = os.path.dirname(slice_list[0])
    if not os.path.exists(outDir):  ## if outfolder is not already existed creating one
        os.makedirs(outDir)

    if os.path.exists(training_data_path):
        training_data = pd.read_pickle(training_data_path)
        date_idx = training_data.index.isin(acq_datelist)
        training_data = training_data[date_idx].values.T
    else:
        print(training_data_path)
        print('Can not find training data! Please save samples from overlap pixels.')
        return None

    if os.path.exists(cluster_model_path):
        cluster_model = pickle.load(open(cluster_model_path, 'rb'))
        labels = cluster_model.labels_
        centroids = cluster_model.cluster_centers_
        centroids = centroids[:, date_idx]
    else:
        print('Can not find cluster model!')
        return None

    print('Imputing')
    for i_slice in np.arange(0, len(slice_list), cpu):
        start_slice = i_slice
        end_slice = min(start_slice+cpu, len(slice_list))
        jobs = []
        for i_step in range(start_slice, end_slice):
            print('Processing line: ', i_step)
            ts_path = slice_list[i_step]
            print('training_data.shape, centroids.shape: ', training_data.shape, centroids.shape)
            if not os.path.exists(ts_path):
                print('Can not find file: ', ts_path)
                continue
            p = Process(target=gap_fill_slice, args=(outDir, ts_path,
                                                          training_data, labels, centroids, reg_kws))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()



def saveGeoTiff(filename, data, dim, trans, prj, dtype=gdal.GDT_Float32):
    if os.path.exists(filename):
        os.remove(filename)

    outds = gdal.GetDriverByName('GTiff').Create(filename, dim[0], dim[1], dim[2], dtype)
    outds.SetGeoTransform(trans)
    outds.SetProjection(prj)
    if dim[2] == 1:
        outds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(dim[2]):
            outds.GetRasterBand(i + 1).WriteArray(data[:, :, i])
    outds = None
    return True


def slice2map(line_list, name_list, outDir, trans, prj):
    """
    This function generates gap filled slice files to tiff images.

    Parameters
    ----------

    line_list: list
            list of slice file
    name_list: list
            list of output images (n_timesteps). Each acquisition date should have an individual name
    outDir: String
            Specification of output location.
    trans: list
            Transform of output image
    prj: string
            Projection of output image
    """
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    temp_line = np.load(line_list[0])
    img_layer = np.zeros((len(line_list), len(temp_line)))
    n_layers = temp_line.shape[1]
    dim = [len(temp_line), len(line_list), 1]
    for i_layer in range(n_layers):
        print('Processing layer: ', i_layer)
        for i_line, line in enumerate(line_list):
            if os.path.exists(line):
                img_layer[i_line, :] = np.load(line)[:, i_layer]
        saveGeoTiff(name_list[i_layer], img_layer, dim, trans, prj)


