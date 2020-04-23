import ard_gap_filler as gf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    ##############
    # Example of pixel time series fill
    # A quick and easy way to check the pixel results
    ######################
    work_dir = r'test/resources/'
    n_cpu = 1
    dates_list = np.load(os.path.join(work_dir, 'dates.npy'), allow_pickle=True)
    outname = os.path.join(work_dir, 'training_Atlanta')
    training_data_path = '{}.pkl'.format(outname)
    cluster_model_path = '{}.model'.format(outname)
    training_data = pd.read_pickle(training_data_path)
    date_idx = training_data.index.isin(dates_list)
    training_data = training_data[date_idx].values.T
    cluster_model = pickle.load(open(cluster_model_path, 'rb'))
    labels = cluster_model.labels_
    centroids = cluster_model.cluster_centers_
    centroids = centroids[:, date_idx]
    ts_org = np.load(r'test/resources/data/clear_full_2591.npy')
    ts_y = ts_org[800, :]
    gap_filled, gap_filled_std = gf.gap_fill_pixel(ts_y, training_data, labels, centroids)
    
    # plot the results #
    ts_y = np.ma.masked_array(ts_y, mask=(ts_y == 0))
    plt.scatter(dates_list, gap_filled, marker='o', alpha=0.7, label='Gap filled time series')
    plt.scatter(dates_list, ts_y, marker='+', alpha=1.0, label='Orginal time series')
    plt.legend()
    plt.show()
    #####################################################################

    ###################
    # Example of slice time series fill
    # This may take several hours depending on how many computing resource
    # #####################
    # work_dir = r'test/resources/'
    # outDir = r'test/resources/data_export'
    # if not os.path.exists(outDir):
    #     os.mkdir(outDir)
    #
    # file_list = []
    # for name in glob.glob('test/resources/data/*.npy'):
    #     file_list.append(name)
    #
    # n_cpu = 1
    # dates_list = np.load(os.path.join(work_dir, 'dates.npy'), allow_pickle=True)
    # outname = os.path.join(work_dir, 'training_Atlanta')
    # training_data_path = '{}.pkl'.format(outname)
    # cluster_model_path = '{}.model'.format(outname)
    # gf.fill_gaps_batch(file_list, dates_list, training_data_path, cluster_model_path, cpu=n_cpu, outDir=outDir)
    # # plot example result
    # dates_list = np.load(os.path.join(work_dir, 'dates.npy'), allow_pickle=True)
    # ts_org = np.load(os.path.join(work_dir, 'data', 'clear_full_2500.npy'))
    # ts_filled = np.load(os.path.join(outDir, 'clear_full_2500_gapfilled.npy'))
    # ts_org = np.ma.masked_array(ts_org, mask=(ts_org == 0))
    # print('ts_org[100, :], ', ts_org[100, :])
    # print(ts_org.shape, dates_list)
    # print('ts_filled[100, :], ', ts_filled[100, :])
    # plt.scatter(dates_list, ts_filled[100, :], marker='o', alpha=0.7, label='Gap filled time series')
    # plt.scatter(dates_list, ts_org[100, :], marker='+', alpha=1.0, label='Orginal time series')
    # plt.legend()
    # plt.show()
    ####################################################


