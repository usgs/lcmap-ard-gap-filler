import gap_fill_ard as gfa

import os
import glob
import numpy as np


work_dir = r'test/resources/'
acq_datelist = np.load(os.path.join(work_dir, 'dates.npy'), allow_pickle=True)
file_list = []
for name in glob.glob(os.path.join(work_dir, 'data', '*.npy')):
    file_list.append(name)
temp_arr = np.load(file_list[0])
img_stack = np.zeros((len(file_list), temp_arr.shape[0], temp_arr.shape[1]))
file_list.sort()
for i_row, f in enumerate(file_list):
    img_stack[i_row, :, :] = np.load(f)
gfa.gather_training(acq_datelist, img_stack, outDir=work_dir, total_size=200, save_slice=False)