import torch
import numpy as np
import os
root_dir = '../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param/'
data_list_cdm = [ f for f in os.listdir(root_dir) if f.endswith('.npy')]
#print(data_list_cdm)
data_file_path = os.path.join(root_dir, data_list_cdm[0])
data = np.load(data_file_path, allow_pickle=True)
print(data[5])