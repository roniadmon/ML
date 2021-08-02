import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import os
from scipy import stats

os.chdir(os.getcwd())
fieldname = '_hourly_200902-201912.npy'
x1_arr = np.load('z1000'+fieldname) # geopotential height data (9*9 resolution)
x2_arr = np.load('ta100'+fieldname) # potential vorticity data (9*9 resolution)
x3_arr = np.load('z500'+fieldname) # geopotential height data (9*9 resolution)

x1_arr_flat = stats.zscore(x1_arr.reshape([x1_arr.shape[0],x1_arr.shape[1]*x1_arr.shape[2]]))
x2_arr_flat = stats.zscore(x2_arr.reshape([x2_arr.shape[0],x2_arr.shape[1]*x2_arr.shape[2]]))
x3_arr_flat = stats.zscore(x2_arr.reshape([x2_arr.shape[0],x2_arr.shape[1]*x2_arr.shape[2]]))
y_arr = np.load('rain__hourly_20090201-20191231.npy') # rain data

tensor_x = torch.Tensor(np.concatenate([x1_arr_flat,x2_arr_flat,x3_arr_flat],axis=1))
tensor_y = torch.Tensor(y_arr)

forecast_dataset = TensorDataset(tensor_x,tensor_y) # creates a dataset based on tensors
forecast_dataloader = DataLoader(forecast_dataset)
