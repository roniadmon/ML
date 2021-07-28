import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import os
from scipy import stats

os.chdir(os.getcwd())
fieldname = '_19790101-20190228.npy'
x1_arr = np.load('z1000'+fieldname) # geopotential height data (9*9 resolution)
x2_arr = np.load('pv300'+fieldname) # potential vorticity data (9*9 resolution)

x1_arr_flat = stats.zscore(x1_arr.reshape([x1_arr.shape[0],x1_arr.shape[1]*x1_arr.shape[2]])) # normalize and flatten
x2_arr_flat = stats.zscore(x2_arr.reshape([x2_arr.shape[0],x2_arr.shape[1]*x2_arr.shape[2]]))
y_arr = np.load('rain_basin_19790101-20190228.npy') # rain data

tensor_x = torch.Tensor(np.concatenate([x1_arr_flat,x2_arr_flat],axis=1)) # join z and pv data
tensor_y = torch.Tensor(y_arr)

forecast_dataset = TensorDataset(tensor_x,tensor_y) # creates a dataset based on tensors
forecast_dataloader = DataLoader(forecast_dataset)
