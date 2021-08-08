import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm1d,Dropout

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.regressor = nn.Sequential(
            Linear(6*6*3,128),
            Dropout(0.5),
            ReLU(inplace=True),
            Linear(128,64),
            ReLU(inplace=True),
            Linear(64,32),
            ReLU(inplace=True),
            Linear(32,2)
        )

        
    def forward(self,x):
        xflat = torch.flatten(x,1,-1)
        reg_result = self.regressor(xflat)
        return reg_result