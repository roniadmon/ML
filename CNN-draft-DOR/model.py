
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm2d

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        #VGG option
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True))

        # regular CNN
        self.features_orig = nn.Sequential(
            Conv2d(2,20,3,padding=1,bias=False),
            BatchNorm2d(20),
            ReLU(inplace=True),
           
            Conv2d(20,20,3,padding=1,bias=False),
            BatchNorm2d(20),
            ReLU(inplace=True),
           
            Conv2d(20,20,3,padding=1,bias=False),
            BatchNorm2d(20),
            ReLU(inplace=True),
            
            Conv2d(20,1,3,padding=1,bias=False)
        )
        
        # for VGG
        self.classifier = nn.Sequential(
            Linear(2*2*256,512),
            ReLU(inplace=True),
            Linear(512,128),
            ReLU(inplace=True),
            Linear(128,4)
        )
        
        # for CNN
        self.regressor = nn.Sequential(
            Linear(81,100),
            ReLU(inplace=True),
            Linear(100,200),
            ReLU(inplace=True),
            Linear(200,50),
            ReLU(inplace=True),
            Linear(50,4)
        )

        
    def forward(self,x):
        out = self.features_orig(x) 
        outflat = torch.flatten(out,1)
        reg_result = self.regressor(outflat)
        return reg_result