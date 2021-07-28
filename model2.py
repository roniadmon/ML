
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            ,Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,ReLU(inplace=True)
            , MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            , Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            , ReLU(inplace=True)
            , Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            , ReLU(inplace=True)
            , MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            , Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            , ReLU(inplace=True)
            , Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            , ReLU(inplace=True)
            , MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.classifier = nn.Sequential(
            Linear(512*2*2,256),
            ReLU(inplace=True),
            Linear(256,100),
            ReLU(inplace=True),
            Linear(100,10)
        );

    def forward(self, x):
        
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
