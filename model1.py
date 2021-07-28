import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from scipy import stats
import numpy as np
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(9, 69)
            , nn.ReLU(inplace=True)
            , nn.Linear(69, 20)
            , nn.ReLU(inplace=True)
            , nn.Linear(20, 10)
            , nn.ReLU(inplace=True)
            , nn.Linear(10, 4));

        self.classifier = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 4));

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
