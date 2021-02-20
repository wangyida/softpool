from math import pi
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class PointNetFeat(nn.Module):
    def __init__(self, num_points=8192, dim_pn=1024):
        super(PointNetFeat, self).__init__()
        self.dim_pn = dim_pn
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_pn, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(dim_pn)

        self.fourier_map1 = Periodics(dim_input=3, dim_output=32)
        self.fourier_map2 = Periodics(
            dim_input=32, dim_output=128, is_first=False)
        self.fourier_map3 = Periodics(
            dim_input=128, dim_output=128, is_first=False)

        self.num_points = num_points

    def forward(self, inputs):
        """
        x = self.fourier_map1(inputs)
        x = self.fourier_map2(x)
        x = self.fourier_map3(x)
        """
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.dim_pn)
        return x

