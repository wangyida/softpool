import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# Produce a set of pointnet features in several sorted cloud
def train2cabins(windows, num_cabin=8):
    size_bth = list(windows.shape)[0]
    size_feat = list(windows.shape)[1]
    regions = list(windows.shape)[2]
    num_points = list(windows.shape)[3]
    cabins = torch.zeros(size_bth, size_feat, regions, num_cabin).cuda()
    points_cabin = num_points // num_cabin
    for idx in range(num_cabin):
        cabins[:, :, :, idx] = torch.max(
            windows[:, :, :, idx * points_cabin:(idx + 1) * points_cabin],
            dim=3,
            keepdim=False)[0]

    return cabins


class Sorter(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Sorter, self).__init__()
        self.conv1d = torch.nn.Conv1d(dim_in, dim_out, 1).cuda()

    def forward(self, x):
        val_activa = self.conv1d(x)
        id_activa = torch.argmax(val_activa, dim=1)
        return val_activa, id_activa


class SoftPool(nn.Module):
    def __init__(self, regions=16, cabins=8, sp_ratio=4):
        super(SoftPool, self).__init__()
        self.regions = regions
        self.num_cabin = cabins
        self.sp_ratio = sp_ratio

    def forward(self, x):
        [self.size_bth, self.size_feat, self.pnt_per_sort] = list(x.shape)
        self.pnt_per_sort //= self.sp_ratio
        # cabin -2
        conv2d_1 = nn.Conv2d(
            self.size_feat, self.size_feat, kernel_size=(1, 3),
            stride=(1, 1)).cuda()
        # cabin -2
        conv2d_2 = nn.Conv2d(
            self.size_feat, self.size_feat, kernel_size=(1, 3),
            stride=(1, 1)).cuda()
        conv2d_3 = nn.Conv2d(
            self.size_feat,
            self.size_feat,
            kernel_size=(1, self.num_cabin - 2 * (3 - 1)),
            stride=(1, 1)).cuda()
        conv2d_5 = nn.Conv2d(
            self.size_feat,
            self.size_feat,
            kernel_size=(self.regions, 1),
            stride=(1, 1)).cuda()

        sorter = Sorter(self.size_feat, self.regions)
        val_activa, id_activa = sorter(x)

        # initialize empty space for softpool feature
        sp_cube = torch.zeros(self.size_bth, self.size_feat, self.regions,
                              self.pnt_per_sort).cuda()
        sp_idx = torch.zeros(self.size_bth, self.regions + 3, self.regions,
                             self.pnt_per_sort).cuda()

        for region in range(self.regions):
            x_val, x_idx = torch.sort(
                val_activa[:, region, :], dim=1, descending=True)
            index = x_idx[:, :self.pnt_per_sort].unsqueeze(1).repeat(
                1, self.size_feat, 1)

            sp_cube[:, :, region, :] = torch.gather(x, dim=2, index=index)
            sp_idx[:, :, region, :] = x_idx[:, :self.pnt_per_sort].unsqueeze(
                1).repeat(1, self.regions + 3, 1)

        # local pointnet feature
        points_cabin = self.pnt_per_sort // self.num_cabin
        cabins = train2cabins(sp_cube, self.num_cabin)

        # we need to use succession manner to repeat cabin to fit with cube
        sp_windows = torch.repeat_interleave(
            cabins, repeats=points_cabin, dim=3)

        # merge cabins in train
        trains = conv2d_3(conv2d_2(conv2d_1(cabins)))
        # we need to use succession manner to repeat cabin to fit with cube
        sp_trains = trains.repeat(1, 1, 1, self.pnt_per_sort)

        # now make a station
        station = conv2d_5(trains)
        sp_station = station.repeat(1, 1, self.regions, self.pnt_per_sort)

        scope = 'local'
        if scope == 'global':
            sp_cube = torch.cat((sp_cube, sp_windows, sp_trains, sp_station),
                                1).contiguous()

        return sp_cube, sp_idx, cabins, id_activa
