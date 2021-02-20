import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class Periodics(nn.Module):
    def __init__(self, dim_input=2, dim_output=512, is_first=True):
        super(Periodics, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.is_first = is_first
        self.with_frequency = True
        self.with_phase = True
        # Omega determines the upper frequencies
        self.omega_0 = 30
        if self.with_frequency:
            if self.with_phase:
                self.Li = nn.Conv1d(
                    self.dim_input, self.dim_output, 1,
                    bias=self.with_phase).cuda()
            else:
                self.Li = nn.Conv1d(
                    self.dim_input,
                    self.dim_output // 2,
                    1,
                    bias=self.with_phase).cuda()
            # nn.init.normal_(B.weight, std=10.0)
            with torch.no_grad():
                if self.is_first:
                    self.Li.weight.uniform_(-1 / self.dim_input,
                                            1 / self.dim_input)
                else:
                    self.Li.weight.uniform_(
                        -np.sqrt(6 / self.dim_input) / self.omega_0,
                        np.sqrt(6 / self.dim_input) / self.omega_0)
        else:
            self.Li = nn.Conv1d(self.dim_input, self.dim_output, 1).cuda()
            self.BN = nn.BatchNorm1d(self.dim_output).cuda()

    def filter(self):
        filters = torch.cat([
            torch.ones(1, self.dim_output // 32 * 32),
            torch.zeros(1, self.dim_output // 32 * 0)
        ], 1).cuda()
        filters = torch.unsqueeze(filters, 2)
        return filters

    def forward(self, x):
        # here are some options to check how to form the fourier feature
        lp_filter = self.filter()
        if self.with_frequency:
            if self.with_phase:
                sinside = torch.sin(self.Li(x) * self.omega_0)
                return sinside
            else:
                """
                here filter could be applied
                """
                sinside = torch.sin(self.Li(x) * self.omega_0)
                cosside = torch.cos(self.Li(x) * self.omega_0)
                return torch.cat([sinside, cosside], 1)
        else:
            return F.relu(self.BN(self.Li(x)))


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
    def __init__(self, regions=16, cabins=8, sp_ratio=4, size_feat=256):
        super(SoftPool, self).__init__()
        self.regions = regions
        self.num_cabin = cabins
        self.sp_ratio = sp_ratio
        self.size_feat = size_feat

        self.conv2d_1 = nn.Conv2d(
            self.size_feat, self.size_feat, kernel_size=(1, 3),
            stride=(1, 1)).cuda()
        # cabin -2
        self.conv2d_2 = nn.Conv2d(
            self.size_feat, self.size_feat, kernel_size=(1, 3),
            stride=(1, 1)).cuda()
        self.conv2d_3 = nn.Conv2d(
            self.size_feat,
            self.size_feat,
            kernel_size=(1, self.num_cabin - 2 * (3 - 1)),
            stride=(1, 1)).cuda()
        self.conv2d_5 = nn.Conv2d(
            self.size_feat,
            self.size_feat,
            kernel_size=(self.regions, 1),
            stride=(1, 1)).cuda()

        self.sorter = Sorter(self.size_feat, self.regions)

    def forward(self, x):
        [self.size_bth, self.size_feat, self.pnt_per_sort] = list(x.shape)
        self.pnt_per_sort //= self.sp_ratio

        val_activa, id_activa = self.sorter(x)

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
        trains = self.conv2d_3(self.conv2d_2(self.conv2d_1(cabins)))
        # we need to use succession manner to repeat cabin to fit with cube
        sp_trains = trains.repeat(1, 1, 1, self.pnt_per_sort)

        # now make a station
        station = self.conv2d_5(trains)
        sp_station = station.repeat(1, 1, self.regions, self.pnt_per_sort)

        scope = 'local'
        if scope == 'global':
            sp_cube = torch.cat((sp_cube, sp_windows, sp_trains, sp_station),
                                1).contiguous()

        return sp_cube, sp_idx, cabins, id_activa

class SoftPoolFeat(nn.Module):
    def __init__(self, num_points=8192, regions=16, sp_points=2048,
                 sp_ratio=8):
        super(SoftPoolFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        """
        self.fourier_map1 = Periodics(dim_input=3, dim_output=32)
        self.fourier_map2 = Periodics(
            dim_input=32, dim_output=128, is_first=False)
        self.fourier_map3 = Periodics(
            dim_input=128, dim_output=128, is_first=False)
        """

        self.num_points = num_points
        self.regions = regions
        self.sp_points = sp_points // sp_ratio

        self.softpool = SoftPool(self.regions, cabins=8, sp_ratio=sp_ratio)

    def mlp(self, inputs):
        """
        x = self.fourier_map1(inputs)
        x = self.fourier_map2(x)
        x = self.fourier_map3(x)
        """
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

    def forward(self, x, x_seg=None):
        part = x

        x = self.mlp(x)

        sp_cube, sp_idx, cabins, id_activa = self.softpool(x)

        # transform
        id_activa = torch.nn.functional.one_hot(
            id_activa.to(torch.int64), self.regions).transpose(1, 2)
        if x_seg is None:
            point_wi_seg = torch.cat((id_activa.float(), part), 1)
        else:
            point_wi_seg = torch.cat((x_seg.float(), part), 1)

        """
        point_wi_seg = point_wi_seg.transpose(2, 1)
        point_wi_seg = torch.bmm(point_wi_seg, trans)
        point_wi_seg = point_wi_seg.transpose(2, 1)
        """
        point_wi_seg = point_wi_seg.unsqueeze(2).repeat(1, 1, self.regions, 1)

        point_wi_seg = torch.gather(point_wi_seg, dim=3, index=sp_idx.long())
        feature = torch.cat((sp_cube, point_wi_seg), 1).contiguous()

        feature = feature.view(feature.shape[0], feature.shape[1], 1,
                               self.regions * self.sp_points)
        sp_cube = sp_cube.view(sp_cube.shape[0], sp_cube.shape[1], 1,
                               self.regions * self.sp_points)
        sp_idx = sp_idx.view(sp_idx.shape[0], sp_idx.shape[1], 1,
                             self.regions * self.sp_points)
        # return feature, cabins, sp_idx
        return sp_cube, cabins, sp_idx
