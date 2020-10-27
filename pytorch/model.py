from __future__ import print_function
from math import pi
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion
sys.path.append("./MDS/")
import MDS_module


def SoftPool(x, regions=8):
    bth_size = list(x.shape)[0]
    featdim = list(x.shape)[1]
    num_points = list(x.shape)[2]

    # Reduce dimention to sort
    conv1d = torch.nn.Conv1d(featdim, regions, 1).cuda()
    sorter = conv1d(x)

    # initialize empty space for softpool feature
    sp_cube = torch.zeros(bth_size, featdim, regions, num_points).cuda()
    sp_idx = torch.zeros(bth_size, 12 + 3, regions, num_points).cuda()

    for idx in range(regions):
        x_val, x_idx = torch.sort(conv1d(x)[:, idx, :], dim=1, descending=True)
        index = x_idx[:, :].unsqueeze(1).repeat(1, featdim, 1)
        x_order = torch.gather(x, dim=2, index=index)
        sp_cube[:, :, idx, :] = x_order
        sp_idx[:, :, idx, :] = x_idx[:, :].unsqueeze(1).repeat(1, 12 + 3, 1)

    # local pointnet feature
    num_cabin = 16
    points_cabin = num_points // num_cabin
    cabins = Cabins(sp_cube, num_cabin)

    # we need to use succession manner to repeat cabin to fit with cube
    sp_windows = torch.repeat_interleave(cabins, repeats=points_cabin, dim=3)

    # merge cabins in train
    # cabin -4
    conv2d_1 = nn.Conv2d(
        featdim, featdim, kernel_size=(1, 5), stride=(1, 1)).cuda()
    # cabin -4
    conv2d_2 = nn.Conv2d(
        featdim, featdim, kernel_size=(1, 5), stride=(1, 1)).cuda()
    # cabin -4
    conv2d_3 = nn.Conv2d(
        featdim,
        featdim,
        kernel_size=(1, 5),
        stride=(1, 1),
    ).cuda()
    conv2d_4 = nn.Conv2d(
        featdim,
        featdim,
        kernel_size=(1, num_cabin - 3 * (5 - 1)),
        stride=(1, 1)).cuda()
    trains = conv2d_4(conv2d_3(conv2d_2(conv2d_1(cabins))))
    # we need to use succession manner to repeat cabin to fit with cube
    sp_trains = trains.repeat(1, 1, 1, num_points)

    # now make a station
    conv2d_5 = nn.Conv2d(
        featdim, featdim, kernel_size=(regions, 1), stride=(1, 1)).cuda()
    station = conv2d_5(trains)
    sp_station = station.repeat(1, 1, regions, num_points)

    sp_cube = torch.cat((sp_cube, sp_windows, sp_trains, sp_station),
                        1).contiguous()

    return sp_cube, sp_idx, cabins


# Produce a set of pointnet features in several sorted cloud
def Cabins(windows, num_cabin=16):
    bth_size = list(windows.shape)[0]
    featdim = list(windows.shape)[1]
    regions = list(windows.shape)[2]
    num_points = list(windows.shape)[3]
    cabins = torch.zeros(bth_size, featdim, regions, num_cabin).cuda()
    points_cabin = num_points // num_cabin
    for idx in range(num_cabin):
        cabins[:, :, :, idx] = torch.max(
            windows[:, :, :, idx * points_cabin:(idx + 1) * points_cabin],
            dim=3,
            keepdim=False)[0]

    return cabins


class STN3d(nn.Module):
    def __init__(self, dim_pn=1024):
        super(STN3d, self).__init__()
        self.dim_pn = dim_pn
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.dim_pn, 1)
        self.fc1 = nn.Linear(self.dim_pn, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.dim_pn)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(
                np.array([1, 0, 0, 0, 1, 0, 0, 0,
                          1]).astype(np.float32))).view(1, 9).repeat(
                              batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=3 + 12):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(np.eye(self.k).flatten().astype(
                np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class PointNetFeat(nn.Module):
    def __init__(self, num_points=8192, dim_pn=1024):
        super(PointNetFeat, self).__init__()
        self.stn = STNkd(k=3 + 12)
        self.dim_pn = dim_pn
        self.conv1 = torch.nn.Conv1d(3 + 12, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_pn, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(dim_pn)

        self.num_points = num_points

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.dim_pn)
        return x


def fourier_map(x, dim_input=2):
    B = nn.Linear(dim_input, 256)
    nn.init.normal_(B.weight, std=10.0)
    B.weight.requires_grad = False
    sinside = torch.sin(2 * pi * B(x.transpose(2, 1)))
    cosside = torch.cos(2 * pi * B(x.transpose(2, 1)))
    return torch.cat([sinside, cosside], -1).transpose(2, 1)


class SoftPoolFeat(nn.Module):
    def __init__(self, num_points=8192, regions=64, sp_points=256):
        super(SoftPoolFeat, self).__init__()
        self.stn = STNkd(k=12 + 3)
        self.conv1 = torch.nn.Conv1d(12 + 3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.num_points = num_points
        self.regions = regions
        self.sp_points = sp_points

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        part = x.unsqueeze(2).repeat(1, 1, self.regions, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        sp_cube, sp_idx, cabins = SoftPool(x, self.regions)
        # 2048 / 63 = 32
        idx_step = torch.floor(
            torch.linspace(0, (sp_cube.shape[3] - 1), steps=self.sp_points))
        sp_cube = sp_cube[:, :, :, :self.sp_points]
        # sp_idx = sp_idx[:, :, :, :self.sp_points]
        # x = x[:, :, :, idx_step.long()]
        # sp_idx = sp_idx[:, :, :, idx_step.long()]
        part = torch.gather(part, dim=3, index=sp_idx.long())
        feature = torch.cat((sp_cube, part), 1).contiguous()
        return feature, cabins, sp_idx, trans


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,
                                     self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,
                                     self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2,
                                     self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.th(self.conv4(x))
        x = self.conv4(x)
        return x


class PointGenCon2D(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon2D, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            self.bottleneck_size,
            self.bottleneck_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            padding_mode='same')
        self.conv2 = torch.nn.Conv2d(
            self.bottleneck_size,
            self.bottleneck_size // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            padding_mode='same')
        self.conv3 = torch.nn.Conv2d(
            self.bottleneck_size // 2,
            self.bottleneck_size // 4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            padding_mode='same')
        self.conv4 = torch.nn.Conv2d(
            self.bottleneck_size // 4,
            3,
            kernel_size=(8, 1),
            stride=(1, 1),
            padding=(0, 0),
            padding_mode='same')

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm2d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm2d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm2d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.th(self.conv4(x))
        x = self.conv4(x)
        return x


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            4, 64, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv2 = torch.nn.Conv1d(
            64, 128, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv3 = torch.nn.Conv1d(
            128, 1024, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv4 = torch.nn.Conv1d(
            1088, 512, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv5 = torch.nn.Conv1d(
            512, 256, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv6 = torch.nn.Conv1d(
            256, 128, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv7 = torch.nn.Conv1d(
            128, 3, kernel_size=5, padding=2, padding_mode='replicate')

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x


class MSN(nn.Module):
    def __init__(self,
                 num_points=8192,
                 n_primitives=16,
                 dim_pn=256,
                 sp_points=1024):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.dim_pn = dim_pn
        self.n_primitives = n_primitives
        self.sp_points = sp_points
        self.pncoder = nn.Sequential(
            PointNetFeat(num_points, 1024), nn.Linear(1024, dim_pn),
            nn.BatchNorm1d(dim_pn), nn.ReLU())
        # self.spcoder = SoftPoolFeat(num_points, regions=self.n_primitives, sp_points=self.sp_points)
        self.spcoder = SoftPoolFeat(
            num_points, regions=self.n_primitives, sp_points=2048)
        # Firstly we do not merge information among regions
        # We merge regional informations in latent space
        self.ptmapper = nn.Sequential(
            nn.Conv2d(
                4 * dim_pn + 12 + 3,
                dim_pn,
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                dim_pn,
                2 * dim_pn,
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(self.n_primitives, 1), 
                stride=(1, 1)),
            nn.ConvTranspose2d(
                2 * dim_pn,
                dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)),
            nn.ConvTranspose2d(
                dim_pn,
                dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)))
        """
            nn.Linear(self.sp_points, self.sp_points),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Linear(self.sp_points // 2, self.sp_points // 2),
            nn.ReLU(),
            nn.Linear(self.sp_points // 2, self.sp_points))
        """
        """
            nn.Conv2d(
                n_primitives,
                n_primitives,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                n_primitives,
                2 * n_primitives,
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                2 * n_primitives,
                2 * n_primitives,
                kernel_size=(1, 7),
                stride=(1, 1),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh(),
            nn.ConvTranspose2d(
                2 * n_primitives,
                n_primitives,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)),
        """
        # nn.Flatten(start_dim=2, end_dim=3))
        """
        self.decoder1 = nn.ModuleList([
            PointGenCon(bottleneck_size=self.dim_pn)
            for i in range(0, self.n_primitives)
        ])
        self.decoder2 = nn.ModuleList([
            PointGenCon(bottleneck_size=2 + 2 * self.dim_pn)
            for i in range(0, self.n_primitives)
        ])
        """
        # self.decoder3 = PointGenCon(bottleneck_size=2 + self.dim_pn)
        self.decoder1 = PointGenCon(bottleneck_size=self.dim_pn)
        self.decoder2 = PointGenCon(bottleneck_size=2 + 2 * self.dim_pn)
        self.decoder3 = PointGenCon(bottleneck_size=2 * 256 + self.dim_pn)
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, part, part_seg):

        # part_seg -> one hot coding
        part_seg = part_seg[:, :, 0]
        part_seg = torch.nn.functional.one_hot(part_seg.to(torch.int64),
                                               12).transpose(1, 2)

        sp_feat, sp_cabins, sp_idx, trans = self.spcoder(
            torch.cat((part_seg.float(), part), 1))
        loss_trans = feature_transform_regularizer(trans)
        pn_feat = self.pncoder(torch.cat((part_seg.float(), part), 1).float())
        pn_feat = pn_feat.unsqueeze(2).expand(
            part.size(0), self.dim_pn, self.num_points).contiguous()
        sp_feat_conv = self.ptmapper(sp_feat)
        out_sp_local = []
        out_seg = []
        out_sp_global = []
        out_pcn = []
        # for i in range(0, self.n_primitives):
        """
        part_regions.append(
            torch.gather(part, dim=2, index=sp_idx[:, :, i, :].long()))
        """
        # stn3d

        rand_grid = Variable(
            torch.cuda.FloatTensor(
                part.size(0), 2, self.num_points // 16 // 8))
        rand_grid.data.uniform_(0, 1)
        # here self.num_points // self.n_primitives = 8*4

        mesh_grid = torch.meshgrid([
            torch.linspace(0.0, 1.0, 64),
            torch.linspace(0.0, 1.0, self.num_points // 64)
        ])
        mesh_grid = torch.cat(
            (torch.reshape(mesh_grid[0], (self.num_points, 1)),
             torch.reshape(mesh_grid[1], (self.num_points, 1))),
            dim=1)
        mesh_grid = torch.transpose(mesh_grid, 0, 1).unsqueeze(0).repeat(
            sp_feat_conv.shape[0], 1, 1)
        mesh_grid = fourier_map(mesh_grid)
        """
        mesh_grid = torch.cat(
            (mesh_grid, torch.zeros(part.size(0), 1, mesh_grid.shape[2])),
            dim=1)
        """
        # y = SoftPool(sp_feat_conv[:, :, i, :])[0][:,:,i,:]
        y = sp_feat_conv[:, :, 0, :]
        out_seg = y
        # y = torch.cat((y, pn_feat), 1).contiguous()
        out_sp_local = self.decoder1(y)
        # pn_feat = torch.max(sp_feat[:,:,:,0], dim=1)[0].unsqueeze(2).expand(part.size(0),sp_feat_conv.size(1), mesh_grid.size(2)).contiguous()

        y = torch.cat(
            (rand_grid.repeat(1, 1, 16 * 8),
             torch.repeat_interleave(
                 torch.reshape(sp_cabins,
                               (sp_cabins.shape[0], sp_cabins.shape[1],
                                sp_cabins.shape[2] * sp_cabins.shape[3])),
                 repeats=self.num_points // 16 // 8,
                 dim=2), pn_feat), 1).contiguous()
        out_sp_global = self.decoder2(y)
        # y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
        y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
        out_pcn = self.decoder3(y)

        # part_regions = torch.cat(part_regions, 2).contiguous()
        part_regions = []
        for i in range(np.size(part_regions)):
            part_regions.append(sp_feat[:, -3:, i, :])
            part_regions[i] = part_regions[i].transpose(1, 2).contiguous()
            out_seg[i] = out_seg[i].transpose(1, 2).contiguous()
            sm = nn.Softmax(dim=2)
            out_seg[i] = sm(out_seg[i])

        out1 = out_sp_local.transpose(1, 2).contiguous()
        out3 = out_sp_global.transpose(1, 2).contiguous()

        out4 = out_pcn.transpose(1, 2).contiguous()
        # out_sp_local = torch.cat(out_sp_local, 2).contiguous()
        # out_sp_global = torch.cat(out_sp_global, 2).contiguous()
        # out_pcn = torch.cat(out_pcn, 2).contiguous()
        # out_seg = torch.cat(out_seg, 2).contiguous()

        dist, _, mean_mst_dis = self.expansion(
            out1, self.num_points // self.n_primitives // 8, 1.5)
        loss_mst = torch.mean(dist)

        id0 = torch.zeros(out_sp_local.shape[0], 1,
                          out_sp_local.shape[2]).cuda().contiguous()
        out_sp_local = torch.cat((out_sp_local, id0), 1)
        id1 = torch.ones(part.shape[0], 1, part.shape[2]).cuda().contiguous()
        part = torch.cat((part, id1), 1)
        """
        id2 = torch.zeros(out_sp_global.shape[0], 1,
                          out_sp_global.shape[2]).cuda().contiguous()
        out_sp_global = torch.cat((out_sp_global, id2), 1)
        id3 = torch.zeros(out_pcn.shape[0], 1,
                          out_pcn.shape[2]).cuda().contiguous()
        out_pcn = torch.cat((out_pcn, id3), 1)
        """
        fusion = torch.cat((out_sp_local, part), 2)
        # fusion = torch.cat((out_sp_global, out_pcn, part), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            fusion[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1],
            mean_mst_dis)
        fusion = MDS_module.gather_operation(fusion, resampled_idx)
        delta = self.res(fusion)
        fusion = fusion[:, 0:3, :]
        out2 = (fusion + delta).transpose(2, 1).contiguous()
        return out1, out2, out3, out4, loss_mst, out_seg, part_regions, loss_trans
