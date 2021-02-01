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
import softpool as sp
sys.path.append("../expansion_penalty/")
import expansion_penalty_module as expansion
sys.path.append("../MDS/")
import MDS_module
import grnet


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


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

    def filter(self, x):
        filters = torch.cat([
            torch.ones(1, dim_output // 128),
            torch.zeros(1, dim_output // 128 * 63)
        ], 1).cuda()
        filters = torch.unsqueeze(filters, 2)
        return filters

    def forward(self, x):
        # here are some options to check how to form the fourier feature
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
    def __init__(self, k=3 + 16):
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

        self.fourier_map1 = Periodics(dim_input=3, dim_output=32)
        self.fourier_map2 = Periodics(
            dim_input=32, dim_output=128, is_first=False)
        self.fourier_map3 = Periodics(
            dim_input=128, dim_output=128, is_first=False)

        self.stn = STNkd(k=regions + 3)

        self.num_points = num_points
        self.regions = regions
        self.sp_points = sp_points // sp_ratio

        self.softpool = sp.SoftPool(self.regions, cabins=8, sp_ratio=sp_ratio)

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

        trans = self.stn(point_wi_seg)
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
        # return feature, cabins, sp_idx, trans
        return sp_cube, cabins, sp_idx, trans


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


class Network(nn.Module):
    def __init__(self,
                 num_points=8192,
                 n_regions=16,
                 dim_pn=256,
                 sp_points=1024):
        super(Network, self).__init__()
        self.num_points = num_points
        self.dim_pn = dim_pn
        self.n_regions = n_regions
        self.sp_points = sp_points
        self.sp_ratio = n_regions

        self.pn_enc = nn.Sequential(
            PointNetFeat(num_points, 1024), nn.Linear(1024, dim_pn),
            nn.BatchNorm1d(dim_pn), nn.ReLU())

        self.softpool_enc = SoftPoolFeat(
            num_points,
            regions=self.n_regions,
            sp_points=2048,
            sp_ratio=self.sp_ratio)

        # Firstly we do not merge information among regions
        # We merge regional informations in latent space
        self.reg_conv1 = nn.Sequential(
            nn.Conv2d(
                1 * dim_pn,
                dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh())
        self.reg_conv2 = nn.Sequential(
            nn.Conv2d(
                dim_pn,
                2 * dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh())
        self.reg_conv3 = nn.Sequential(
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh())

        # input for embedding has 32 points now, then in total it is regions x 32 points
        # down-sampled by 2*2*2=8
        ebd_pnt_reg = self.num_points // (self.sp_ratio * 8)
        if self.n_regions == 1:
            ebd_pnt_out = 256
        elif self.n_regions > 1:
            ebd_pnt_out = 512

        self.embedding = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(1, ebd_pnt_reg), stride=(1, ebd_pnt_reg)),
            nn.MaxPool2d(
                kernel_size=(1, self.n_regions), stride=(1, self.n_regions)),
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, ebd_pnt_out),
                stride=(1, ebd_pnt_out),
                padding=(0, 0)))
        """
        self.embedding = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(1, ebd_pnt_reg), stride=(1, ebd_pnt_reg)),
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, self.n_regions)),
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=(0, 0)),
            nn.UpsamplingBilinear2d(scale_factor=(1, 4)),
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                padding_mode='same'),
            nn.UpsamplingBilinear2d(scale_factor=(1, 16)))
        """
        self.pt_mixing = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())

        self.reg_conv4 = nn.Sequential(
            nn.Conv2d(
                4 * dim_pn,
                4 * dim_pn,
                kernel_size=(self.n_regions, 1),
                stride=(1, 1)))

        self.reg_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh())
        self.reg_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * dim_pn,
                dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh())
        self.reg_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                dim_pn,
                dim_pn,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)), nn.Tanh())
        self.translate = nn.Sequential(
            nn.Conv2d(dim_pn, dim_pn, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh())

        self.decoder1 = PointGenCon(bottleneck_size=self.dim_pn)
        self.decoder2 = PointGenCon(bottleneck_size=2 + self.dim_pn)
        self.decoder3 = PointGenCon(bottleneck_size=2 * 256 + self.dim_pn)
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()
        self.grnet = grnet.GRNet()

    def forward(self, part, part_seg):

        part_seg = part_seg[:, :, 0]
        with_label = False
        if with_label:
            part_seg = torch.nn.functional.one_hot(
                part_seg.to(torch.int64), self.n_regions).transpose(1, 2)

            sp_feat, _, sp_idx, trans = self.softpool_enc(
                x=part, x_seg=part_seg)
        else:
            sp_feat, _, sp_idx, trans = self.softpool_enc(x=part, x_seg=None)
        loss_trans = feature_transform_regularizer(trans[-3:, -3:])
        pn_feat = self.pn_enc(part)
        pn_feat = pn_feat.unsqueeze(2).expand(
            part.size(0), self.dim_pn, self.num_points).contiguous()

        sp_feat_conv1 = self.reg_conv1(sp_feat)  # 256 points
        sp_feat_conv2 = self.reg_conv2(sp_feat_conv1)  # 256 points
        sp_feat_conv3 = self.reg_conv3(sp_feat_conv2)  # 256 points

        if self.n_regions == 1:
            sp_feat_unet = torch.cat(
                (self.embedding(sp_feat_conv3), sp_feat_conv3),
                dim=-1)  # 512 points
        elif self.n_regions > 1:
            sp_feat_unet = self.embedding(sp_feat_conv3)  # 512 points

        sp_feat_deconv3 = self.reg_deconv3(sp_feat_unet)  # 1024 points
        sp_feat_deconv2 = self.reg_deconv2(sp_feat_deconv3)  # 2048 points
        sp_feat_deconv1 = self.reg_deconv1(sp_feat_deconv2)  # 2048 points

        sp_feat_ae = self.reg_deconv1(
            self.reg_deconv2(self.reg_deconv3(sp_feat_conv3)))

        rand_grid = Variable(
            torch.FloatTensor(
                part.size(0), 2,
                self.num_points // self.n_regions // 8)).cuda()
        rand_grid.data.uniform_(0, 1)
        fourier_map2 = Periodics()
        rand_grid = fourier_map2(rand_grid).cuda()

        mesh_y = 8
        mesh_x = self.num_points // (8 * self.n_regions * mesh_y)
        mesh_grid_mini = torch.meshgrid([
            torch.linspace(0.0, 1.0, mesh_x),
            torch.linspace(0.0, 1.0, mesh_y)
        ])
        mesh_grid_mini = torch.cat(
            (torch.reshape(mesh_grid_mini[0], (mesh_x * mesh_y, 1)),
             torch.reshape(mesh_grid_mini[1], (mesh_x * mesh_y, 1))),
            dim=1)
        mesh_grid_mini = torch.transpose(mesh_grid_mini, 0,
                                         1).unsqueeze(0).repeat(
                                             sp_feat_deconv1.shape[0], 1, 1)
        # mesh_grid_mini = fourier_map(mesh_grid_mini).cuda()
        # here self.num_points // self.n_regions = 8*4

        mesh_grid = torch.meshgrid([
            torch.linspace(0.0, 1.0, 64),
            torch.linspace(0.0, 1.0, self.num_points // 64)
        ])
        mesh_grid = torch.cat(
            (torch.reshape(mesh_grid[0], (self.num_points, 1)),
             torch.reshape(mesh_grid[1], (self.num_points, 1))),
            dim=1)
        mesh_grid = torch.transpose(mesh_grid, 0, 1).unsqueeze(0).repeat(
            sp_feat_deconv1.shape[0], 1, 1).cuda()
        fourier_map3 = Periodics()
        mesh_grid = fourier_map3(mesh_grid)
        y = sp_feat_deconv1[:, :, 0, :]
        pcd_seg = y.transpose(1, 2).contiguous()
        sm = nn.Softmax(dim=2)
        pcd_seg = sm(pcd_seg)
        # y = torch.cat((y, pn_feat), 1).contiguous()
        pcd_softpool_trans = self.decoder1(y)
        pcd_softpool = pcd_softpool_trans.transpose(1, 2).contiguous()
        """
        [pcd_grnet_coar, pcd_grnet_fine] = self.grnet(part.transpose(1, 2))
        """

        y = sp_feat_ae[:, :, 0, :]
        pcd_sp_ae = self.decoder1(y)
        pcd_ae = pcd_sp_ae.transpose(1, 2).contiguous()
        """
        y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
        pcd_fold_trans = self.decoder3(y)
        pcd_fold = pcd_fold_trans.transpose(1, 2).contiguous()
        """

        input_chosen = sp_feat[:, -3:, 0, :].transpose(1, 2).contiguous()
        input_chosen = torch.gather(
            part, dim=2, index=sp_idx[:, :3, 0, :].long()).transpose(1, 2)

        dist, _, mean_mst_dis = self.expansion(
            pcd_softpool, self.num_points // self.n_regions, 1.5)
        loss_mst = torch.mean(dist)

        id1 = torch.ones(part.shape[0], 1, part.shape[2]).cuda().contiguous()
        id2 = torch.zeros(pcd_softpool_trans.shape[0], 1,
                          pcd_softpool_trans.shape[2]).cuda().contiguous()
        fuse_observe = torch.cat((part, id1), 1)
        # fuse_observe = torch.cat((pcd_softpool_trans[:, :, :self.num_points // 2:], id1), 1)
        fuse_expand = torch.cat((pcd_softpool_trans, id2), 1)
        fusion = torch.cat((fuse_observe, fuse_expand), 2)
        # fusion = torch.cat((fuse2, pcd_fold_trans, fuse1), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            fusion[:, 0:3, :].transpose(1, 2).contiguous(),
            pcd_softpool.shape[1], mean_mst_dis)
        fusion = MDS_module.gather_operation(fusion, resampled_idx)
        delta = self.res(fusion)
        fusion = fusion[:, 0:3, :]
        pcd_fusion = (fusion + delta).transpose(2, 1).contiguous()
        return [pcd_softpool, pcd_ae
                ], pcd_fusion, pcd_seg, input_chosen, loss_trans, loss_mst
