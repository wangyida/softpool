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
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion
sys.path.append("./MDS/")
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


def fourier_map(x, dim_input=2):
    # here are some options to check how to form the fourier feature
    upgrade_weights = False
    omega0 = 25
    B = nn.Conv1d(dim_input, 256, 1).cuda()
    # nn.init.normal_(B.weight, std=10.0)
    with torch.no_grad():
        B.weight.uniform_(-1 / dim_input, 1 / dim_input)
    B.weight.requires_grad = upgrade_weights
    sinside = torch.sin(2 * pi * B(x) * omega0)
    cosside = torch.cos(2 * pi * B(x) * omega0)
    return torch.cat([sinside, cosside], 1)


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

        self.num_points = num_points

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.dim_pn)
        return x


class SoftPoolFeat(nn.Module):
    def __init__(self, num_points=8192, regions=16, sp_points=256):
        super(SoftPoolFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(512, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.stn = STNkd(k=regions + 3)

        self.num_points = num_points
        self.regions = regions
        self.sp_points = sp_points

        self.softpool = sp.SoftPool(self.regions, cabins=8)

    def mlp(self, inputs):
        x = fourier_map(inputs, dim_input=3)
        x = F.relu(self.bn1(self.conv1(x)))
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

        # 2048 / 63 = 32
        idx_step = torch.floor(
            torch.linspace(0, (sp_cube.shape[3] - 1), steps=self.sp_points))
        sp_cube = sp_cube[:, :, :, :self.sp_points]
        # sp_idx = sp_idx[:, :, :, :self.sp_points]
        # x = x[:, :, :, idx_step.long()]
        # sp_idx = sp_idx[:, :, :, idx_step.long()]
        point_wi_seg = torch.gather(point_wi_seg, dim=3, index=sp_idx.long())
        feature = torch.cat((sp_cube, point_wi_seg), 1).contiguous()
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
        self.pn_enc = nn.Sequential(
            PointNetFeat(num_points, 1024), nn.Linear(1024, dim_pn),
            nn.BatchNorm1d(dim_pn), nn.ReLU())
        self.softpool_enc = SoftPoolFeat(
            num_points, regions=self.n_regions, sp_points=2048)
        # Firstly we do not merge information among regions
        # We merge regional informations in latent space
        self.ptmapper1 = nn.Sequential(
            nn.Conv2d(
                4 * dim_pn + n_regions + 3,
                dim_pn,
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh())
        self.ptmapper2 = nn.Sequential(
            nn.Conv2d(
                dim_pn,
                2 * dim_pn,
                kernel_size=(1, 7),
                stride=(1, 2),
                padding=(0, 3),
                padding_mode='same'), nn.Tanh())
        self.ptmapper3 = nn.Sequential(
            nn.Conv2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 5),
                stride=(1, 2),
                padding=(0, 2),
                padding_mode='same'), nn.Tanh())
        self.embedding = nn.Sequential(
            nn.MaxPool2d((1, 256)),
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=(0, 0)),
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=(0, 0)), nn.Tanh(),
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=(0, 0)), nn.Tanh())

        self.ptmapper4 = nn.Sequential(
            nn.Conv2d(
                4 * dim_pn,
                4 * dim_pn,
                kernel_size=(self.n_regions, 1),
                stride=(1, 1)))

        self.ptmapper3_rev = nn.Sequential(
            nn.ConvTranspose2d(
                2 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh())
        self.ptmapper2_rev = nn.Sequential(
            nn.ConvTranspose2d(
                2 * dim_pn,
                dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh())
        self.ptmapper1_rev = nn.Sequential(
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

            sp_feat, sp_cabins, sp_idx, trans = self.softpool_enc(
                x=part, x_seg=part_seg)
        else:
            sp_feat, sp_cabins, sp_idx, trans = self.softpool_enc(
                x=part, x_seg=None)
        loss_trans = feature_transform_regularizer(trans[-3:, -3:])
        pn_feat = self.pn_enc(part)
        pn_feat = pn_feat.unsqueeze(2).expand(
            part.size(0), self.dim_pn, self.num_points).contiguous()

        sp_feat_conv1 = self.ptmapper1(sp_feat)
        sp_feat_conv2 = self.ptmapper2(sp_feat_conv1)
        sp_feat_conv3 = self.embedding(self.ptmapper3(sp_feat_conv2))

        sp_feat_deconv3 = self.ptmapper3_rev(sp_feat_conv3)  # + sp_feat_conv2
        sp_feat_deconv2 = torch.cat((self.ptmapper2_rev(sp_feat_deconv3),
                                     self.translate(sp_feat_conv1)),
                                    dim=-1)
        sp_feat_deconv1 = self.ptmapper1_rev(sp_feat_deconv2)

        sp_feat_ae = self.ptmapper1_rev(self.translate(sp_feat_conv1))

        rand_grid = Variable(
            torch.FloatTensor(
                part.size(0), 2,
                self.num_points // self.n_regions // 8)).cuda()
        rand_grid.data.uniform_(0, 1)
        rand_grid = fourier_map(rand_grid).cuda()

        mesh_grid_mini = torch.meshgrid(
            [torch.linspace(0.0, 1.0, 8),
             torch.linspace(0.0, 1.0, 16)])
        mesh_grid_mini = torch.cat(
            (torch.reshape(mesh_grid_mini[0], (8 * 16, 1)),
             torch.reshape(mesh_grid_mini[1], (8 * 16, 1))),
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
        mesh_grid = fourier_map(mesh_grid)
        y = sp_feat_deconv1[:, :, 0, :]
        out_seg = y.transpose(1, 2).contiguous()
        sm = nn.Softmax(dim=2)
        out_seg = sm(out_seg)
        # y = torch.cat((y, pn_feat), 1).contiguous()
        out_softpool_trans = self.decoder1(y)
        out_softpool = out_softpool_trans.transpose(1, 2).contiguous()

        [out_grnet_coar, out_grnet_fine] = self.grnet(part.transpose(1, 2))

        y = sp_feat_ae[:, :, 0, :]
        out_sp_ae = self.decoder1(y)
        out_ae = out_sp_ae.transpose(1, 2).contiguous()

        # here 8 is the number of cabins
        y = torch.cat(
            (mesh_grid_mini.repeat(1, 1, 8 * self.n_regions).cuda(),
             torch.repeat_interleave(
                 torch.reshape(sp_cabins,
                               (sp_cabins.shape[0], sp_cabins.shape[1],
                                sp_cabins.shape[2] * sp_cabins.shape[3])),
                 repeats=self.num_points // 8 // self.n_regions,
                 dim=2)), 1).contiguous()
        out_sp_global = self.decoder2(y)
        out3 = out_sp_global.transpose(1, 2).contiguous()

        y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
        out_fold_trans = self.decoder3(y)
        out_fold = out_fold_trans.transpose(1, 2).contiguous()

        part_regions = sp_feat[:, -3:, 0, :].transpose(1, 2).contiguous()

        dist, _, mean_mst_dis = self.expansion(
            out_softpool, self.num_points // self.n_regions // 8, 1.5)
        loss_mst = torch.mean(dist)

        id1 = torch.ones(part.shape[0], 1, part.shape[2]).cuda().contiguous()
        id2 = torch.zeros(out_softpool_trans.shape[0], 1,
                          out_softpool_trans.shape[2]).cuda().contiguous()
        fuse1 = torch.cat((part, id1), 1)
        fuse2 = torch.cat((out_softpool_trans[:, :, :self.num_points], id2), 1)
        """
        id3 = torch.ones(out_fold_trans.shape[0], 1,
                          out_fold_trans.shape[2]).cuda().contiguous()
        out_fold_trans = torch.cat((out_fold_trans, id3), 1)
        """
        fusion = torch.cat((fuse2, fuse1), 2)
        # fusion = torch.cat((fuse2, out_fold_trans, fuse1), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            fusion[:, 0:3, :].transpose(1, 2).contiguous(),
            out_softpool.shape[1], mean_mst_dis)
        fusion = MDS_module.gather_operation(fusion, resampled_idx)
        delta = self.res(fusion)
        fusion = fusion[:, 0:3, :]
        out_fusion = (fusion + delta).transpose(2, 1).contiguous()
        return [out_softpool, out_ae], out_fusion, out_fold, [
            out_grnet_coar, out_grnet_fine
        ], out_seg, part_regions, loss_trans, loss_mst
