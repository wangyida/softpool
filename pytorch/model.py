from __future__ import print_function
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


def SoftPool(x):
    bth_size = list(x.shape)[0]
    featdim = list(x.shape)[1]
    points = list(x.shape)[2]
    sp_cube = torch.zeros(bth_size, featdim, featdim, points).cuda()
    sp_idx = torch.zeros(bth_size, 3, featdim, points).cuda()
    for idx in range(featdim):
        x_val, x_idx = torch.sort(x[:, idx, :], dim=1)
        index = x_idx[:, :].unsqueeze(1).repeat(1, featdim, 1)
        sp_cube[:, :, idx, :] = torch.gather(x, dim=2, index=index)
        sp_idx[:, :, idx, :] = x_idx[:, :].unsqueeze(1).repeat(1, 3, 1)
    return sp_cube, sp_idx


class STN3d(nn.Module):
    def __init__(self, num_points=2500, dim_pn=64):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_pn, 1)
        self.fc1 = nn.Linear(dim_pn, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, dim_pn)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


class PointNetFeat(nn.Module):
    def __init__(self, num_points=8192, dim_pn=1024):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
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
    def __init__(self, num_points=8192, dim_pn=64, N_p=16):
        super(SoftPoolFeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_pn, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(dim_pn)

        self.num_points = num_points
        self.N_p = N_p

    def forward(self, x):
        batchsize = x.size()[0]
        partial = x.unsqueeze(2).repeat(1, 1, 64, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, sp_idx = SoftPool(x)
        # 2048 / 63 = 32
        idx_step = torch.floor(
            torch.linspace(0, (x.shape[3] - 1), steps=self.N_p))
        # x = x[:, :, :, :self.N_p]
        x = x[:, :, :, idx_step.long()]
        # sp_idx = sp_idx[:, :, :, :self.N_p]
        sp_idx = sp_idx[:, :, :, idx_step.long()]
        partial = torch.gather(partial, dim=3, index=sp_idx.long())
        x = torch.cat((x, partial), 1).contiguous()
        return x, sp_idx


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


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

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
    def __init__(self, num_points=8192, n_primitives=64, dim_pn=64):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.dim_pn = dim_pn
        self.n_primitives = n_primitives
        self.pncoder = nn.Sequential(
            PointNetFeat(num_points), nn.Linear(1024, 256),
            nn.BatchNorm1d(256), nn.ReLU())
        self.N_p = 32
        self.spcoder = nn.Sequential(SoftPoolFeat(num_points, N_p=self.N_p))
        # Firstly we do not merge information among regions
        # We merge regional informations in latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(
                dim_pn + 3,
                dim_pn,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                dim_pn,
                dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                dim_pn,
                2 * dim_pn,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                2 * dim_pn,
                4 * dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                4 * dim_pn,
                8 * dim_pn,
                kernel_size=(1, 8),
                stride=(1, 1),
                padding=(0, 0),
                padding_mode='same'), nn.Tanh(),
            nn.ConvTranspose2d(
                8 * dim_pn,
                4 * dim_pn,
                kernel_size=(1, 8),
                stride=(1, 1),
                padding=(0, 0)), nn.Tanh(),
            nn.ConvTranspose2d(
                4 * dim_pn,
                2 * dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh(),
            nn.ConvTranspose2d(
                2 * dim_pn,
                dim_pn,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh(),
            nn.Conv2d(
                dim_pn,
                dim_pn,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                padding_mode='same'), nn.Tanh())
            # nn.Flatten(start_dim=2, end_dim=3))
        self.decoder1 = nn.ModuleList([
            PointGenCon(bottleneck_size=self.dim_pn + 256)
            # PointGenCon(dim_pn=2 + self.dim_pn)
            for i in range(0, self.n_primitives)
        ])
        self.decoder2 = nn.ModuleList([
            PointGenCon(bottleneck_size=3 + 256)
            # PointGenCon(bottleneck_size=2 + self.dim_pn)
            for i in range(0, self.n_primitives)
        ])
        self.decoder3 = nn.ModuleList([
            PointGenCon(bottleneck_size=3 + 256)
            # PointGenCon(bottleneck_size=2 + self.dim_pn)
            for i in range(0, self.n_primitives)
        ])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, partial):
        sp_feat, sp_idx = self.spcoder(partial)
        pn_feat = self.pncoder(partial)
        pn_feat = pn_feat.unsqueeze(2).expand(partial.size(0), 256,
                                              32).contiguous()
        partial_regions = []
        sp_feat_conv = self.encoder(sp_feat)
        out_sp_local = []
        out_seg = []
        out_sp_global = []
        out_pcn = []
        for i in range(0, self.n_primitives):
            partial_regions.append(
                torch.gather(partial, dim=2, index=sp_idx[:, :, i, :].long()))
            deform = 'patch_pcn'
            if deform == 'patch_msn':
                rand_grid = Variable(
                    torch.cuda.FloatTensor(
                        partial.size(0), 2,
                        self.num_points // self.n_primitives))
                rand_grid.data.uniform_(0, 1)
                # here self.num_points // self.n_primitives = 8*4
            elif deform == 'patch_pcn':
                mesh_grid = torch.meshgrid(
                    [torch.linspace(0.0, 1.0, 8),
                     torch.linspace(0.0, 1.0, 4)])
                mesh_grid = torch.cat(
                    (torch.reshape(mesh_grid[0],
                                   (self.num_points // self.n_primitives, 1)),
                     torch.reshape(mesh_grid[1],
                                   (self.num_points // self.n_primitives, 1))),
                    dim=1)
                mesh_grid = torch.transpose(mesh_grid, 0,
                                            1).unsqueeze(0).repeat(
                                                sp_feat_conv.shape[0], 1, 1)
                mesh_grid = torch.cat(
                    (mesh_grid, torch.zeros(partial.size(0), 1, 32)), dim=1)
            # y = sp_feat_conv.unsqueeze(2).expand(partial.size(0),sp_feat_conv.size(1), mesh_grid.size(2)).contiguous()
            # y = sp_feat_conv[:, :, i].unsqueeze(2).expand(partial.size(0), sp_feat_conv.size(1), rand_grid.size(2)).contiguous()
            y = sp_feat_conv[:, :, i, :]
            # y = sp_feat_conv
            out_seg.append(y)
            y = torch.cat((y, pn_feat), 1).contiguous()
            out_sp_local.append(self.decoder1[i](y))
            # pn_feat = torch.max(sp_feat[:,:,:,0], dim=1)[0].unsqueeze(2).expand(partial.size(0),sp_feat_conv.size(1), mesh_grid.size(2)).contiguous()
            y = torch.cat((self.decoder1[i](y), pn_feat), 1).contiguous()
            # y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
            out_sp_global.append(self.decoder2[i](y))
            y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
            out_pcn.append(self.decoder3[i](y))

        partial_regions = torch.cat(partial_regions, 2).contiguous()
        partial_regions = partial_regions.transpose(1, 2).contiguous()
        out_sp_local = torch.cat(out_sp_local, 2).contiguous()
        out1 = out_sp_local.transpose(1, 2).contiguous()
        out_sp_global = torch.cat(out_sp_global, 2).contiguous()
        out3 = out_sp_global.transpose(1, 2).contiguous()
        out_pcn = torch.cat(out_pcn, 2).contiguous()
        out4 = out_pcn.transpose(1, 2).contiguous()
        out_seg = torch.cat(out_seg, 2).contiguous()
        out_seg = out_seg.transpose(1, 2).contiguous()
        sm = nn.Softmax(dim=2)
        out_seg = sm(out_seg)

        dist, _, mean_mst_dis = self.expansion(
            out1, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)
        dist, _, mean_mst_dis = self.expansion(
            out3, self.num_points // self.n_primitives, 1.5)
        dist, _, mean_mst_dis = self.expansion(
            out4, self.num_points // self.n_primitives, 1.5)
        loss_mst += torch.mean(dist)

        id0 = torch.zeros(out_sp_local.shape[0], 1,
                          out_sp_local.shape[2]).cuda().contiguous()
        out_sp_local = torch.cat((out_sp_local, id0), 1)
        id1 = torch.ones(partial.shape[0], 1,
                         partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)
        id2 = torch.zeros(out_sp_global.shape[0], 1,
                          out_sp_global.shape[2]).cuda().contiguous()
        out_sp_global = torch.cat((out_sp_global, id2), 1)
        id3 = torch.zeros(out_pcn.shape[0], 1,
                          out_pcn.shape[2]).cuda().contiguous()
        out_pcn = torch.cat((out_pcn, id3), 1)
        fusion = torch.cat((out_sp_global, out_pcn, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            fusion[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1],
            mean_mst_dis)
        fusion = MDS_module.gather_operation(fusion, resampled_idx)
        delta = self.res(fusion)
        fusion = fusion[:, 0:3, :]
        out2 = (fusion + delta).transpose(2, 1).contiguous()
        return out1, out2, out3, out4, loss_mst, out_seg, partial_regions
