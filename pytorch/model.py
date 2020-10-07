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
        x_val, x_idx = torch.sort(x[:, idx, :], dim=1, descending=True)
        index = x_idx[:, :].unsqueeze(1).repeat(1, featdim, 1)
        x_order = torch.gather(x, dim=2, index=index)
        # here is differential soft-pool feature
        sp_cube[:, :, idx, :] = x_order
        sp_idx[:, :, idx, :] = x_idx[:, :].unsqueeze(1).repeat(1, 3, 1)
    return sp_cube, sp_idx


class STN3d(nn.Module):
    def __init__(self, num_points=8192, dim_pn=1024):
        super(STN3d, self).__init__()
        self.num_points = num_points
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

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

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
    def __init__(self, num_points=8192, regions=64, sp_points=2048):
        super(SoftPoolFeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, regions, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(regions)

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
        x, sp_idx = SoftPool(x)
        # 2048 / 63 = 32
        idx_step = torch.floor(
            torch.linspace(0, (x.shape[3] - 1), steps=self.sp_points))
        x = x[:, :, :, :self.sp_points]
        sp_idx = sp_idx[:, :, :, :self.sp_points]
        # x = x[:, :, :, idx_step.long()]
        # sp_idx = sp_idx[:, :, :, idx_step.long()]
        part = torch.gather(part, dim=3, index=sp_idx.long())
        # out = torch.cat((x, part), 1).contiguous()
        out = x
        return out, sp_idx, trans


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
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(
            self.bottleneck_size,
            self.bottleneck_size // 2,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(
            self.bottleneck_size // 2,
            self.bottleneck_size // 4,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1))
        self.conv4 = torch.nn.Conv2d(
            self.bottleneck_size // 4,
            3,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1))

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
    def __init__(self,
                 num_points=8192,
                 n_primitives=8,
                 dim_pn=256,
                 sp_points=2048):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.dim_pn = dim_pn
        self.n_primitives = n_primitives
        self.sp_points = sp_points
        self.pncoder = nn.Sequential(
            PointNetFeat(num_points, dim_pn=2048), 
            nn.Linear(2048, dim_pn),
            nn.BatchNorm1d(dim_pn), 
            nn.ReLU())
        self.spcoder = SoftPoolFeat(
            num_points, regions=self.n_primitives, sp_points=self.sp_points)
        # Firstly we do not merge information among regions
        # We merge regional informations in latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.sp_points, self.num_points),
            # nn.ReLU(),
            nn.Linear(self.sp_points, self.sp_points // 2),
            # nn.ReLU(),
            nn.Linear(self.sp_points // 2, self.num_points))
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
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=(0, 0)),
            nn.Conv2d(
                n_primitives,
                n_primitives,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                padding_mode='same'))
            nn.Conv2d(
                2 * n_primitives,
                4 * n_primitives,
                kernel_size=(1, 5),
                stride=(1, 2),
                padding=(0, 2),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                4 * n_primitives,
                4 * n_primitives,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                padding_mode='same'), nn.Tanh(),
            nn.Conv2d(
                4 * n_primitives,
                4 * n_primitives,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='same'), nn.Tanh(),
            nn.ConvTranspose2d(
                4 * n_primitives,
                2 * n_primitives,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh(),
            nn.ConvTranspose2d(
                2 * n_primitives,
                n_primitives,
                kernel_size=(1, 2),
                stride=(1, 2),
                padding=(0, 0)), nn.Tanh(),
        """
        # nn.Flatten(start_dim=2, end_dim=3))
        self.decoder1 = nn.ModuleList([
            PointGenCon(bottleneck_size=self.n_primitives + self.dim_pn)
            # PointGenCon(n_primitives=2 + self.n_primitives)
            for i in range(0, self.n_primitives)
        ])
        self.decoder2 = nn.ModuleList([
            PointGenCon(bottleneck_size=3 + self.dim_pn)
            # PointGenCon(bottleneck_size=2 + self.n_primitives)
            for i in range(0, self.n_primitives)
        ])
        self.decoder3 = nn.ModuleList([
            PointGenCon(bottleneck_size=3 + self.dim_pn)
            # PointGenCon(bottleneck_size=2 + self.n_primitives)
            for i in range(0, self.n_primitives)
        ])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, part):
        sp_feat, sp_idx, trans = self.spcoder(part)
        loss_trans = feature_transform_regularizer(trans)
        pn_feat = self.pncoder(part)
        pn_feat = pn_feat.unsqueeze(2).expand(
            part.size(0), self.dim_pn, self.num_points).contiguous()
        part_regions = []
        sp_feat_conv = self.encoder(sp_feat)
        out_sp_local = []
        out_seg = []
        out_sp_global = []
        out_pcn = []
        for i in range(0, self.n_primitives):
            part_regions.append(
                torch.gather(part, dim=2, index=sp_idx[:, :, i, :].long()))
            """
            # stn3d
            part_regions.append(
                    sp_feat[:,-3:,i,:])
            """
            deform = 'patch_pcn'
            if deform == 'patch_msn':
                rand_grid = Variable(
                    torch.cuda.FloatTensor(
                        part.size(0), 2, self.num_points // self.n_primitives))
                rand_grid.data.uniform_(0, 1)
                # here self.num_points // self.n_primitives = 8*4
            elif deform == 'patch_pcn':
                mesh_grid = torch.meshgrid([
                    torch.linspace(0.0, 1.0, 64),
                    torch.linspace(0.0, 1.0, self.num_points // 64)
                ])
                mesh_grid = torch.cat(
                    (torch.reshape(mesh_grid[0],
                                   (self.num_points // self.n_primitives *
                                    self.n_primitives, 1)),
                     torch.reshape(mesh_grid[1],
                                   (self.num_points // self.n_primitives *
                                    self.n_primitives, 1))),
                    dim=1)
                mesh_grid = torch.transpose(mesh_grid, 0,
                                            1).unsqueeze(0).repeat(
                                                sp_feat_conv.shape[0], 1, 1)
                mesh_grid = torch.cat(
                    (mesh_grid, torch.zeros(
                        part.size(0), 1, mesh_grid.shape[2])),
                    dim=1)
            y = sp_feat_conv[:, :, i, :]
            # y = SoftPool(sp_feat_conv[:, :, i, :])[0][:,:,i,:]
            # y = sp_feat_conv
            out_seg.append(y)
            y = torch.cat((y, pn_feat), 1).contiguous()
            out_sp_local.append(self.decoder1[i](y))
            # pn_feat = torch.max(sp_feat[:,:,:,0], dim=1)[0].unsqueeze(2).expand(part.size(0),sp_feat_conv.size(1), mesh_grid.size(2)).contiguous()
            y = torch.cat((self.decoder1[i](y), pn_feat), 1).contiguous()
            out_sp_global.append(self.decoder2[i](y))
            # y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
            y = torch.cat((mesh_grid.cuda(), pn_feat), 1).contiguous()
            out_pcn.append(self.decoder3[i](y))

        # part_regions = torch.cat(part_regions, 2).contiguous()
        out1 = []
        out3 = []
        out4 = []
        for i in range(np.size(part_regions)):
            part_regions[i] = part_regions[i].transpose(1, 2).contiguous()
            out1.append(out_sp_local[i].transpose(1, 2).contiguous())
            out3.append(out_sp_global[i].transpose(1, 2).contiguous())
            out4.append(out_pcn[i].transpose(1, 2).contiguous())

            out_seg[i] = out_seg[i].transpose(1, 2).contiguous()
            sm = nn.Softmax(dim=2)
            out_seg[i] = sm(out_seg[i])
        # out_sp_local = torch.cat(out_sp_local, 2).contiguous()
        # out_sp_global = torch.cat(out_sp_global, 2).contiguous()
        # out_pcn = torch.cat(out_pcn, 2).contiguous()
        # out_seg = torch.cat(out_seg, 2).contiguous()

        dist, _, mean_mst_dis = self.expansion(
            out1[0], self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)
        dist, _, mean_mst_dis = self.expansion(
            out3[0], self.num_points // self.n_primitives, 1.5)
        dist, _, mean_mst_dis = self.expansion(
            out4[0], self.num_points // self.n_primitives, 1.5)
        loss_mst += torch.mean(dist)

        id0 = torch.zeros(out_sp_local[0].shape[0], 1,
                          out_sp_local[0].shape[2]).cuda().contiguous()
        out_sp_local[0] = torch.cat((out_sp_local[0], id0), 1)
        id1 = torch.ones(part.shape[0], 1, part.shape[2]).cuda().contiguous()
        part = torch.cat((part, id1), 1)
        id2 = torch.zeros(out_sp_global[0].shape[0], 1,
                          out_sp_global[0].shape[2]).cuda().contiguous()
        out_sp_global[0] = torch.cat((out_sp_global[0], id2), 1)
        id3 = torch.zeros(out_pcn[0].shape[0], 1,
                          out_pcn[0].shape[2]).cuda().contiguous()
        out_pcn[0] = torch.cat((out_pcn[0], id3), 1)
        fusion = torch.cat((out_sp_local[0], part), 2)
        # fusion = torch.cat((out_sp_global, out_pcn, part), 2)

        resampled_idx = MDS_module.minimum_density_sample(
            fusion[:, 0:3, :].transpose(1, 2).contiguous(), out1[0].shape[1],
            mean_mst_dis)
        fusion = MDS_module.gather_operation(fusion, resampled_idx)
        delta = self.res(fusion)
        fusion = fusion[:, 0:3, :]
        out2 = (fusion + delta).transpose(2, 1).contiguous()
        return out1, out2, out3, out4, loss_mst, out_seg, part_regions, loss_trans
