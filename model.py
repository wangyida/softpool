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
import pointnet as pn
import softpool as sp


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


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


class Network(nn.Module):
    def __init__(self,
                 num_points=8192,
                 n_regions=16,
                 dim_pn=256,
                 sp_points=1024,
                 model_lists=['softpool', 'msn', 'folding', 'grnet']):
        super(Network, self).__init__()
        self.num_points = num_points
        self.dim_pn = dim_pn
        self.n_regions = n_regions
        self.sp_points = sp_points
        self.sp_ratio = n_regions
        self.model_lists = model_lists

        if ('softpool' in self.model_lists):
            from MSN import msn
            self.softpool_enc = sp.SoftPoolFeat(
                num_points,
                regions=self.n_regions,
                sp_points=2048,
                sp_ratio=self.sp_ratio)

            import MSN.expansion_penalty.expansion_penalty_module as expansion
            self.expansion = expansion.expansionPenaltyModule()
            # Firstly we do not merge information among regions
            # We merge regional informations in latent space
            self.reg_encode = nn.Sequential(
                nn.Conv2d(
                    1 * dim_pn,
                    dim_pn,
                    kernel_size=(1, 3),
                    stride=(1, 2),
                    padding=(0, 1),
                    padding_mode='replicate'), nn.LeakyReLU(0.2),
                nn.Conv2d(
                    dim_pn,
                    2 * dim_pn,
                    kernel_size=(1, 3),
                    stride=(1, 2),
                    padding=(0, 1),
                    padding_mode='replicate'), nn.LeakyReLU(0.2),
                nn.Conv2d(
                    2 * dim_pn,
                    2 * dim_pn,
                    kernel_size=(1, 3),
                    stride=(1, 2),
                    padding=(0, 1),
                    padding_mode='replicate'), nn.LeakyReLU(0.2))

            # input for embedding has 32 points now, then in total it is regions x 32 points
            # down-sampled by 2*2*2=8
            ebd_pnt_reg = (self.num_points) // (self.sp_ratio * 8)
            if self.n_regions == 1:
                ebd_pnt_out = 256
            elif self.n_regions > 1:
                ebd_pnt_out = 512

            self.embedding = nn.Sequential(
                nn.MaxPool2d(
                    kernel_size=(1, ebd_pnt_reg), stride=(1, ebd_pnt_reg)),
                nn.MaxPool2d(
                    kernel_size=(1, self.n_regions),
                    stride=(1, self.n_regions)),
                nn.ConvTranspose2d(
                    2 * dim_pn,
                    2 * dim_pn,
                    kernel_size=(1, ebd_pnt_out),
                    stride=(1, ebd_pnt_out),
                    padding=(0, 0)), nn.LeakyReLU(0.2))

            self.reg_deconv3 = nn.Sequential(
                nn.ConvTranspose2d(
                    2 * dim_pn,
                    2 * dim_pn,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0)), nn.LeakyReLU(0.2))
            self.reg_deconv2 = nn.Sequential(
                nn.ConvTranspose2d(
                    2 * dim_pn,
                    dim_pn,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0)), nn.LeakyReLU(0.2))
            self.reg_deconv1 = nn.Sequential(
                nn.ConvTranspose2d(
                    dim_pn,
                    dim_pn,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0)), nn.LeakyReLU(0.2))

            self.sp_dec_mlp = msn.PointGenCon(bottleneck_size=self.dim_pn)
            self.sp_dec_residual = msn.PointNetRes()
        if ('folding' in self.model_lists):
            from MSN import msn
            self.pn_enc = nn.Sequential(
                pn.PointNetFeat(num_points, 1024), nn.Linear(1024, dim_pn),
                nn.BatchNorm1d(dim_pn), nn.ReLU())
            self.decoder_fold = msn.PointGenCon(
                bottleneck_size=2 + self.dim_pn)
        if ('msn' in self.model_lists):
            import MSN.expansion_penalty.expansion_penalty_module as expansion
            import MSN.MDS.MDS_module as MDS_module
            from MSN import msn
            self.pn_enc = nn.Sequential(
                pn.PointNetFeat(num_points, 1024), nn.Linear(1024, dim_pn),
                nn.BatchNorm1d(dim_pn), nn.ReLU())
            self.expansion = expansion.expansionPenaltyModule()
            self.msn = msn.MSN()
        if ('grnet' in self.model_lists):
            from GRNet import grnet
            self.grnet = grnet.GRNet()
        if ('pointcnn' in self.model_lists):
            from pointcnn.PointCNN import PointCnnLayer
            x = 8
            xconv_param_name = ('K', 'D', 'P', 'C')
            xconv_params = [
                dict(zip(xconv_param_name, xconv_param))
                for xconv_param in [(8, 1, -1,
                                     256), (12, 2, 768,
                                            256), (16, 2, 384,
                                                   512), (16, 4, 128, 1024)]
            ]

            xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
            xdconv_params = [
                dict(zip(xdconv_param_name, xdconv_param))
                for xdconv_param in [(16, 4, 3, 3), (16, 2, 3,
                                                     2), (12, 2, 2,
                                                          1), (8, 2, 1, 0)]
            ]

            fc_param_name = ('C', 'dropout_rate')
            fc_params = [
                dict(zip(fc_param_name, fc_param))
                for fc_param in [(32 * x, 0.0), (32 * x, 0.5), (3, 0.5)]
            ]

            self.pointcnn = PointCnnLayer(
                ["features"], [xconv_params, xdconv_params, fc_params]).cuda()
        if ('pcn' in self.model_lists):
            from pcn import PCN
            self.pcn = PCN().cuda()
        if ('pointgcn' in self.model_lists):
            import GCN.gcn_encode as gcn3d
            from GCN.gcn_decode import Generator
            self.gcn_enc = gcn3d.GCN3DFeatDeep(support_num=1, neighbor_num=20)
            self.gcn_dec = Generator(
                features=[dim_pn, 256, 256, 256, 128, 128, 128, 3],
                degrees=[1, 2, 2, 2, 2, 2, 16],
                support=10,
                root_num=4)

    def forward(self, part, part_seg):

        if ('msn' in self.model_lists):
            # transpose part when using GCN
            pn_feat = self.pn_enc(part)
            [pcd_msn1, pcd_msn2, loss_mst, mean_mst_dis] = self.msn(
                part, pn_feat)

        if ('softpool' in self.model_lists):
            part_seg = part_seg[:, :, 0]
            with_label = False
            if with_label:
                part_seg = torch.nn.functional.one_hot(
                    part_seg.to(torch.int64), self.n_regions).transpose(1, 2)

                sp_feat, _, sp_idx = self.softpool_enc(x=part, x_seg=part_seg)
            else:
                sp_feat, _, sp_idx = self.softpool_enc(x=part, x_seg=None)

            input_chosen = sp_feat[:, -3:, 0, :].transpose(1, 2).contiguous()
            input_chosen = torch.gather(
                part, dim=2, index=sp_idx[:, :3, 0, :].long()).transpose(1, 2)

            sp_feat_encode = self.reg_encode(sp_feat)  # 256 points

            if self.n_regions == 1:
                sp_feat_unet = torch.cat(
                    (self.embedding(sp_feat_encode), sp_feat_encode),
                    dim=-1)  # 512 points
            elif self.n_regions > 1:
                sp_feat_unet = self.embedding(sp_feat_encode)  # 512 points

            sp_feat_deconv = self.reg_deconv3(sp_feat_unet)  # 1024 points
            sp_feat_low = self.reg_deconv2(sp_feat_deconv)  # 2048 points
            sp_feat_high = self.reg_deconv1(sp_feat_low)  # 4096 points

            pcd_sp_low_t = self.sp_dec_mlp(sp_feat_low[:, :, 0, :])
            pcd_sp_low = pcd_sp_low_t.transpose(1, 2).contiguous()
            pcd_sp_high_t = self.sp_dec_mlp(sp_feat_high[:, :, 0, :])
            pcd_sp_high = pcd_sp_high_t.transpose(1, 2).contiguous()

            id1 = torch.ones(part.shape[0], 1,
                             part.shape[2]).cuda().contiguous()
            id2 = torch.zeros(pcd_sp_low_t.shape[0], 1,
                              pcd_sp_low_t.shape[2]).cuda().contiguous()
            id3 = torch.zeros(pcd_sp_high_t.shape[0], 1,
                              pcd_sp_high_t.shape[2]).cuda().contiguous()
            labeled_observe = torch.cat((part, id1), 1)
            labeled_low = torch.cat((pcd_sp_low_t, id2), 1)
            labeled_high = torch.cat((pcd_sp_high_t, id3), 1)
            fusion_low = torch.cat((labeled_observe, labeled_low), 2)
            fusion_high = torch.cat((labeled_observe, labeled_high), 2)

            dist, _, mean_mst_dis_l = self.expansion(
                pcd_sp_low, 1024 // np.max((4, self.n_regions)), 1.5)
            loss_mst = torch.mean(dist)
            import MSN.MDS.MDS_module as MDS_module
            resampled_idx_low = MDS_module.minimum_density_sample(
                fusion_low[:, 0:3, :].transpose(1, 2).contiguous(),
                pcd_sp_low.shape[1], mean_mst_dis_l)
            fusion_low = MDS_module.gather_operation(fusion_low,
                                                     resampled_idx_low)
            pcd_fusion_low = (fusion_low[:, 0:3, :] +
                              self.sp_dec_residual(fusion_low)).transpose(
                                  2, 1).contiguous()

            dist, _, mean_mst_dis_h = self.expansion(
                pcd_sp_high, 2048 // np.max((4, self.n_regions)), 1.5)
            loss_mst += torch.mean(dist)
            resampled_idx_high = MDS_module.minimum_density_sample(
                fusion_high[:, 0:3, :].transpose(1, 2).contiguous(),
                pcd_sp_high.shape[1], mean_mst_dis_h)
            fusion_high = MDS_module.gather_operation(fusion_high,
                                                      resampled_idx_high)
            pcd_fusion_high = (fusion_high[:, 0:3, :] +
                               self.sp_dec_residual(fusion_high)).transpose(
                                   2, 1).contiguous()

        if ('folding' in self.model_lists):
            # transpose part when using GCN
            pn_feat = self.pn_enc(part)
            mesh_grid = torch.meshgrid([
                torch.linspace(0.0, 1.0, 64),
                torch.linspace(0.0, 1.0, self.num_points // 64)
            ])
            mesh_grid = torch.cat(
                (torch.reshape(mesh_grid[0], (self.num_points, 1)),
                 torch.reshape(mesh_grid[1], (self.num_points, 1))),
                dim=1)
            mesh_grid = torch.transpose(mesh_grid, 0, 1).unsqueeze(0).repeat(
                part.shape[0], 1, 1).cuda()
            # fourier_map3 = Periodics()
            # mesh_grid = fourier_map3(mesh_grid)
            pn_feat = pn_feat.unsqueeze(2).expand(
                part.size(0), self.dim_pn, self.num_points).contiguous()
            y = torch.cat((mesh_grid, pn_feat), 1).contiguous()
            pcd_fold_t = self.decoder_fold(y)
            pcd_fold = pcd_fold_t.transpose(1, 2).contiguous()

        if ('grnet' in self.model_lists):
            [
                pcd_grnet_voxel, pcd_grnet_fine, pcd_grnet_coar,
                grnet_seg_fine, grnet_seg_coar, voxels
            ] = self.grnet(part.transpose(1, 2))

        if ('pointcnn' in self.model_lists):
            pcd_pcnn = self.pointcnn(part.transpose(1, 2))

        if ('pcn' in self.model_lists):
            pcn_coarse, pcn_fine = self.pcn(part)
        if ('pointgcn' in self.model_lists):
            gcn_feat = self.gcn_enc(part.transpose(1, 2))
            """
            mask = torch.ones(1, 32, 1).cuda()
            mask[:,:31,:] *= 0.0
            gcn_feat *= mask
            """
            # pcd_gcn = self.gcn_dec([gcn_feat.unsqueeze(1)])
            pcd_gcn = self.gcn_dec([gcn_feat])

        # start to organize
        pred_softpool = [
            pcd_sp_low, pcd_sp_high, pcd_fusion_low, pcd_fusion_high,
            input_chosen, loss_mst
        ] if ('softpool' in self.model_lists) else []
        pred_msn = [pcd_msn1, pcd_msn2, loss_mst
                    ] if ('msn' in self.model_lists) else []
        pred_folding = [pcd_fold] if ('folding' in self.model_lists) else []
        pred_grnet = [
            pcd_grnet_voxel, pcd_grnet_fine, pcd_grnet_coar, grnet_seg_fine,
            grnet_seg_coar, voxels
        ] if ('grnet' in self.model_lists) else []
        pred_pcnn = [pcd_pcnn] if ('pointcnn' in self.model_lists) else []
        pred_pcn = [pcn_coarse, pcn_fine
                    ] if ('pcn' in self.model_lists) else []
        pred_pgcn = [pcd_gcn] if ('pointgcn' in self.model_lists) else []
        return pred_softpool, pred_msn, pred_folding, pred_grnet, pred_pcnn, pred_pcn, pred_pgcn
