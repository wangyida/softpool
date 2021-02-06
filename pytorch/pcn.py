import open3d
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
# from emd import earth_mover_distance
# from chamfer_distance import ChamferDistance

# chamfer_dist = ChamferDistance()

class PCNEMD(nn.Module):
    def __init__(self):
        super(PCNEMD, self).__init__()
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.npts = [1]
        #alpha  = [10000, 20000, 50000],[0.01, 0.1, 0.5, 1.0]
        #### ENCODER

        ## first mlp
        mlps1 = [128, 256]
        first_mlp_list = []
        in_features = 3
        for m in range(0,len(mlps1)-1):
            first_mlp_list.append(nn.Conv1d(in_features, mlps1[m], 1))
            first_mlp_list.append(nn.ReLU())
            in_features = mlps1[m]
        first_mlp_list.append(nn.Conv1d(in_features, mlps1[-1], 1))
        self.first_mpl = nn.Sequential(*first_mlp_list)


        ## Second  mlp
        mlps2 = [512, 1024]
        second_mlp_list = []
        in_features = 512
        for m in range(0,len(mlps2)-1):
            second_mlp_list.append(nn.Conv1d(in_features, mlps2[m], 1))
            second_mlp_list.append(nn.ReLU())
            in_features = mlps2[m]
        second_mlp_list.append(nn.Conv1d(in_features, mlps2[-1], 1))
        self.second_mpl = nn.Sequential(*second_mlp_list)


        #### DECODER
        coarse1 = [1024,1024,self.num_coarse*3]
        in_features = 1024
        decoder_list = []
        for m in range(0,len(coarse1)-1):
            decoder_list.append(nn.Linear(in_features, coarse1[m]))
            in_features = coarse1[m]
        decoder_list.append(nn.Linear(in_features, coarse1[-1]))
        self.decoder = nn.Sequential(*decoder_list)

        ## FOLDING
        mlpsfold = [512, 512,3]
        fold_mlp_list = []
        in_features = 1029
        for m in range(0,len(mlpsfold)-1):
            fold_mlp_list.append(nn.Conv1d(in_features, mlpsfold[m], 1))
            fold_mlp_list.append(nn.ReLU())
            in_features = mlpsfold[m]
        fold_mlp_list.append(nn.Conv1d(in_features, mlpsfold[-1], 1))
        self.fold_mpl = nn.Sequential(*fold_mlp_list)

    def point_maxpool(self,features,npts,keepdims=True):
        # splitted = torch.split(features,npts[0],dim=1)
        # outputs = [torch.max(f,dim=2,keepdims=keepdims)[0] for f in splitted]
        # return torch.cat(outputs,dim=0)
        return torch.max(features,dim=2,keepdims=keepdims)[0]


    def point_unpool(self,features,npts):
        # features = torch.split(features,features.shape[0],dim=0)
        # outputs  = [f.repeat([1,npts[i],1]) for i,f in enumerate(features)]
        # return torch.cat(outputs,dim=1)
        return features.repeat([1,1,256])


    def forward(self, xyz):
        xyz = xyz.permute(0,2,1)
        #####ENCODER
        features = self.first_mpl(xyz)
        features_global = self.point_maxpool(features.permute(0,2,1),self.npts,keepdims=True)
        features_global = self.point_unpool(features_global,self.npts)

        features = torch.cat([features,features_global.permute(0,2,1)],dim=1)
        features = self.second_mpl(features)
        features = self.point_maxpool(features.permute(0,2,1),self.npts).squeeze(2)

        ##DECODER
        coarse = self.decoder(features)
        coarse = coarse.view(-1,self.num_coarse,3)

        ##FOLDING
        grid_row = torch.linspace(-0.05,0.05,self.grid_size).cuda()
        grid_column = torch.linspace(-0.05,0.05,self.grid_size).cuda()
        grid = torch.meshgrid(grid_row,grid_column)
        grid = torch.reshape(torch.stack(grid,dim=2),(-1,2)).unsqueeze(0)
        grid_feat = grid.repeat([features.shape[0],self.num_coarse,1])
        # print("grid_Feat",grid_feat.shape)

        point_feat = coarse.unsqueeze(2).repeat([1,1,self.grid_size**2,1])
        point_feat = torch.reshape(point_feat, [-1,self.num_fine,3])
        # print("point_Feat",point_feat.shape)
        global_feat = features.unsqueeze(1).repeat([1,self.num_fine,1])
        # print("global_Feat",global_feat.shape)
        feat = torch.cat([grid_feat,point_feat,global_feat],dim=2)

        center = coarse.unsqueeze(2).repeat([1,1,self.grid_size**2,1])
        center = torch.reshape(center, [-1,self.num_fine,3])

        fine = self.fold_mpl(feat.permute(0,2,1))
        # print("fine shape",fine.shape," center shape",center.shape)
        fine = fine.permute(0,2,1)  + center

        return coarse, fine

    def create_loss(self,coarse,fine,gt,alpha):
        gt_ds = gt[:,:coarse.shape[1],:]
        loss_coarse = earth_mover_distance(coarse, gt_ds, transpose=False)
        dist1, dist2 = chamfer_dist(fine, gt)
        loss_fine = (torch.mean(dist1)) + (torch.mean(dist2))
        
        loss = loss_coarse + alpha * loss_fine

        return loss




if __name__ == '__main__':
    # alpha [ 0.01,0.1,0.5,1.0]
    for i in range(10):
        xyz = torch.rand(1, 1024,3).cuda()
        pcd1 = open3d.PointCloud()
        pcd1.points = open3d.Vector3dVector(xyz.data.cpu().numpy()[0])
        pcd1.colors = open3d.Vector3dVector(np.ones((1024,3))* [0.00,0.53,0.90])
        colors = torch.rand(1, 2048,3).cuda()
        net = PCNEMD()
        net.cuda()
        coarse, fine = net(xyz)
        net.create_loss(coarse,fine,xyz,1.0)

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(coarse.data.cpu().numpy()[0]+np.array([1.0,0.0,0.0]))
        pcd.colors = open3d.Vector3dVector(np.ones((1024,3))* [0.76,0.23,0.14])

        pcd2 = open3d.PointCloud()
        pcd2.points = open3d.Vector3dVector(fine.data.cpu().numpy()[0]+np.array([-1.0,0.0,0.0]))
        pcd2.colors = open3d.Vector3dVector(np.ones((fine.shape[1],3))* [0.16,0.53,0.44])
        open3d.draw_geometries([pcd,pcd1,pcd2])
        exit()
