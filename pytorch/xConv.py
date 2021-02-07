import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch import cuda, FloatTensor, LongTensor
from typing import Tuple, Callable, Optional
from typing import Union


class Dense(nn.Module):
    def __init__(self, inn, out, drop=0, acti=True):
        super(Dense, self).__init__()
        self.inn = inn
        self.out = out
        self.acti = acti
        self.drop = drop
        self.linear = nn.Linear(inn, out)
        self.elu = nn.ELU()
        if (self.drop > 0):
            self.dropout = nn.Dropout(drop)

    def forward(self, x):
        out = self.linear(x.float())
        if (self.acti):
            out = self.elu(out)
        if (self.drop > 0):
            out = self.dropout(out)
            return out
        return out


class XConv(nn.Module):
    def __init__(self, C_in: int, C_out: int, dims: int, K: int, P: int,
                 C_mid: int, depth_multiplier: int):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)
        self.C_in = C_in
        self.C = C_out
        self.P = P
        self.K = K
        ###get x ###
        #x = x.permute(0,3,1,2)#
        self.conv1 = nn.Sequential(
            nn.Conv2d(dims, K * K, (1, K), bias=True), nn.ELU())
        #x = x.permute(0,2,3,1)#
        self.x_dense1 = Dense(K * K, K * K)
        self.x_dense2 = Dense(K * K, K * K, acti=False)

        ### end Conv ###
        #x = x.permute(0,3,1,2)#
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                C_mid + C_in, (C_mid + C_in) * depth_multiplier, (1, K),
                groups=C_mid + C_in),
            nn.Conv2d((C_mid + C_in) * depth_multiplier, C_out, 1, bias=True),
            nn.ELU(), nn.BatchNorm2d(C_out, momentum=0.9))
        #x = x.permute(0,2,3,1)#
    def forward(self, rep_pt, pts, fts):
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
         _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
         indices = indices_dilated[:, :, ::D, :]
        """

        N = len(pts)
        P = rep_pt.shape[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)
        ##FIRST STEP :  Move pts to local coordinates of the reference point ##
        #print("COUCOU  " +   str(pts.shape) + "    :    " +   str(p_center.shape))
        pts_local = pts - p_center  # (N, P, K, dims)
        #print("HELLO " + str(pts_local.shape))

        ##SECOND STEP : We lift every point individually to C_mid space
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted = self.dense2(fts_lifted0)  # (N, P, K, C_mid)
        ## THIRD STEP : We check if there are already features as input (first layer or not) and cocnatenate Fsigma and previous F
        if fts is None:
            fts_cat = fts_lifted
        else:
            #print("concatenation : " + str(fts_lifted.shape) + " : " + str(fts.shape) )
            fts_cat = torch.cat((fts_lifted, fts),
                                -1)  # (N, P, K, C_mid + C_in)

        ##FOURTH STEP : We need to learn the transformation matrix
        X_shape = (N, P, self.K, self.K)
        X = pts_local.permute(0, 3, 1, 2)
        X = self.conv1(X)
        X = X.permute(0, 2, 3, 1)
        X = self.x_dense1(X)
        X = self.x_dense2(X)
        #print("X SHAPE  "+ str(X.shape))
        X = X.view(*X_shape)
        #print("X SHAPE  "+ str(X.shape))

        ## FIFTH STEP : we weight and permute F* with X
        fts_X = torch.matmul(X, fts_cat)
        #print("FTS_X SHAPE  " + str(fts_X.shape))
        #SIXTH STEP : Last convolution giving us the output of the X convolution
        X2 = fts_X.permute(0, 3, 1, 2)
        X2 = self.conv2(X2)
        x2 = X2.permute(0, 2, 3, 1)
        fts_p = X2.squeeze(dim=2)
        #print(fts_p.shape)

        # tranform to (N,P,K,C/K)
        fts_p = fts_p.permute(0, 2, 1, 3)
        fts_shape = (N, len(fts_p[0]), 8, int(self.C / self.K))
        fts_p = fts_p.squeeze(dim=3)
        #fts_p = fts_p.view(fts_shape)
        # print("################# END CONV FEATURES###############")
        # print(fts_p.shape)
        # print("##################################################")
        return fts_p


### TEST THE X CONVOLUTION ###
if __name__ == "__main__":
    N = 4
    D = 3
    C_in = 8
    C_out = 32
    N_neighbors = 100
    convo = XConv(4, 8, 2, 10, 1, 1000, 10).cuda()
    print(convo)
