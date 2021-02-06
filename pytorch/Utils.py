import open3d
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch import cuda, FloatTensor, LongTensor
from xConv import XConv,Dense
import sys

from sklearn.neighbors import  NearestNeighbors
#import matplotlib.pyplot as plt
params = {'batch_size': 5,
          'shuffle': False,
          'num_workers': 4}
def knn_indices_func_cpu(rep_pts : FloatTensor,  # (N, pts, dim)
                         pts : FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> LongTensor:         # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: dilatation factor
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    if rep_pts.is_cuda:
        rep_pts = rep_pts.cpu()
    if pts.is_cuda:
        pts = pts.cpu()
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()

    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "auto").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis = 0))

    return region_idx
