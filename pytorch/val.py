import sys
import matplotlib.cm
from matplotlib import pyplot as plt
import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import h5py 
import os
import visdom
sys.path.append("./emd/")
import emd_module as emd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default='./trained_model/network.pth',
    help='optional reload model path')
parser.add_argument(
    '--num_points', type=int, default=8192, help='number of points')
parser.add_argument(
    '--n_primitives',
    type=int,
    default=16,
    help='number of primitives in the atlas')
parser.add_argument(
    '--env', type=str, default="MSN_VAL", help='visdom environment')

opt = parser.parse_args()
print(opt)

network = MSN(num_points=opt.num_points, n_primitives=opt.n_primitives)
network.cuda()
network.apply(weights_init)

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

network.eval()
# with open(os.path.join('./data/valid_suncg_fur.list')) as file:
with open(os.path.join('./data/valid_shapenet.list')) as file:
    model_list = [line.strip().replace('/', '/') for line in file]

# partial_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial_fur/"
# gt_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete_fur/"
partial_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
gt_dir = "/media/wangyida/HDD/database/shapenet/val/gt/"

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


EMD = emd.emdModule()

labels_generated_points = torch.Tensor(
    range(1, (opt.n_primitives + 1) * (opt.num_points // opt.n_primitives) +
          1)).view(opt.num_points // opt.n_primitives,
                   (opt.n_primitives + 1)).transpose(0, 1)
labels_generated_points = (labels_generated_points) % (opt.n_primitives + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

labels_inputs_points = torch.Tensor(
    range(1, (opt.n_primitives + 1) * (2048 // opt.n_primitives) +
          1)).view(2048 // opt.n_primitives,
                   (opt.n_primitives + 1)).transpose(0, 1)
labels_inputs_points = (labels_inputs_points) % (opt.n_primitives + 1)
labels_inputs_points = labels_inputs_points.contiguous().view(-1)

with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        subfold = model[:model.rfind('/')]
        partial = torch.zeros((1, 2048, 3), device='cuda')
        partial_regions = torch.zeros((1, 2048, 3), device='cuda')
        gt = torch.zeros((1, opt.num_points, 3), device='cuda')
        for j in range(1):
            """
            pcd = o3d.read_point_cloud(
                os.path.join(partial_dir, model + '.pcd'))
            """
            fh5 = h5py.File(os.path.join(partial_dir, model + '.h5'), 'r')
            partial[j, :, :] = torch.from_numpy(
                resample_pcd(np.array(fh5['data']), 2048))
            """
            pcd = o3d.read_point_cloud(os.path.join(gt_dir, model + '.pcd'))
            """
            fh5 = h5py.File(os.path.join(gt_dir, model + '.h5'), 'r')
            gt[j, :, :] = torch.from_numpy(
                resample_pcd(np.array(fh5['data']), opt.num_points))

        output1, output2, expansion_penalty, out_seg, partial_regions = network(
            partial.transpose(2, 1).contiguous())
        dist, _ = EMD(output1, gt, 0.002, 10000)
        emd1 = torch.sqrt(dist).mean()
        dist, _ = EMD(output2, gt, 0.002, 10000)
        emd2 = torch.sqrt(dist).mean()
        idx = random.randint(0, 0)
        """
        vis.scatter(X = gt[idx].data.cpu(), win = 'GT',
                    opts = dict(title = model, markersize = 2))
        vis.scatter(X = partial[idx].data.cpu(), win = 'INPUT',
                    opts = dict(title = model, markersize = 2))
        vis.scatter(X = output1[idx].data.cpu(),
                    Y = labels_generated_points[0:output1.size(1)],
                    win = 'COARSE',
                    opts = dict(title = model, markersize=2))
        vis.scatter(X = output2[idx].data.cpu(),
                    win = 'OUTPUT',
                    opts = dict(title = model, markersize=2))
        """
        print(opt.env +
              ' val [%d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %
              (i + 1, len(model_list), emd1.item(), emd2.item(),
               expansion_penalty.mean().item()))
        os.makedirs('pcds/regions', exist_ok=True)
        os.makedirs('pcds/regions/'+subfold, exist_ok=True)
        pts_coord = partial_regions[idx].data.cpu()[:, 0:3]
        maxi = labels_inputs_points.max()
        # import ipdb; ipdb.set_trace()
        pts_color = matplotlib.cm.rainbow(
            labels_inputs_points[0:partial_regions.size(1)] / maxi)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/regions/', '%s.pcd' % model),
            pcd,
            write_ascii=True,
            compressed=True)
        os.makedirs('pcds/output1', exist_ok=True)
        os.makedirs('pcds/output1/'+subfold, exist_ok=True)
        pts_coord = output1[idx].data.cpu()[:, 0:3]
        maxi = labels_generated_points.max()
        # import ipdb; ipdb.set_trace()
        pts_color = matplotlib.cm.rainbow(
            labels_generated_points[0:output1.size(1)] / maxi)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/output1/', '%s.pcd' % model),
            pcd,
            write_ascii=True,
            compressed=True)
        os.makedirs('pcds/output2', exist_ok=True)
        os.makedirs('pcds/output2/'+subfold, exist_ok=True)
        pts_coord = output2[idx].data.cpu()[:, 0:3]
        mini = output2[idx].min()
        pts_color = matplotlib.cm.cool(output2[idx].data.cpu()[:, 1] -
                                       mini)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/output2/', '%s.pcd' % model),
            pcd,
            compressed=True)
        os.makedirs('pcds/input', exist_ok=True)
        os.makedirs('pcds/input/'+subfold, exist_ok=True)
        pts_coord = partial[idx].data.cpu()[:, 0:3]
        mini = partial[idx].min()
        pts_color = matplotlib.cm.cool(partial[idx].data.cpu()[:, 1] -
                                       mini)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/input/', '%s.pcd' % model),
            pcd,
            write_ascii=True,
            compressed=True)
        os.makedirs('pcds/gt', exist_ok=True)
        os.makedirs('pcds/gt/'+subfold, exist_ok=True)
        pts_coord = gt[idx].data.cpu()[:, 0:3]
        mini = gt[idx].min()
        pts_color = matplotlib.cm.cool(gt[idx].data.cpu()[:, 1] - mini)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/gt/', '%s.pcd' % model), pcd, compressed=True)

        """
        os.makedirs('pcds/spblocks', exist_ok=True)
        os.makedirs('pcds/spblocks/'+subfold, exist_ok=True)
        softpoolblock = softpool[idx].data.cpu()[:, 0:3, :]
        softpoolblock = softpoolblock.reshape((64, 32, 3))
        plt.imsave(
            os.path.join('./pcds/spblocks/', '%s.png' % model), softpoolblock)
        """
