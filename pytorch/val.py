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
from chamfer_pkg.dist_chamfer import chamferDist as cd
cd = cd()

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
parser.add_argument(
    '--dataset', type=str, default="shapenet", help='dataset for evaluation')

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
if opt.dataset == 'suncg':
    with open(os.path.join('./data/valid_suncg_fur.list')) as file:
        model_list = [line.strip().replace('/', '/') for line in file]
    partial_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial_fur/"
    gt_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete_fur/"
elif opt.dataset == 'shapenet':
    hash_tab = {
        'all': {
            'name': 'Test',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '04530566': {
            'name': 'Watercraft',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '02933112': {
            'name': 'Cabinet',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '04379243': {
            'name': 'Table',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '02691156': {
            'name': 'Airplane',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '02958343': {
            'name': 'Car',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '03001627': {
            'name': 'Chair',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '04256520': {
            'name': 'Couch',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        },
        '03636649': {
            'name': 'Lamp',
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cnt': 0
        }
    }
    # with open(os.path.join('./data/valid_shapenet.list')) as file:
    with open(os.path.join('./data/test_shapenet.list')) as file:
        model_list = [line.strip().replace('/', '/') for line in file]
    # partial_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
    partial_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"
    # gt_dir = "/media/wangyida/HDD/database/shapenet/val/gt/"
    gt_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"

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
    range(1, (opt.n_primitives + 1) * (2048 // opt.n_primitives) + 1)).view(
        2048 // opt.n_primitives, (opt.n_primitives + 1)).transpose(0, 1)
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
            if opt.dataset == 'suncg':
                pcd = o3d.read_point_cloud(
                    os.path.join(partial_dir, model + '.pcd'))
                partial[j, :, :] = torch.from_numpy(
                    resample_pcd(np.array(pcd.points), 2048))
                pcd = o3d.read_point_cloud(
                    os.path.join(gt_dir, model + '.pcd'))
                gt[j, :, :] = torch.from_numpy(
                    resample_pcd(np.array(pcd.points), opt.num_points))
            elif opt.dataset == 'shapenet':
                fh5 = h5py.File(os.path.join(partial_dir, model + '.h5'), 'r')
                partial[j, :, :] = torch.from_numpy(
                    resample_pcd(np.array(fh5['data']), 2048))
                fh5 = h5py.File(os.path.join(gt_dir, model + '.h5'), 'r')
                gt[j, :, :] = torch.from_numpy(
                    resample_pcd(np.array(fh5['data']), opt.num_points))

        output1, output2, output3, expansion_penalty, out_seg, partial_regions = network(
            partial.transpose(2, 1).contiguous())
        dist, _ = EMD(output1, gt, 0.002, 10000)
        emd1 = torch.sqrt(dist).mean()
        hash_tab[str(subfold)]['cnt'] += 1
        hash_tab[str(subfold)]['emd1'] += emd1

        dist, _ = EMD(output2, gt, 0.002, 10000)
        emd2 = torch.sqrt(dist).mean()
        hash_tab[str(subfold)]['emd2'] += emd2

        dist, _ = EMD(output3, gt, 0.002, 10000)
        emd3 = torch.sqrt(dist).mean()
        hash_tab[str(subfold)]['emd3'] += emd3

        dist, _ = cd.forward(input1=output1, input2=gt)
        cd1 = dist.mean()
        hash_tab[str(subfold)]['cd1'] += cd1

        dist, _ = cd.forward(input1=output2, input2=gt)
        cd2 = dist.mean()
        hash_tab[str(subfold)]['cd2'] += cd2

        dist, _ = cd.forward(input1=output3, input2=gt)
        cd3 = dist.mean()
        hash_tab[str(subfold)]['cd3'] += cd3

        idx = random.randint(0, 0)
        print(
            opt.env +
            ' val [%d/%d]  emd1: %f emd2: %f emd3: %f cd2: %f expansion_penalty: %f, mean cd2: %f'
            % (i + 1, len(model_list), emd1.item(), emd2.item(), emd3.item(),
               cd2.item(), expansion_penalty.mean().item(),
               hash_tab[str(subfold)]['cd2'] / hash_tab[str(subfold)]['cnt']))
        os.makedirs('pcds/regions', exist_ok=True)
        os.makedirs('pcds/regions/' + subfold, exist_ok=True)
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
        os.makedirs('pcds/output1/' + subfold, exist_ok=True)
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
        os.makedirs('pcds/output3', exist_ok=True)
        os.makedirs('pcds/output3/' + subfold, exist_ok=True)
        pts_coord = output3[idx].data.cpu()[:, 0:3]
        maxi = labels_generated_points.max()
        # import ipdb; ipdb.set_trace()
        pts_color = matplotlib.cm.rainbow(
            labels_generated_points[0:output1.size(1)] / maxi)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/output3/', '%s.pcd' % model),
            pcd,
            write_ascii=True,
            compressed=True)
        os.makedirs('pcds/output2', exist_ok=True)
        os.makedirs('pcds/output2/' + subfold, exist_ok=True)
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
        # Submission
        os.makedirs('benchmark', exist_ok=True)
        os.makedirs('benchmark/' + subfold, exist_ok=True)
        with h5py.File('benchmark/' + model + '.h5', "w") as f:
            f.create_dataset("data", data=np.float32(pts_coord))

        os.makedirs('pcds/input', exist_ok=True)
        os.makedirs('pcds/input/' + subfold, exist_ok=True)
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
        os.makedirs('pcds/gt/' + subfold, exist_ok=True)
        pts_coord = gt[idx].data.cpu()[:, 0:3]
        mini = gt[idx].min()
        pts_color = matplotlib.cm.cool(gt[idx].data.cpu()[:, 1] - mini)[:, 0:3]
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.float32(pts_coord))
        pcd.colors = o3d.Vector3dVector(np.float32(pts_color))
        o3d.write_point_cloud(
            os.path.join('./pcds/gt/', '%s.pcd' % model), pcd, compressed=True)

    """
    if opt.dataset == 'shapenet':
        for i in ['04530566', '02933112', '04379243', '02691156', '02958343', '03001627', '04256520', '03636649']:
            print('%s cd1: %f cd2: %f cd3: %f' % (hash_tab[i]['name'], hash_tab[i]['cd1'] / hash_tab[i]['cnt'], hash_tab[i]['cd2'] / hash_tab[i]['cnt'], hash_tab[i]['cd3'] / hash_tab[i]['cnt']))
    """
