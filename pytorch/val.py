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
from dataset import resample_pcd
cd = cd()


def points_save(points, colors, root='pcds/regions', child='all', pfile=''):
    os.makedirs(root, exist_ok=True)
    os.makedirs(root + '/' + child, exist_ok=True)
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(np.float32(points))
    pcd.colors = o3d.Vector3dVector(np.float32(colors))
    o3d.write_point_cloud(
        os.path.join(root, '%s.pcd' % pfile),
        pcd,
        write_ascii=True,
        compressed=True)


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
    with open(os.path.join('./data/valid_suncg.list')) as file:
        model_list = [line.strip().replace('/', '/') for line in file]
    # part_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial_fur/"
    part_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/"
    gt_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/"
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
    complete3d_benchmark = False
    if complete3d_benchmark == True:
        with open(os.path.join('./data/test_shapenet.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"
        gt_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"
    else:
        with open(os.path.join('./data/valid_shapenet.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
        # part_dir = "/media/wangyida/HDD/database/shapenet/val/gt/"
        gt_dir = "/media/wangyida/HDD/database/shapenet/val/gt/"

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

EMD = emd.emdModule()

labels_generated_points = torch.Tensor(
    range(1, (opt.n_primitives + 1) * (opt.num_points // opt.n_primitives) +
          1)).view(opt.num_points // opt.n_primitives,
                   (opt.n_primitives + 1)).transpose(0, 1)
labels_generated_points = (labels_generated_points) % (opt.n_primitives + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

labels_inputs_points = torch.Tensor(range(0, opt.num_points)).view(1, opt.num_points).transpose(
    0, 1)
labels_inputs_points = (labels_inputs_points) % (opt.num_points + 1)
labels_inputs_points = labels_inputs_points.contiguous().view(-1)

with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        subfold = model[:model.rfind('/')]
        part = torch.zeros((1, opt.num_points, 3), device='cuda')
        part_seg = torch.zeros((1, opt.num_points, 3), device='cuda')
        part_regions = torch.zeros((1, opt.num_points, 3), device='cuda')
        gt = torch.zeros((1, opt.num_points, 3), device='cuda')
        gt_regions = torch.zeros((1, opt.num_points, 3), device='cuda')
        for j in range(1):
            if opt.dataset == 'suncg':
                pcd = o3d.read_point_cloud(
                    os.path.join(part_dir, model + '.pcd'))
                part_sampled, idx_sampled = resample_pcd(
                    np.array(pcd.points), opt.num_points)
                part_seg_sampled = np.round(
                    np.array(pcd.colors)[idx_sampled] * 11)
                part[j, :, :] = torch.from_numpy(part_sampled)
                part_seg[j, :, :] = torch.from_numpy(part_seg_sampled)

                pcd = o3d.read_point_cloud(
                    os.path.join(gt_dir, model + '.pcd'))
                gt_sampled, idx_sampled = resample_pcd(
                    np.array(pcd.points), opt.num_points)
                gt_seg_sampled = np.round(
                    np.array(pcd.colors)[idx_sampled] * 11)
                gt[j, :, :] = torch.from_numpy(gt_sampled)
            elif opt.dataset == 'shapenet':
                fh5 = h5py.File(os.path.join(part_dir, model + '.h5'), 'r')
                part[j, :, :], _ = torch.from_numpy(
                    resample_pcd(np.array(fh5['data']), opt.num_points))
                fh5 = h5py.File(os.path.join(gt_dir, model + '.h5'), 'r')
                gt[j, :, :], _ = torch.from_numpy(
                    resample_pcd(np.array(fh5['data']), opt.num_points))

        output1, output2, output3, output4, expansion_penalty, out_seg, part_regions, _ = network(
            part.transpose(2, 1).contiguous(), part_seg)
        """
        _, _, _, _, _, _, gt_regions, _ = network(
            gt.transpose(2, 1).contiguous())
        """
        if opt.dataset == 'shapenet' and complete3d_benchmark == False:
            dist, _ = EMD(output1[0], gt, 0.002, 10000)
            emd1 = torch.sqrt(dist).mean()
            hash_tab[str(subfold)]['cnt'] += 1
            hash_tab[str(subfold)]['emd1'] += emd1

            dist, _ = EMD(output2, gt, 0.002, 10000)
            emd2 = torch.sqrt(dist).mean()
            hash_tab[str(subfold)]['emd2'] += emd2

            dist, _ = EMD(output3[0], gt, 0.002, 10000)
            emd3 = torch.sqrt(dist).mean()
            hash_tab[str(subfold)]['emd3'] += emd3

            dist, _ = cd.forward(input1=output1[0], input2=gt)
            cd1 = dist.mean()
            hash_tab[str(subfold)]['cd1'] += cd1

            dist, _ = cd.forward(input1=output2, input2=gt)
            cd2 = dist.mean()
            hash_tab[str(subfold)]['cd2'] += cd2

            dist, _ = cd.forward(input1=output3[0], input2=gt)
            cd3 = dist.mean()
            hash_tab[str(subfold)]['cd3'] += cd3

            idx = random.randint(0, 0)
            print(
                opt.env +
                ' val [%d/%d]  emd1: %f emd2: %f emd3: %f cd2: %f expansion_penalty: %f, mean cd2: %f'
                %
                (i + 1, len(model_list), emd1.item(), emd2.item(), emd3.item(),
                 cd2.item(), expansion_penalty.mean().item(),
                 hash_tab[str(subfold)]['cd2'] / hash_tab[str(subfold)]['cnt'])
            )

        # save input
        pts_coord = part[0].data.cpu()[:, 0:3]
        mini = part[0].min()
        pts_color = matplotlib.cm.cool(part[0].data.cpu()[:, 1] - mini)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/input',
            child=subfold,
            pfile=model)

        # save gt
        pts_coord = gt[0].data.cpu()[:, 0:3]
        mini = gt[0].min()
        pts_color = matplotlib.cm.cool(gt[0].data.cpu()[:, 1] - mini)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/gt',
            child=subfold,
            pfile=model)

        # save selected points on input
        pts_coord = []
        for i in range(np.size(part_regions)):
            pts_coord.append(part_regions[i][0].data.cpu()[:, 0:3])
            maxi = labels_inputs_points.max()
            pts_color = matplotlib.cm.rainbow(
                labels_inputs_points[0:part_regions[i].size(1)] / maxi)[:, 0:3]
            points_save(
                points=pts_coord[i],
                colors=pts_color,
                root='pcds/regions_part',
                child=subfold,
                pfile=model + '-' + str(i))

        # save selected points on groung truth
        """
        pts_coord = []
        for i in range(np.size(gt_regions)):
            pts_coord.append(gt_regions[i][0].data.cpu()[:, 0:3])
            maxi = labels_inputs_points.max()
            pts_color = matplotlib.cm.plasma(
                labels_inputs_points[0:gt_regions[i].size(1)] / maxi)[:, 0:3]
            points_save(
                points=pts_coord[i],
                colors=pts_color,
                root='pcds/regions_gt',
                child=subfold,
                pfile=model + '-' + str(i))
        """

        pts_coord = []
        for i in range(np.size(output1)):
            # save output1
            pts_coord.append(output1[i][0].data.cpu()[:, 0:3])
            maxi = labels_generated_points.max()
            pts_color = matplotlib.cm.rainbow(
                labels_generated_points[0:output1[i].size(1)] / maxi)[:, 0:3]
            points_save(
                points=pts_coord[i],
                colors=pts_color,
                root='pcds/output1',
                child=subfold,
                pfile=model + '-' + str(i))

        # save output2
        pts_coord = output2[0].data.cpu()[:, 0:3]
        mini = output2[0].min()
        pts_color = matplotlib.cm.cool(output2[0].data.cpu()[:, 1] -
                                       mini)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/output2',
            child=subfold,
            pfile=model)
        # Submission
        if opt.dataset == 'shapenet' and complete3d_benchmark == True:
            os.makedirs('benchmark', exist_ok=True)
            os.makedirs('benchmark/' + subfold, exist_ok=True)
            with h5py.File('benchmark/' + model + '.h5', "w") as f:
                f.create_dataset("data", data=np.float32(pts_coord))

        pts_coord = []
        for i in range(np.size(output3)):
            # save output3
            pts_coord.append(output3[i][0].data.cpu()[:, 0:3])
            maxi = labels_generated_points.max()
            pts_color = matplotlib.cm.rainbow(
                labels_generated_points[0:output3[i].size(1)] / maxi)[:, 0:3]
            points_save(
                points=pts_coord[i],
                colors=pts_color,
                root='pcds/output3',
                child=subfold,
                pfile=model + '-' + str(i))

        # save output4
        pts_coord = output4[0].data.cpu()[:, 0:3]
        maxi = labels_generated_points.max()
        pts_color = matplotlib.cm.rainbow(
            labels_generated_points[0:output4.size(1)] / maxi)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/output4',
            child=subfold,
            pfile=model)

    if opt.dataset == 'shapenet' and complete3d_benchmark == False:
        for i in [
                '04530566', '02933112', '04379243', '02691156', '02958343',
                '03001627', '04256520', '03636649'
        ]:
            print(
                '%s cd1: %f cd2: %f cd3: %f' %
                (hash_tab[i]['name'], hash_tab[i]['cd1'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd2'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd3'] / hash_tab[i]['cnt']))
