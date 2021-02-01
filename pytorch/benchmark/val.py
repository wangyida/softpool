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
sys.path.append("../emd/")
import emd_module as emd
sys.path.append("../chamfer/")
import dist_chamfer as cd
from dataset import resample_pcd, read_points
EMD = emd.emdModule()
CD = cd.chamferDist()


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
    '--num_points', type=int, default=2048, help='number of points')
parser.add_argument(
    '--n_regions',
    type=int,
    default=16,
    help='number of primitives in the atlas')
parser.add_argument(
    '--env', type=str, default="SoftPool_VAL", help='visdom environment')
parser.add_argument(
    '--dataset', type=str, default="shapenet", help='dataset for evaluation')

opt = parser.parse_args()
print(opt)

network = Network(num_points=opt.num_points, n_regions=opt.n_regions)
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
    part_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial/"
    gt_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/"
elif opt.dataset == 'fusion':
    with open(os.path.join('./data/test_fusion.list')) as file:
        model_list = [line.strip().replace('/', '/') for line in file]
    part_dir = "/media/wangyida/HDD/database/050_200/test/pcd_partial/"
    gt_dir = "/media/wangyida/HDD/database/050_200/test/pcd_complete/"
elif opt.dataset == 'shapenet':
    hash_tab = {
        'all': {
            'name': 'Test',
            'label': 100,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '04530566': {
            'name': 'Watercraft',
            'label': 1,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '02933112': {
            'name': 'Cabinet',
            'label': 2,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '04379243': {
            'name': 'Table',
            'label': 3,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '02691156': {
            'name': 'Airplane',
            'label': 4,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '02958343': {
            'name': 'Car',
            'label': 5,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '03001627': {
            'name': 'Chair',
            'label': 6,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '04256520': {
            'name': 'Couch',
            'label': 7,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cnt': 0
        },
        '03636649': {
            'name': 'Lamp',
            'label': 8,
            'emd1': 0.0,
            'emd2': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
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
        # with open(os.path.join('./data/valid_shapenet.list')) as file:
        with open(os.path.join('../data/visual_shapenet.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
        gt_dir = "/media/wangyida/HDD/database/shapenet16384/val/gt/"

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

labels_generated_points = torch.Tensor(
    range(1,
          (opt.n_regions + 1) * (opt.num_points // opt.n_regions) + 1)).view(
              opt.num_points // opt.n_regions, (opt.n_regions + 1)).transpose(
                  0, 1)
labels_generated_points = (labels_generated_points) % (opt.n_regions + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

labels_inputs_points = torch.Tensor(range(0, opt.num_points)).view(
    1, opt.num_points).transpose(0, 1)
labels_inputs_points = (labels_inputs_points) % (opt.num_points + 1)
labels_inputs_points = labels_inputs_points.contiguous().view(-1)

with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        subfold = model[:model.rfind('/')]
        part = torch.zeros((1, opt.num_points, 3), device='cuda')
        part_seg = torch.zeros((1, opt.num_points, 3), device='cuda')
        input_chosen = torch.zeros((1, opt.num_points, 3), device='cuda')
        gt = torch.zeros((1, opt.num_points * 8, 3), device='cuda')
        gt_seg = torch.zeros((1, opt.num_points * 8, 3), device='cuda')
        gt_regions = torch.zeros((1, opt.num_points * 8, 3), device='cuda')
        """
        def read_points(filename, dataset=self.dataset):
            if self.dataset == 'suncg':
                pcd = o3d.read_point_cloud(filename)
                coord = torch.from_numpy(np.array(pcd.points)).float()
                color = torch.from_numpy(np.array(pcd.colors)).float()
                return coord, color
            elif self.dataset == 'shapenet':
                fh5 = h5py.File(filename, 'r')
                label = float(self.hash_tab[filename.split("/")[-2]]['label'])
                coord = torch.from_numpy(np.array(fh5['data'])).float()
                color = torch.from_numpy(np.ones_like(np.array(fh5['data'])) / 11 * label).float()
                return coord, color
        """

        for j in range(1):
            if opt.dataset == 'suncg' or opt.dataset == 'fusion':
                part1, part_color = read_points(
                    os.path.join(part_dir, model + '.pcd'), opt.dataset)
                gt1, gt_color = read_points(
                    os.path.join(gt_dir, model + '.pcd'), opt.dataset)
                part[j, :, :], idx_sampled = resample_pcd(
                    part1, opt.num_points)
                part_seg[j, :, :] = np.round(part_color[idx_sampled] * 11)
                gt[j, :, :], idx_sampled = resample_pcd(
                    gt1, opt.num_points * 8)
                gt_seg[j, :, :] = np.round(gt_color[idx_sampled] * 11)
            elif opt.dataset == 'shapenet':
                part1, part_color = read_points(
                    os.path.join(part_dir, model + '.h5'), opt.dataset)
                gt1, gt_color = read_points(
                    os.path.join(gt_dir, model + '.h5'), opt.dataset)
                part[j, :, :], idx_sampled = resample_pcd(
                    part1, opt.num_points)
                part_seg[j, :, :] = np.round(part_color[idx_sampled] * 11)
                gt[j, :, :], idx_sampled = resample_pcd(
                    gt1, opt.num_points * 8)
                gt_seg[j, :, :] = np.round(gt_color[idx_sampled] * 11)

        output1, output2, out_seg, input_chosen, _, _ = network(
            part.transpose(2, 1).contiguous(), part_seg)
        if opt.dataset == 'shapenet' and complete3d_benchmark == False:

            dist, _, _, _ = CD.forward(input1=output1[0], input2=gt)
            cd1 = dist.mean() * 1e4
            hash_tab[str(subfold)]['cd1'] += cd1

            dist, _, _, _ = CD.forward(input1=output2, input2=gt)
            cd2 = dist.mean() * 1e4
            hash_tab[str(subfold)]['cd2'] += cd2
            """
            dist, _, _, _ = CD.forward(input1=output3, input2=gt)
            cd3 = dist.mean() * 1e4
            hash_tab[str(subfold)]['cd3'] += cd3

            dist, _, _, _ = CD.forward(input1=output4, input2=gt)
            cd4 = dist.mean() * 1e4
            hash_tab[str(subfold)]['cd4'] += cd4
            """

            hash_tab[str(subfold)]['cnt'] += 1
            idx = random.randint(0, 0)
            print(
                opt.env + ' val [%d/%d]  cd1: %f cd2: %f mean cd2 so far: %f' %
                (i + 1, len(model_list), cd1.item(), cd2.item(),
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
        pts_color = matplotlib.cm.rainbow(gt_seg[0, :, 0].cpu() / 11)[:, :3]

        # pts_color = matplotlib.cm.cool(gt[0].data.cpu()[:, 1] - mini)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/gt',
            child=subfold,
            pfile=model)

        # save selected points on input
        pts_coord = input_chosen[0].data.cpu()[:, 0:3]
        """
        dist, _, idx1, _ = CD.forward(input1=input_chosen, input2=gt)
        pts_color = matplotlib.cm.rainbow(gt_seg[0, :, 0][idx1[0].long()].cpu() / 11)[:, 0:3]
        """
        maxi = labels_inputs_points.max() // 2
        pts_color = matplotlib.cm.rainbow(
            labels_inputs_points[0:input_chosen.size(1)] / maxi)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/regions_part',
            child=subfold,
            pfile=model)

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

        # save output1
        for stage in range(len(output1)):
            pts_coord = output1[stage][0].data.cpu()[:, 0:3]
            maxi = labels_generated_points.max()
            if stage == 0:
                pts_color = matplotlib.cm.rainbow(
                    labels_generated_points[0:output1[stage].size(1)] /
                    maxi)[:, 0:3]
            else:
                pts_color = matplotlib.cm.cool(
                    output1[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
            points_save(
                points=pts_coord,
                colors=pts_color,
                root='pcds/output1',
                child=subfold,
                pfile=model + '-' + str(stage))

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

        # save output3
        """
        pts_coord = output3[0].data.cpu()[:, 0:3]
        maxi = labels_generated_points.max()
        pts_color = matplotlib.cm.rainbow(
            labels_generated_points[0:output3.size(1)] / maxi)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/output3',
            child=subfold,
            pfile=model)

        # save output4
        pts_coord = output4[0].data.cpu()[:, 0:3]
        maxi = labels_generated_points.max()

        dist, _, idx1, _ = CD.forward(input1=output4, input2=gt)
        pts_color = matplotlib.cm.rainbow(
            gt_seg[0, :, 0][idx1[0].long()].cpu() / 11)[:, 0:3]
        cd4 = dist.mean()
        pts_color = matplotlib.cm.rainbow(
            labels_generated_points[0:output4.size(1)] / maxi)[:, 0:3]
        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/output4',
            child=subfold,
            pfile=model)
        """

    if opt.dataset == 'shapenet' and complete3d_benchmark == False:
        for i in [
                '04530566', '02933112', '04379243', '02691156', '02958343',
                '03001627', '04256520', '03636649'
        ]:
            print(
                '%s cd1: %f cd2: %f' %
                (hash_tab[i]['name'], hash_tab[i]['cd1'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd2'] / hash_tab[i]['cnt']))
