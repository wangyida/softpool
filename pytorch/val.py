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
sys.path.append("./chamfer/")
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
parser.add_argument(
    '--methods',
    nargs='+',
    default=['softpool', 'msn', 'folding', 'grnet'],
    help='a list of methods')

opt = parser.parse_args()
print(opt)

network = Network(
    num_points=opt.num_points,
    n_regions=opt.n_regions,
    model_lists=opt.methods)
network.cuda()
# network.apply(weights_init)

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
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '04530566': {
            'name': 'Watercraft',
            'label': 1,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '02933112': {
            'name': 'Cabinet',
            'label': 2,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '04379243': {
            'name': 'Table',
            'label': 3,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '02691156': {
            'name': 'Airplane',
            'label': 4,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '02958343': {
            'name': 'Car',
            'label': 5,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '03001627': {
            'name': 'Chair',
            'label': 6,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '04256520': {
            'name': 'Couch',
            'label': 7,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
            'cnt': 0
        },
        '03636649': {
            'name': 'Lamp',
            'label': 8,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'cd1': 0.0,
            'cd2': 0.0,
            'cd3': 0.0,
            'cd4': 0.0,
            'cd5': 0.0,
            'cd6': 0.0,
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
        with open(os.path.join('./data/visual_shapenet.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
        gt_dir = "/media/wangyida/HDD/database/shapenet16384/val/gt/"

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port


def labels_points(num_points, divisions):
    labels_for_points = torch.Tensor(
        range(1, (divisions + 1) * (num_points // divisions) + 1))
    labels_for_points = labels_for_points.view(num_points // divisions,
                                               (divisions + 1)).transpose(
                                                   0, 1)
    labels_for_points = (labels_for_points) % (divisions + 1)
    labels_for_points = labels_for_points.contiguous().view(-1)
    return labels_for_points


with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        if opt.dataset == 'suncg':
            subfold = 'all_samples'
        else:
            subfold = model[:model.rfind('/')]
        part = torch.zeros((1, opt.num_points, 3), device='cuda')
        part_seg = torch.zeros((1, opt.num_points, 3), device='cuda')
        input_chosen = torch.zeros((1, opt.num_points, 3), device='cuda')
        gt = torch.zeros((1, opt.num_points * 8, 3), device='cuda')
        gt_seg = torch.zeros((1, opt.num_points * 8, 3), device='cuda')
        gt_regions = torch.zeros((1, opt.num_points * 8, 3), device='cuda')

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

        output1, output2, output3, output4, output5, output6 = network(
            part.transpose(2, 1).contiguous(), part_seg)
        if opt.dataset == 'shapenet' and complete3d_benchmark == False:
            if output1:
                _, dist, _, _ = CD.forward(input1=output1[1], input2=gt)
                cd1 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd1'] += cd1
            else:
                hash_tab[str(subfold)]['cd1'] += 0

            if output2:
                _, dist, _, _ = CD.forward(input1=output2[1], input2=gt)
                cd2 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd2'] += cd2
            else:
                hash_tab[str(subfold)]['cd2'] += 0

            if output3:
                _, dist, _, _ = CD.forward(input1=output3[0], input2=gt)
                cd3 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd3'] += cd3
            else:
                hash_tab[str(subfold)]['cd3'] += 0

            if output4:
                _, dist, _, _ = CD.forward(input1=output4[2], input2=gt)
                cd4 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd4'] += cd4
            else:
                hash_tab[str(subfold)]['cd4'] += 0

            if output5:
                _, dist, _, _ = CD.forward(input1=output5[0], input2=gt)
                cd5 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd5'] += cd5
            else:
                hash_tab[str(subfold)]['cd5'] += 0

            if output6:
                _, dist, _, _ = CD.forward(input1=output6[1], input2=gt)
                cd6 = dist.mean() * 1e4
                hash_tab[str(subfold)]['cd6'] += cd6
            else:
                hash_tab[str(subfold)]['cd6'] += 0

            hash_tab[str(subfold)]['cnt'] += 1
            idx = random.randint(0, 0)
            print(
                opt.env +
                ' val [%d/%d]  cd1: %.2f cd2: %.2f cd3: %.2f cd4: %.2f cd5: %.2f cd6: %.2f mean cd2 so far: %.2f'
                %
                (i + 1, len(model_list), cd1.item() if output1 else 0,
                 cd2.item() if output2 else 0, cd3.item() if output3 else 0,
                 cd4.item() if output4 else 0, cd5.item() if output5 else 0,
                 cd6.item() if output6 else 0, hash_tab[str(subfold)]['cd2'] /
                 hash_tab[str(subfold)]['cnt']))
        if opt.dataset == 'suncg':
            model = 'all_samples/' + model

        # save input
        pts_coord = part[0].data.cpu()[:, 0:3]
        mini = part[0].min()
        pts_color = matplotlib.cm.copper(part[0].data.cpu()[:, 1] + 1)[:, 0:3]
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

        points_save(
            points=pts_coord,
            colors=pts_color,
            root='pcds/gt',
            child=subfold,
            pfile=model)

        # save selected points on input
        if output1:
            pts_coord = output1[4][0].data.cpu()[:, 0:3]
            labels_for_points = labels_points(
                num_points=opt.num_points, divisions=opt.num_points)
            maxi = labels_for_points.max()
            pts_color = matplotlib.cm.plasma(
                labels_for_points[0:input_chosen.size(1)] / maxi)[:, 0:3]
            points_save(
                points=pts_coord,
                colors=pts_color,
                root='pcds/input_sort',
                child=subfold,
                pfile=model)

            # save selected points on groung truth
            """
            pts_coord = []
            for i in range(np.size(gt_regions)):
                pts_coord.append(gt_regions[i][0].data.cpu()[:, 0:3])
                maxi = labels_for_points.max()
                pts_color = matplotlib.cm.plasma(
                    labels_for_points[0:gt_regions[i].size(1)] / maxi)[:, 0:3]
                points_save(
                    points=pts_coord[i],
                    colors=pts_color,
                    root='pcds/regions_gt',
                    child=subfold,
                    pfile=model + '-' + str(i))
            """

            # save output1
            for stage in range(5):
                pts_coord = output1[stage][0].data.cpu()[:, 0:3]
                if stage == 0:
                    labels_for_points = labels_points(
                        num_points=opt.num_points,
                        divisions=np.max((2, opt.n_regions)))
                    maxi = labels_for_points.max()
                    pts_color = matplotlib.cm.rainbow(
                        labels_for_points[0:output1[stage].size(1)] /
                        maxi)[:, 0:3]
                else:
                    pts_color = matplotlib.cm.copper(
                        output1[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds1',
                    child=subfold,
                    pfile=model + '-' + str(stage))

            # Submission
            if opt.dataset == 'shapenet' and complete3d_benchmark == True:
                os.makedirs('benchmark', exist_ok=True)
                os.makedirs('benchmark/' + subfold, exist_ok=True)
                with h5py.File('benchmark/' + model + '.h5', "w") as f:
                    f.create_dataset("data", data=np.float32(pts_coord))

        if output2:
            # save output2
            for stage in range(2):
                pts_coord = output2[stage][0].data.cpu()[:, 0:3]
                if stage == 0:
                    labels_for_points = labels_points(
                        num_points=8192, divisions=16)
                    maxi = labels_for_points.max()
                    pts_color = matplotlib.cm.rainbow(
                        labels_for_points[0:output2[stage].size(1)] /
                        maxi)[:, 0:3]
                else:
                    pts_color = matplotlib.cm.copper(
                        output2[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds2',
                    child=subfold,
                    pfile=model + '-' + str(stage))

        if output3:
            # save output3
            for stage in range(len(output3)):
                pts_coord = output3[stage][0].data.cpu()[:, 0:3]
                labels_for_points = labels_points(
                    num_points=opt.num_points, divisions=opt.num_points)
                maxi = labels_for_points.max()
                pts_color = matplotlib.cm.rainbow(
                    labels_for_points[0:input_chosen.size(1)] / maxi)[:, 0:3]
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds3',
                    child=subfold,
                    pfile=model)

        if output4:
            # save output4
            for stage in range(3):
                pts_coord = output4[stage][0].data.cpu()[:, 0:3]

                _, dist, idx1, _ = CD.forward(input1=output4[stage], input2=gt)
                if stage == 0:
                    pts_color = matplotlib.cm.copper(
                        output4[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
                elif stage == 1:
                    pts_color = matplotlib.cm.rainbow(
                        torch.argmax(output4[3][0][:, :].cpu(),
                                     dim=-1).float() / 11)[:, 0:3]
                else:
                    pts_color = matplotlib.cm.rainbow(
                        torch.argmax(output4[4][0][:, :].cpu(),
                                     dim=-1).float() / 11)[:, 0:3]
                cd4 = dist.mean()
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds4',
                    child=subfold,
                    pfile=model + '-' + str(stage))

            # save mesh from voxels
            os.makedirs('pcds/all_mesh/%s' % subfold, exist_ok=True)
            import mcubes
            voxels = torch.flip(
                output4[5][0, 0, :, :, :].transpose(2, 0).transpose(2, 1), [0])
            voxels = np.array(voxels.cpu())
            vertices, triangles = mcubes.marching_cubes(voxels, 0)
            vertices /= 64.0
            vertices -= 0.5
            # vertices[:, 2] += 0.0125
            mcubes.export_obj(vertices, triangles,
                              'pcds/all_mesh/%s.obj' % model)

        if output5:
            # save output5
            for stage in range(len(output5)):
                pts_coord = output5[stage][0].data.cpu()[:, 0:3]
                pts_color = matplotlib.cm.copper(
                    output5[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds5',
                    child=subfold,
                    pfile=model)

        if output6:
            # save outpu6
            for stage in range(len(output6)):
                pts_coord = output6[stage][0].data.cpu()[:, 0:3]
                if stage == 1:
                    labels_for_points = labels_points(
                        num_points=16384, divisions=1024)
                    maxi = labels_for_points.max()
                    pts_color = matplotlib.cm.rainbow(
                        labels_for_points[0:output6[stage].size(1)] /
                        maxi)[:, 0:3]
                else:
                    pts_color = matplotlib.cm.copper(
                        output6[stage][0].data.cpu()[:, 1] + 1)[:, 0:3]
                points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/all_clouds6',
                    child=subfold,
                    pfile=model + '-' + str(stage))

    if opt.dataset == 'shapenet' and complete3d_benchmark == False:
        names_categories = [
            '04530566', '02933112', '04379243', '02691156', '02958343',
            '03001627', '04256520', '03636649'
        ]
        min_samples = 1
        for i in names_categories:
            hash_tab[i]['cnt'] = np.max((hash_tab[i]['cnt'], min_samples))
            print(
                '%s cd1: %.2f cd2: %.2f cd3: %.2f cd4: %.2f cd5: %.2f cd6: %.2f'
                %
                (hash_tab[i]['name'], hash_tab[i]['cd1'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd2'] / hash_tab[i]['cnt'], hash_tab[i]['cd3'] /
                 hash_tab[i]['cnt'], hash_tab[i]['cd4'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd5'] / hash_tab[i]['cnt'],
                 hash_tab[i]['cd6'] / hash_tab[i]['cnt']))
