import open3d as o3d
import torch
import h5py
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random

#from utils import *


def read_points(filename, dataset):
    if dataset == 'suncg' or dataset == 'fusion':
        pcd = o3d.read_point_cloud(filename)
        coord = torch.from_numpy(np.array(pcd.points)).float()
        color = torch.from_numpy(np.array(pcd.colors)).float()
        return coord, color
    elif dataset == 'shapenet':
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
                'cnt': 0
            }
        }
        fh5 = h5py.File(filename, 'r')
        label = float(hash_tab[filename.split("/")[-2]]['label'])
        coord = torch.from_numpy(np.array(fh5['data'])).float()
        color = torch.from_numpy(
            np.ones_like(np.array(fh5['data'])) / 11 * label).float()
        return coord, color


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]], idx[:n]


class ShapeNet(data.Dataset):
    def __init__(self, train=True, npoints=2048, dataset_name='shapenet'):
        self.dataset = dataset_name
        if train:
            if self.dataset == 'suncg':
                self.list_path = './data/train_suncg.list'
            elif self.dataset == 'fusion':
                self.list_path = './data/train_fusion.list'
            elif self.dataset == 'shapenet':
                self.list_path = './data/train_shapenet.list'
        else:
            if self.dataset == 'suncg':
                self.list_path = './data/valid_suncg.list'
            elif self.dataset == 'fusion':
                self.list_path = './data/test_fusion.list'
            elif self.dataset == 'shapenet':
                self.list_path = './data/valid_shapenet.list'
        self.npoints = npoints
        self.train = train

        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip().replace('/', '/') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list)

    def __getitem__(self, index):
        model_id = self.model_list[index]
        scan_id = index

        if self.train:
            if self.dataset == 'suncg':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_partial/",
                        '%s.pcd' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            if self.dataset == 'fusion':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/train/pcd_partial/",
                        '%s.pcd' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/train/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == 'shapenet':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/train/partial/",
                        '%s.h5' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet16384/train/gt/",
                        '%s.h5' % model_id), self.dataset)
        else:
            if self.dataset == 'suncg':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial/",
                        '%s.pcd' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == 'fusion':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/test/pcd_partial/",
                        '%s.pcd' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/test/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == 'shapenet':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/partial/",
                        '%s.h5' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet16384/val/gt/",
                        '%s.h5' % model_id), self.dataset)
        part_sampled, idx_sampled = resample_pcd(part, self.npoints)
        part_seg = np.round(part_color[idx_sampled] * 11)
        comp_sampled, idx_sampled = resample_pcd(comp, self.npoints * 8)
        comp_seg = np.round(comp_color[idx_sampled] * 11)
        """
        comp_seg = []
        for i in range (1, 12):
            import ipdb; ipdb.set_trace()
            comp_seg.append(resample_pcd(comp_sampled[comp_color == i], 512))
        """
        return model_id, part_sampled, comp_sampled, part_seg, comp_seg

    def __len__(self):
        return self.len
