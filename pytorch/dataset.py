import open3d as o3d
import torch
import h5py
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random

#from utils import *


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]], idx[:n]


class ShapeNet(data.Dataset):
    def __init__(self, train=True, npoints=8192):
        self.dataset = 'suncg'
        if train:
            if self.dataset == 'suncg':
                self.list_path = './data/train_suncg.list'
            elif self.dataset == 'shapenet':
                self.list_path = './data/train_shapenet.list'
        else:
            if self.dataset == 'suncg':
                self.list_path = './data/valid_suncg.list'
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

        def read_pcd(filename, d_format='pcd'):
            # if d_format == 'pcd':
            if self.dataset == 'suncg':
                pcd = o3d.read_point_cloud(filename)
                coord = torch.from_numpy(np.array(pcd.points)).float()
                color = torch.from_numpy(np.array(pcd.colors)).float()
                return coord, color
            # elif d_format == 'h5':
            elif self.dataset == 'shapenet':
                fh5 = h5py.File(filename, 'r')
                coord = torch.from_numpy(np.array(fh5['data'])).float()
                color = torch.from_numpy(np.array(fh5['data'])).float()
                return coord, color

        if self.train:
            if self.dataset == 'suncg':
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_complete/",
                        '%s.pcd' % model_id))
                """
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_partial/",
                        '%s.pcd' % model_id))
                """
                comp, comp_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_complete/",
                        '%s.pcd' % model_id))
            elif self.dataset == 'shapenet':
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/train/partial/",
                        '%s.h5' % model_id))
                comp, comp_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/train/gt/",
                        '%s.h5' % model_id))
                """
                part, _ = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/gt/",
                        '%s.h5' % model_id))
                comp, comp_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/gt/",
                        '%s.h5' % model_id))
                """
        else:
            if self.dataset == 'suncg':
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/",
                        '%s.pcd' % model_id))
                """
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial/",
                        '%s.pcd' % model_id))
                """
                comp, comp_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/",
                        '%s.pcd' % model_id))
            elif self.dataset == 'shapenet':
                part, part_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/partial/",
                        '%s.h5' % model_id))
                """
                part, _ = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/gt/",
                        '%s.h5' % model_id))
                """
                comp, comp_color = read_pcd(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/gt/",
                        '%s.h5' % model_id))
        part_sampled, idx_sampled = resample_pcd(part, self.npoints // 2)
        part_seg = np.round(part_color[idx_sampled] * 11)
        comp_sampled, idx_sampled = resample_pcd(comp, self.npoints)
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
