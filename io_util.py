# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from open3d import *
import matplotlib


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.concatenate([np.array(pcd.points), np.array(pcd.colors)], 1)


def save_pcd(filename, points):
    pcd = PointCloud()
    color = matplotlib.cm.Set3((np.argmax(points[:, 3:], -1) + 1)/11 - 0.5/11)
    pcd.points = Vector3dVector(points[:,0:3])
    pcd.colors = Vector3dVector(color[:,0:3])
    write_point_cloud(filename, pcd)
