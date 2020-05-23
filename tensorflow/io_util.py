# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
from open3d import *
import matplotlib


def read_pcd(filename):
    set_verbosity_level(VerbosityLevel.Error)
    pcd = read_point_cloud(filename)
    if pcd.colors:
        return np.concatenate([np.array(pcd.points), np.array(pcd.colors)], 1)
    else:
        colors = matplotlib.cm.cool(np.array(pcd.points)[:, 0])
        return np.concatenate([np.array(pcd.points), colors[:, 0:3]], 1)


def save_pcd(filename, points):
    set_verbosity_level(VerbosityLevel.Error)
    pcd = PointCloud()
    pcd.points = Vector3dVector(points[:, 0:3])
    pcd.colors = Vector3dVector(points[:, 3:6])
    write_point_cloud(filename, pcd)
