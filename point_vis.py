# examples/Python/Basic/pointcloud.py

import argparse
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-f',
        action="store",
        dest="file_path",
        default="",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(results.file_path)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=100))
    o3d.visualization.draw_geometries([pcd])
