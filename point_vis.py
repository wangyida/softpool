# examples/Python/Basic/pointcloud.py

import argparse
import csv
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
    parser.add_argument(
        '-l',
        action="store",
        dest="files_list",
        default="",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    if results.file_path != '':
        print("Load a ply point cloud, print it, and render it")
        pcd = o3d.io.read_point_cloud(results.file_path)
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd])

        print("Recompute the normal of the downsampled point cloud")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=100))
        o3d.visualization.draw_geometries([pcd])
    elif results.files_list != '':
        # Get the number of validation samples
        points_all = np.zeros((1,3))
        colors_all = np.zeros((1,3)) 
        with open(results.files_list, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            n_files_valid = len(data)
            for i in range(n_files_valid):
                print("Load a ply point cloud, print it, and render it")
                pcd = o3d.io.read_point_cloud(data[i][0])
                npy_points = np.asarray(pcd.points)
                npy_points[:, 0] += (i*1.0002)
                npy_colors = np.asarray(pcd.colors)
                points_all = np.concatenate((points_all, npy_points), 0)
                colors_all = np.concatenate((colors_all, npy_colors), 0)
                print(pcd)
                print(np.asarray(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points_all[:,0:3])
    pcd.colors = o3d.utility.Vector3dVector(colors_all[:,0:3])
    o3d.visualization.draw_geometries([pcd])

    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=100))
    o3d.visualization.draw_geometries([pcd])
