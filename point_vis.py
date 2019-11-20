# examples/Python/Basic/pointcloud.py

import argparse
import csv
import trimesh
import numpy as np
import open3d as o3d
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-n',
        action="store",
        type=int,
        dest="numbers",
        default=5,
        help='Number of samples')
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
    parser.add_argument(
        '-l2',
        action="store",
        dest="files_list2",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l3',
        action="store",
        dest="files_list3",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l4',
        action="store",
        dest="files_list4",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l5',
        action="store",
        dest="files_list5",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l6',
        action="store",
        dest="files_list6",
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
            radius=0.05, max_nn=64))
        o3d.visualization.draw_geometries([pcd])

        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.0)
        o3d.visualization.draw_geometries([cl])
        display_inlier_outlier(pcd, ind)

        """
        print("Radius oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=0.5)
        o3d.visualization.draw_geometries([cl])
        display_inlier_outlier(pcd, ind)

        print("Downsample the point cloud with a voxel of 0.02")
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        o3d.visualization.draw_geometries([pcd])

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=64))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        o3d.visualization.draw_geometries([mesh])
        """
    elif results.files_list != '':
        # Get the number of validation samples
        with open(results.files_list, "r") as f:
            points_all = np.zeros((1,3))
            colors_all = np.zeros((1,3)) 
            reader = csv.reader(f)
            data = list(reader)
            for i in range(results.numbers):
                pcd = o3d.io.read_point_cloud(data[i][0])
                npy_points = np.asarray(pcd.points)
                npy_points[:, 0] += (i*1.1002)
                npy_colors = np.asarray(pcd.colors)
                points_all = np.concatenate((points_all, npy_points), 0)
                colors_all = np.concatenate((colors_all, npy_colors), 0)
        if results.files_list2 != '':
            with open(results.files_list2, "r") as f:
                points_all2 = np.zeros((1,3))
                colors_all2 = np.zeros((1,3)) 
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i*1.1002)
                    npy_points[:, 2] += 1.0002
                    # npy_points[:, 1] += .4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all2 = np.concatenate((points_all2, npy_points), 0)
                    colors_all2 = np.concatenate((colors_all2, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all2), 0)
                    colors_all = np.concatenate((colors_all, colors_all2), 0)
        if results.files_list3 != '':
            with open(results.files_list3, "r") as f:
                points_all3 = np.zeros((1,3))
                colors_all3 = np.zeros((1,3)) 
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i*1.1002)
                    npy_points[:, 2] += 2*1.0002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all3 = np.concatenate((points_all3, npy_points), 0)
                    colors_all3 = np.concatenate((colors_all3, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all3), 0)
                    colors_all = np.concatenate((colors_all, colors_all3), 0)
        if results.files_list4 != '':
            with open(results.files_list4, "r") as f:
                points_all4 = np.zeros((1,3))
                colors_all4 = np.zeros((1,3)) 
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i*1.1002)
                    npy_points[:, 2] += 3*1.0002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all4 = np.concatenate((points_all4, npy_points), 0)
                    colors_all4 = np.concatenate((colors_all4, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all4), 0)
                    colors_all = np.concatenate((colors_all, colors_all4), 0)
        if results.files_list5 != '':
            with open(results.files_list5, "r") as f:
                points_all5 = np.zeros((1,3))
                colors_all5 = np.zeros((1,3)) 
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i*1.1002)
                    npy_points[:, 2] += 4*1.0002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all5 = np.concatenate((points_all5, npy_points), 0)
                    colors_all5 = np.concatenate((colors_all5, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all5), 0)
                    colors_all = np.concatenate((colors_all, colors_all5), 0)
        if results.files_list6 != '':
            with open(results.files_list6, "r") as f:
                points_all6 = np.zeros((1,3))
                colors_all6 = np.zeros((1,3)) 
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i*1.1002)
                    npy_points[:, 2] += 5*1.0002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all6 = np.concatenate((points_all6, npy_points), 0)
                    colors_all6 = np.concatenate((colors_all6, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all6), 0)
                    colors_all = np.concatenate((colors_all, colors_all6), 0)
        pcd.points = o3d.utility.Vector3dVector(points_all[:,0:3])
        pcd.colors = o3d.utility.Vector3dVector(colors_all[:,0:3])
        # o3d.visualization.draw_geometries([pcd])

        print("Recompute the normal of the downsampled point cloud")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=32))

        o3d.visualization.draw_geometries([pcd])

        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.0)
        o3d.visualization.draw_geometries([cl])
        display_inlier_outlier(pcd, ind)

