# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("/Users/yidawang/Downloads/pcds/00000000_e5c91fa85ebb60073d89c6ed56e66593_fl001_rm0001_0000.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
