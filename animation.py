import sys
import numpy as np
import open3d as py3d

pcd1 = py3d.io.read_point_cloud(
    "/Users/yidawang/Downloads/cvpr20/shapenet_reg_11/gt/02958343/4e3a0f69a63ee72426d1f3c517cdece5.ply"
)
pcd2 = py3d.io.read_point_cloud(
    "/Users/yidawang/Downloads/cvpr20/shapenet_reg_11/output2/02958343/4e3a0f69a63ee72426d1f3c517cdece5.ply"
)

# pcd1.paint_uniform_color([1, 0, 0])
# pcd2.paint_uniform_color([0, 0, 1])

kdt = py3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
pcd1.estimate_normals(search_param=kdt)
points = np.array(pcd1.points)
points[:, 0] += 1
pcd1.points = py3d.utility.Vector3dVector(points[:, 0:3])
pcd2.estimate_normals(search_param=kdt)

th = 0.02
criteria = py3d.registration.ICPConvergenceCriteria(
    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1)
est_method = py3d.registration.TransformationEstimationPointToPoint()


def one_step_ICP(vis):
    vis.get_view_control().rotate(x=5.0, y=5.0)
    info = py3d.registration.registration_icp(
        pcd1,
        pcd2,
        max_correspondence_distance=th,
        estimation_method=est_method,
        criteria=criteria)
    print("fitness {0:.6f} RMSE {1:.6f}".format(info.fitness,
                                                info.inlier_rmse))
    pcd1.transform(info.transformation)
    return True


py3d.visualization.draw_geometries_with_animation_callback(
    [pcd1, pcd2], one_step_ICP, "iterations", 640, 480)
