import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

def depth2pc(depth, fx, fy, cx, cy, w, h, depth_scale=1, ):
    h_grid, w_grid= np.mgrid[0: h, 0: w]
    z = depth / depth_scale
    x = (w_grid - cx) * z / fx
    y = (h_grid - cy) * z / fy
    xyz = np.dstack((x, y, z)).reshape(-1, 3)
    return xyz


def get_extrinsic(extr):
    quat = extr[0, :4]
    t = extr[0, 4:]
    H = np.eye(4)
    H[:3, :3] = Rotation.from_quat(quat).as_matrix()
    H[:3, 3] = t
    return H

if __name__ == '__main__':
    # anno_path = '/disk3/data/amodal_grasp/packed_raw/mesh_pose_list/f10ffb3093864eaa950b62e8de95aadc.npy'
    # image_path = '/disk3/data/amodal_grasp/packed_raw/scenes/f10ffb3093864eaa950b62e8de95aadc.npz'

    anno_path = 'data_test/packed_raw/mesh_pose_list/f10ffb3093864eaa950b62e8de95aadc.npy'
    image_path = 'data_test/packed_raw/scenes/f10ffb3093864eaa950b62e8de95aadc.npz'
    anno = list(np.load(anno_path, allow_pickle=True))
    image = dict(np.load(image_path))
    rgb = image['rgb_imgs']
    depth = image['depth_imgs'][0]
    mask = image['mask_imgs']

    xyz = depth2pc(depth, fx=540, fy=540, cx=320, cy=240, w=640, h=480, depth_scale=1)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,
                                                             origin=[0, 0, 0])

    ex = get_extrinsic(image['extrinsics'])
    ex = np.linalg.inv(ex)

    # xyz = np.hstack((xyz, np.zeros((xyz.shape[0], 1))))
    # T = ex[0, 4:]
    #
    # R = Rotation.from_quat(ex[0, :4]).as_matrix()
    R = ex[:3, :3]
    T = ex[:3, 3]
    xyz = (R @ xyz.T).T + T

    # xyz = xyz @ ex
    # xyz = xyz[:, :3]




    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    o3d.visualization.draw_geometries([pc, axis])

    # ex = get_extrinsic(image['extrinsics'])
    # ex = np.linalg.inv(ex)
    H = anno[1]['pose']
    # H[:, 3] = np.array([0, 0, 0, 1])
    # H_inv = np.linalg.inv(H)
    # H_inv[:3, 3]*=0.01
    axis.transform(H)
    # pc.transform(ex)
    o3d.visualization.draw_geometries([pc, axis])
