import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from graspnetAPI import GraspNet
from tqdm import tqdm
import pandas as pd
import cv2
from graspnetAPI.utils.utils import xmlReader
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial.transform import Rotation
import time
from sklearn.neighbors import KDTree
import json


def load_obj_poses(obj_poses_path):
    xml = xmlReader(obj_poses_path)
    obj_poses = xml.get_pose_list()
    obj_poses_dict = {}
    for pose in obj_poses:
        obj_poses_dict[str(pose.id).zfill(3)] = pose.mat_4x4
    return obj_poses_dict



def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def index2uv(w, h, index):
    v = index // w
    u = index % w
    return np.hstack((u[:, None], v[:, None]))

if __name__ == '__main__':
    # cfg
    data_root = '/disk5/data/graspnet'
    camera = 'kinect'
    split = 'train'

    # grasp_cfg
    invisible_distance_th = 0.01

    model_grasp_root = os.path.join(data_root, 'amodal_grasp/grasp_label')
    g = GraspNet(data_root, camera=camera, split=split)
    scene_ids = g.sceneIds
    grasp_dict = {str(i).zfill(3): np.load(os.path.join(model_grasp_root, f'{str(i).zfill(3)}.npz')) for i in tqdm(g.objIds)}

    column_names = {0: 'qx',
                    1: 'qy',
                    2: 'qz',
                    3: 'qw',
                    4: 'x',
                    5: 'y',
                    6: 'z',
                    7: 'x_projected',
                    8: 'y_projected',
                    9: 'z_projected',
                    10: 'u',
                    11: 'v',
                    12: 'width',
                    13: 'depth',
                    14: 'score',
                    15: 'tolerance',
                    16: 'point_id',
                    17: 'obj_id'}
    with open(os.path.join(data_root, 'amodal_grasp/column_names_in_scenes.json'), 'w') as f:
        json.dump(column_names, f)

    for scene_id in scene_ids:
        # loading
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        obj_ids_path = os.path.join(data_root, f'scenes/{scene_name}/object_id_list.txt')
        obj_ids = np.loadtxt(obj_ids_path)
        #####################################
        collision_path = os.path.join(data_root, f'amodal_grasp/collision_label/{scene_name}.npz')
        collisions = np.load(collision_path)
        #####################################
        pc_root = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/pc')
        seg_root = os.path.join(data_root, f'scenes/{scene_name}/{camera}/label')
        ann_root = os.path.join(data_root, f'scenes/{scene_name}/{camera}/annotations')
        visible_mask_save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/grasps')
        os.makedirs(visible_mask_save_dir, exist_ok=True)
        for ann_id in tqdm(range(256), desc=f'Processing grasp in scene_{str(scene_id).zfill(4)}'):
            filename = str(ann_id).zfill(4)
            pc_path = os.path.join(pc_root, f'{filename}.npz')
            seg_path = os.path.join(seg_root, f'{filename}.png')
            ann_path = os.path.join(ann_root, f'{filename}.xml')
            pc = np.load(pc_path)['pc']
            seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            seg_unique = np.unique(seg)
            obj_poses_dict = load_obj_poses(ann_path)
            # res = {}
            res = []
            for i, obj_id in enumerate((obj_ids)):
                if obj_id + 1 not in seg_unique:
                    continue
                obj_name = str(int(obj_id)).zfill(3)
                obj_pose = obj_poses_dict[obj_name]
                pc_obj_index = (seg == int(obj_id + 1)).flatten().nonzero()[0]
                pc_obj_in_scene = pc.reshape(-1, 3)[pc_obj_index]  # 物体在scene中的 pc
                pc_obj = grasp_dict[obj_name]['points']
                pc_obj = transform_point_cloud(pc_obj, obj_pose)
                knn = KDTree(pc_obj_in_scene)
                dists, index = knn.query(pc_obj, return_distance=True)
                visible_mask = (dists < invisible_distance_th)
                visible_points_id = np.where(visible_mask == 1)[0]
                index = index.flatten()
                pc_obj_projected = pc_obj_in_scene[index]
                pc_obj_index_projected = pc_obj_index[index]
                uv = index2uv(w=1280, h=720, index=pc_obj_index_projected)

                #####################################################
                collision_mask = collisions[obj_name]
                grasp_obj = grasp_dict[obj_name]['grasps']
                grasp_obj = grasp_obj[collision_mask]
                points_id = grasp_obj[:, -2] # 物体点云的序号
                grasp_obj_mask = np.isin(points_id, visible_points_id)
                grasp_obj = grasp_obj[grasp_obj_mask]

                grasp_obj_quat = grasp_obj[:, :4]
                grasp_obj_R = Rotation.from_quat(quat=grasp_obj_quat).as_matrix()
                grasp_obj_xyz = grasp_obj[:, 4:7]
                H = (np.eye(4)[None, :, :]).repeat(len(grasp_obj), 0)
                H[:, :3, 3] = grasp_obj_xyz
                H[:, :3, :3] = grasp_obj_R
                H_in_scene = obj_pose.dot(H).transpose(1, 0, 2)
                grasp_obj_quat_in_scene = Rotation.from_matrix(H_in_scene[:, :3, :3]).as_quat()
                grasp_obj_xyz_in_scene = H_in_scene[:, :3, 3]
                points_id = grasp_obj[:, -2].astype('int32')
                pc_obj_projected = pc_obj_projected[points_id]
                uv = uv[points_id]
                obj_id = np.array([obj_id] * len(grasp_obj))[:, None]
                # column_names:
                # qx, qy, qz, qw, x, y, z, x_projected, y_projected, z_projected, u, v, width, depth, score, tolerance, point_id, obj_id
                data = np.hstack((grasp_obj_quat_in_scene, grasp_obj_xyz_in_scene, pc_obj_projected, uv, grasp_obj[:, 7: 12], obj_id))
                res.append(data)
                ################################################
                # res[obj_name] = np.hstack((pc_obj_projected, uv, visible_mask)).astype('float16')
            # np.savez_compressed(
            #     os.path.join(
            #         os.path.join(visible_mask_save_dir, f'{filename}.npz')),
            #     **res
            #     )


            res = np.vstack(res).astype('float16')
            np.savez_compressed(
                os.path.join(
                    os.path.join(visible_mask_save_dir, f'{filename}.npz')),
                grasps=res
                )