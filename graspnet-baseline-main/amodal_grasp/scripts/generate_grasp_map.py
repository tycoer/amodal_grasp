import os
import numpy as np
from sklearn.neighbors import KDTree
from graspnetAPI import GraspNet
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def remove_zero_point_in_pc(pc):
    nonzero_mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[nonzero_mask], nonzero_mask


def generate_grasp_map(seg, nocs, grasp_dict,
                       quality_th=0.03,
                       ):
    h, w = seg.shape[:2]
    labels = np.unique(seg)[1:]
    grasp_map = np.zeros((h * w, 8), dtype='float32')
    # grasp_map = np.zeros((h * w, 7), dtype='float32')

    for i in labels:
        obj_index = np.where((seg == i).flatten() == True)[0]
        obj_nocs_in_scene = nocs.reshape(-1, 3)[obj_index]
        obj_name = str(i - 1).zfill(3)
        obj_nocs = grasp_dict[obj_name][:, 4:7]

        knn = KDTree(obj_nocs)
        distance, knn_idx = knn.query(obj_nocs_in_scene, return_distance=True)
        quat_width_depth_tolerance = grasp_dict[obj_name][:, [0, 1, 2, 3, 7, 8, 10]]
        quat_width_depth_tolerance = quat_width_depth_tolerance[knn_idx.flatten()]

        # quality = (distance < quality_th)
        # quat_width_depth_tolerance[quality.flatten()] == 0
        # qual_quat_width_depth_tolerance = np.hstack((quality.astype('int32'), quat_width_depth_tolerance))
        # grasp_map[obj_index] = qual_quat_width_depth_tolerance

        distance_quat_width_depth_tolerance = np.hstack((distance, quat_width_depth_tolerance))
        grasp_map[obj_index] = distance_quat_width_depth_tolerance
        
        # grasp_map[obj_index] = quat_width_depth_tolerance
    # grasp_map = grasp_map.reshape(h, w, 7)
    grasp_map = grasp_map.reshape(h, w, 8)
    return grasp_map


if __name__ == '__main__':
    data_root = '/disk2/data/graspnet'
    camera = 'kinect'
    split = 'train'

    g = GraspNet(data_root, camera, split)
    nocs_para_path = os.path.join(data_root, f'amodal_grasp/nocs_para.json')
    with open(nocs_para_path, 'r') as f:
        nocs_para_dict = json.load(f)

    grasp_dict = {}
    for i in tqdm(np.int32(g.objIds), desc='Filtering grasp ...'):
        obj_name = str(i).zfill(3)
        grasp = np.load(os.path.join(data_root, f'amodal_grasp/grasp_label/{obj_name}.npz'))['grasps']
        point_id = grasp[:, -2]
        point_id, index = np.unique(point_id, return_index=True)
        grasp = grasp[index]
        obj_xyz = grasp[:, 4:7]
        nocs_para = nocs_para_dict[obj_name]
        norm_factor, norm_corner = nocs_para['norm_factor'], np.array(nocs_para['norm_corner'])
        obj_nocs = (obj_xyz - norm_corner[0]) * norm_factor + np.array((0.5, 0.5, 0.5)).reshape(1, 3) - 0.5 * (
                norm_corner[1] - norm_corner[0]) * norm_factor
        grasp[:, 4:7] = obj_nocs
        grasp_dict[obj_name] =  grasp

    for i in g.sceneIds:
        scene_name = f'scene_{str(i).zfill(4)}'
        save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/grasp_map')
        os.makedirs(save_dir, exist_ok=True)
        for j in tqdm(range(256), desc=f'Processing {scene_name} ...'):
            filename = str(j).zfill(4)
            seg_path = os.path.join(data_root, f'scenes/{scene_name}/{camera}/label/{filename}.png')
            box_path = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/2d_box_xyxy/{filename}.json')
            nocs_path = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/nocs/{filename}.png')
            nocs = cv2.imread(nocs_path, cv2.IMREAD_UNCHANGED)
            nocs = np.float32(nocs / 255)
            seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            grasp_map = generate_grasp_map(seg, nocs, grasp_dict)
            grasp_map = np.float16(grasp_map)
            np.savez_compressed(os.path.join(save_dir, f'{filename}.npz'),
                                grasp_map=grasp_map)