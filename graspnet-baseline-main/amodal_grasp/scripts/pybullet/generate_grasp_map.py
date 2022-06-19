import os
import numpy as np
from sklearn.neighbors import KDTree
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from glob import glob
import trimesh

def remove_zero_point_in_pc(pc):
    nonzero_mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[nonzero_mask], nonzero_mask


def generate_grasp_map(seg, nocs, grasp_dict,
                       quality_th=0.03, label_map=None,
                       ):
    h, w = seg.shape[:2]
    labels = np.unique(seg)[1:]
    grasp_map = np.zeros((h * w, 8), dtype='float32')
    # grasp_map = np.zeros((h * w, 7), dtype='float32')

    for i in labels:
        obj_index = np.where((seg == i).flatten() == True)[0]
        obj_nocs_in_scene = nocs.reshape(-1, 3)[obj_index]
        obj_name = label_map[i]
        obj_nocs = grasp_dict[obj_name][:, 4:7]

        knn = KDTree(obj_nocs)
        distance, knn_idx = knn.query(obj_nocs_in_scene, return_distance=True)
        quat_width_depth_tolerance = grasp_dict[obj_name][:, [0, 1, 2, 3, 7, 8, 10]]
        quat_width_depth_tolerance = quat_width_depth_tolerance[knn_idx.flatten()]

        quality = (distance < quality_th)
        # quat_width_depth_tolerance[quality.flatten()] == 0
        qual_quat_width_depth_tolerance = np.hstack((quality.astype('int32'), quat_width_depth_tolerance))
        grasp_map[obj_index] = qual_quat_width_depth_tolerance * quality

        # distance_quat_width_depth_tolerance = np.hstack((distance, quat_width_depth_tolerance))
        # grasp_map[obj_index] = distance_quat_width_depth_tolerance
        
        # grasp_map[obj_index] = quat_width_depth_tolerance
    # grasp_map = grasp_map.reshape(h, w, 7)
    grasp_map = grasp_map.reshape(h, w, 8)
    return grasp_map


if __name__ == '__main__':
    data_root = '/disk2/data/graspnet/amodal_grasp/pybullet'
    max_width = 0.08
    distance_th = 0.01
    obj_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
               12, 13, 14, 15, 34, 37, 43,
               46, 58, 61, 65,
               ]

    nocs_para_path = os.path.join(data_root, f'nocs_para.json')
    with open(nocs_para_path, 'r') as f:
        nocs_para_dict = json.load(f)

    os.makedirs(f'{data_root}/grasp_nocs', exist_ok=True)
    grasp_dict = {}
    for i in tqdm(np.int32(obj_ids), desc='Filtering grasp ...'):
        # if i == 43:
        #     print(1)
        obj_name = str(i).zfill(3)
        grasp = np.load(os.path.join(data_root, f'grasp_label/{obj_name}.npz'))['grasps']
        width = grasp[:, 7]
        grasp = grasp[width <= max_width]
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
        trimesh.PointCloud(obj_nocs).export(f'{data_root}/grasp_nocs/{obj_name}.ply')



    scene_root = join(data_root, 'scenes')
    anns_root = join(data_root, 'mesh_pose_list')
    rgbs_root = join(data_root, 'scenes/rgb')
    segs_root = join(data_root, 'scenes/seg')
    depths_root = join(data_root, 'scenes/depth')
    extrs_root = join(data_root, 'scenes/extrinsic')
    nocs_root = join(data_root, 'scenes/nocs')
    anns_new_root = join(data_root, 'scenes/ann')

    rgbs_path = sorted(glob(rgbs_root + '/*.jpg'))
    segs_path = sorted(glob(segs_root + '/*.png'))
    depths_path = sorted(glob(depths_root + '/*.png'))
    extrs_path = sorted(glob(extrs_root + '/*.txt'))
    nocs_path = sorted(glob(nocs_root + '/*.jpg'))
    anns_new_path = sorted(glob(anns_new_root + '/*.json'))

    save_dir = os.path.join(data_root, f'scenes/grasp')
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(rgbs_path)), desc='Generating grasp map...'):
        scene_name = f'scene_{str(i).zfill(4)}'
        filename = rgbs_path[i].split('/')[-1][:-4]

        with open(anns_new_path[i], 'r') as f:
            ann = json.load(f)
        nocs = cv2.imread(nocs_path[i], cv2.IMREAD_UNCHANGED)
        nocs = np.float32(nocs / 255)
        seg = cv2.imread(segs_path[i], cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgbs_path[i], cv2.IMREAD_UNCHANGED)
        label_map = {int(k): v['mesh_path'].split('/')[-2] for k, v in ann.items()}
        grasp_map = generate_grasp_map(seg, nocs, grasp_dict, distance_th, label_map)
        grasp_map = np.float16(grasp_map)
        np.savez_compressed(os.path.join(save_dir, f'{filename}.npz'),
                            grasp_map=grasp_map)