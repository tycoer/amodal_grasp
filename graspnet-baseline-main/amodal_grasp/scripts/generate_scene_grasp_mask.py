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

def load_obj_poses(obj_poses_path):
    xml = xmlReader(obj_poses_path)
    obj_poses = xml.get_pose_list()
    obj_poses_dict = {}
    for pose in obj_poses:
        obj_poses_dict[str(pose.id).zfill(3)] = pose.mat_4x4
    return obj_poses_dict



if __name__ == '__main__':
    # cfg
    data_root = '/disk5/data/graspnet'
    camera = 'kinect'
    split = 'train'
    # collision_cfg
    process_collision = False
    fric_coef_thresh = 0.4
    # grasp_cfg
    invisible_distance_th = 0.01
    process_grasp_in_scene = True
    project_grasp_point_in_scene = True

    assert not (process_collision == False and process_grasp_in_scene == False)
    model_grasp_root = os.path.join(data_root, 'amodal_grasp/grasp_label')
    g = GraspNet(data_root, camera=camera, split=split)
    scene_ids = g.sceneIds
    grasp_labels_processed = {i: pd.read_csv(os.path.join(model_grasp_root, f'{str(i).zfill(3)}_labels.csv')) for i in tqdm(g.objIds)}

    for scene_id in scene_ids:
        # loading
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        obj_ids_path = os.path.join(data_root, f'scenes/{scene_name}/object_id_list.txt')
        obj_ids = np.loadtxt(obj_ids_path)
        if process_collision:
            collisions_processed = {}
            collisions = g.loadCollisionLabels(scene_id)[scene_name]
            collisions_save_dir = os.path.join(data_root, f'amodal_grasp/collision_label/{scene_name}')
            os.makedirs(collisions_save_dir, exist_ok=True)
            for i, obj_id in enumerate(tqdm(obj_ids, desc=f'Processing collision in scene_{str(scene_id).zfill(4)}')):
                index = np.int32(grasp_labels_processed[obj_id]['index'])
                collision = collisions[i].flatten()
                collision = collision[index]
                collisions_processed[str(int(obj_id)).zfill(3)] = collision
            collision_save_path = os.path.join(collisions_save_dir, 'collision_labels.npz')
            np.savez_compressed(collision_save_path,
                                **collisions_processed)



        if process_grasp_in_scene:
            pc_root = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/pc')
            seg_root =  os.path.join(data_root, f'scenes/{scene_name}/{camera}/label')
            ann_root = os.path.join(data_root, f'scenes/{scene_name}/{camera}/annotations')
            grasp_save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/grasp')
            collision_path = os.path.join(data_root, f'amodal_grasp/collision_label/{scene_name}/collision_labels.npz')
            collision = np.load(collision_path)
            os.makedirs(grasp_save_dir, exist_ok=True)
            for ann_id in tqdm(range(256), desc=f'Processing grasp in scene_{str(scene_id).zfill(4)}'):
                filename = str(ann_id).zfill(4)
                pc_path = os.path.join(pc_root, f'{filename}.npz')
                seg_path = os.path.join(seg_root, f'{filename}.png')
                ann_path = os.path.join(ann_root, f'{filename}.xml')
                pc = np.load(pc_path)['pc']
                seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                seg_unique = np.unique(seg)
                obj_poses_dict = load_obj_poses(ann_path)

                grasp_processed_in_scene = np.zeros((0, 17))
                for i, obj_id in enumerate((obj_ids)):
                    if obj_id + 1 not in seg_unique:
                        continue
                    obj_name = str(int(obj_id)).zfill(3)
                    obj_pose = obj_poses_dict[obj_name]
                    pc_obj_in_scene = pc[seg == int(obj_id + 1)].reshape(-1, 3) # 物体在scene中的 pc
                    collision_mask = collision[obj_name]
                    grasp = grasp_labels_processed[obj_id][~collision_mask]
                    grasp_xyz = grasp.loc[:, ['x', 'y','z']]
                    grasp_quat = grasp.loc[:, ['qx', 'qy', 'qz', 'qw']]
                    grasp_R = Rotation.from_quat(grasp_quat).as_matrix()
                    H = (np.eye(4)[None, :, :]).repeat(len(grasp_xyz), 0)
                    H[:, :3, :3] =  grasp_R
                    H[:, :3, 3] = grasp_xyz

                    H_in_scene = obj_pose.dot(H).transpose(1, 0, 2)
                    grasp_quat_in_scene = Rotation.from_matrix(H_in_scene[:, :3, :3]).as_quat()
                    grasp_xyz_in_scene = H_in_scene[:, :3, 3]
                    knn = KDTree(pc_obj_in_scene)
                    dists, index = knn.query(grasp_xyz_in_scene, return_distance=True)
                    visible_mask = (dists < invisible_distance_th)

                    grasp_copy = grasp.copy().to_numpy()
                    grasp_xyz_in_scene_project = pc_obj_in_scene[index.flatten()]
                    obj_id = np.array(len(grasp_copy) * [obj_id])
                    visible = visible_mask.astype('int32').flatten()
                    grasp_copy = np.hstack((grasp_quat_in_scene, grasp_xyz_in_scene, grasp_xyz_in_scene_project, grasp_copy[:, 7: 12], obj_id[:, None], visible[:, None]))
                    grasp_processed_in_scene = np.vstack((grasp_processed_in_scene, grasp_copy))

                grasp_processed_in_scene = np.float16(grasp_processed_in_scene)
                grasp_processed_in_scene = pd.DataFrame(columns=['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z','x_projected', 'y_projected',
                                                                   'z_projected', 'width', 'depth', 'score',
                                                                   'tolerance', 'point_id', 'obj_id', 'visible'],
                                                        data=grasp_processed_in_scene)
                grasp_processed_in_scene.to_csv(os.path.join(grasp_save_dir, f'{str(int(ann_id)).zfill(4)}.csv'),
                                                index=False)


