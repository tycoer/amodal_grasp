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


if __name__ == '__main__':
    # cfg
    data_root = '/disk5/data/graspnet'
    camera = 'kinect'
    split = 'train'

    model_grasp_root = os.path.join(data_root, 'amodal_grasp/grasp_label')
    g = GraspNet(data_root, camera=camera, split=split)
    scene_ids = g.sceneIds
    grasp_labels_processed = {i: np.load(os.path.join(model_grasp_root, f'{str(i).zfill(3)}.npz')) for i in tqdm(g.objIds)}

    for scene_id in scene_ids:
        # loading
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        obj_ids_path = os.path.join(data_root, f'scenes/{scene_name}/object_id_list.txt')
        obj_ids = np.loadtxt(obj_ids_path)

        collisions_processed = {}
        collisions = g.loadCollisionLabels(scene_id)[scene_name]
        collisions_save_dir = os.path.join(data_root, f'amodal_grasp/collision_label')
        os.makedirs(collisions_save_dir, exist_ok=True)
        for i, obj_id in enumerate(tqdm(obj_ids, desc=f'Processing collision in scene_{str(scene_id).zfill(4)}')):
            index = np.int32(grasp_labels_processed[obj_id]['grasps'][:, -1])
            collision = collisions[i].flatten()
            collision = collision[index]
            collisions_processed[str(int(obj_id)).zfill(3)] = ~collision
        collision_save_path = os.path.join(collisions_save_dir, f'{scene_name}.npz')
        np.savez_compressed(collision_save_path,
                            **collisions_processed)
