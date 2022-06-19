import trimesh
import open3d as o3d
import json
import numpy as np
import os
if __name__ == '__main__':
    data_root = '/disk1/data/giga/data_packed_train_raw'
    scene_cam_root = os.path.join(data_root, 'scenes_cam')
    grasp_cam_by_scene_path = os.path.join(data_root, 'grasps_cam_by_scene.json')
    with open(grasp_cam_by_scene_path, 'r') as f:
        grasp_cam_by_scene = json.load(f)

    keys = list(grasp_cam_by_scene.keys())
    key = keys[0]
    grasp = grasp_cam_by_scene[key]
    pc = o3d.io.read_point_cloud(os.path.join(scene_cam_root, key + '.ply'))



