import numpy as np
import trimesh
import os
from tqdm import tqdm
import json

def get_nocs_para(points):

    # points 必须是完整物体（不能是partical的！！！）的点云或mesh的顶点
    # 这个 points 可以是 transform过的
    # points.shape : (n, 3)
    tight_w = max(points[:, 0]) - min(points[:, 0])
    tight_l = max(points[:, 1]) - min(points[:, 1])
    tight_h = max(points[:, 2]) - min(points[:, 2])

    # corner_pts[i+1] = np.amin(part_gts, axis=1)
    norm_factor = np.sqrt(1) / np.sqrt(tight_w ** 2 + tight_l ** 2 + tight_h ** 2)
    norm_factor = norm_factor.tolist() # scale
    corner_pt_left = np.amin(points, axis=0, keepdims=False).tolist()
    corner_pt_right = np.amax(points, axis=0, keepdims=False).tolist()
    # norm_corner = np.array([corner_pt_left, corner_pt_right])
    norm_corner = [corner_pt_left, corner_pt_right]
    return norm_factor, norm_corner

if __name__ == '__main__':
    data_root = '/disk2/data/graspnet'
    models_root = os.path.join(data_root, 'models')
    nocs_para = {}
    for i in tqdm(sorted(range(88))):
        dir_name = str(i).zfill(3)
        model_path = os.path.join(models_root, f'{dir_name}/nontextured.ply')
        mesh = trimesh.load(model_path)
        norm_factor, norm_corner = get_nocs_para(mesh.vertices)
        nocs_para[dir_name] = dict(norm_corner=norm_corner, norm_factor=norm_factor)
    save_path = os.path.join(data_root, 'amodal_grasp/nocs_para.json')
    with open(save_path, 'w') as f:
        json.dump(nocs_para, f, indent=2)