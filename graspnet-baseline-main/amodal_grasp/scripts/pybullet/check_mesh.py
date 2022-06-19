import os
import trimesh
import numpy as np
from tqdm import tqdm
data_root = '/Disk2/hanyang/pybullet'
voxel_size = 24

models_root = os.path.join(data_root, 'models_collision')
max_num_verts = []
for i in tqdm(os.listdir(models_root)):
    model_path = os.path.join(models_root, i)
    if model_path.endswith('.obj'):
        mesh = trimesh.load(model_path)
        verts = mesh.vertices
        print(model_path, verts.max(), (np.inf in verts) or (np.nan in verts))
        max_num_verts.append(len(verts))

print(max(max_num_verts))