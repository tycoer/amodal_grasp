import os
import trimesh
import numpy as np
from tqdm import tqdm
data_root = '/disk5/data/graspnet'

models_root = os.path.join(data_root, 'amodal_grasp/models')
for i in tqdm(sorted(os.listdir(models_root))):
    binvox_path = os.path.join(models_root, f'{i}/nontextured.binvox')
    binvox = trimesh.load(binvox_path)
    mesh = binvox.marching_cubes
    mesh_save_path = os.path.join(models_root, f'{i}/nontextured_simplified.ply')
    mesh : trimesh.Trimesh
    mesh.export(mesh_save_path)
