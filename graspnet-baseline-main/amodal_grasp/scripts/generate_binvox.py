import os
import trimesh
import numpy as np
from tqdm import tqdm
data_root = '/disk5/data/graspnet'
voxel_size = 24

models_root = os.path.join(data_root, 'amodal_grasp/models')
for i in tqdm(os.listdir(models_root)):
    model_path = os.path.join(models_root, f'{i}/nontextured.ply')
    os.system(f'../binvox/binvox -d {voxel_size} -e {model_path}')