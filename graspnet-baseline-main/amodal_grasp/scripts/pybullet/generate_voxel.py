import os
import trimesh
import numpy as np
from tqdm import tqdm
data_root = '/disk2/data/graspnet/amodal_grasp/pybullet'
voxel_size = 24

models_root = os.path.join(data_root, 'models_collision')
for i in tqdm(os.listdir(models_root)):
    model_path = os.path.join(models_root, i)
    os.system(f'../binvox/binvox -d {voxel_size} -e {model_path}')