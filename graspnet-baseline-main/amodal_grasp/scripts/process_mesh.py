import trimesh
import os
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    data_root = '/disk5/data/graspnet'
    models_root = os.path.join(data_root, 'models')
    meshes = {}
    for i in tqdm(os.listdir(models_root)):
        model_path = os.path.join(models_root, f'{i}/nontextured.ply')
        mesh = trimesh.load(model_path)
        meshes[i] = (np.array(mesh.vertices, dtype='float32'), np.array(mesh.faces, dtype='int32'))
    np.savez_compressed(
        os.path.join(data_root, 'models.npz'),
        **meshes)
