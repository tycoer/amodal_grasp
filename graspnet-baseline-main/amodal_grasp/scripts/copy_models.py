import shutil
import os
from tqdm import tqdm
if __name__ == '__main__':

    data_root = '/disk2/data/graspnet'
    save_dir = os.path.join(data_root, 'amodal_grasp/models')

    os.makedirs(save_dir, exist_ok=True)
    models_root = os.path.join(data_root, 'models')
    for i in tqdm(os.listdir(models_root)):
        if len(i) != 3:
            continue
        model_src_path = os.path.join(models_root, f'{i}/nontextured_simplified.ply')
        model_dst_path = os.path.join(save_dir, f'{i}.ply')
        shutil.copy(model_src_path, model_dst_path)