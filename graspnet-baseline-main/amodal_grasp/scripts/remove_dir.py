import shutil
import os
from graspnetAPI import GraspNet
from tqdm import tqdm
if __name__ == '__main__':
    data_root = '/disk5/data/graspnet'
    camera = 'kinect'
    g = GraspNet('/disk5/data/graspnet')
    for i in g.sceneIds:
        for j in tqdm(range(256), desc=f'i'):
            grasp_root = os.path.join(data_root, f'amodal_grasp/scenes/scene_{str(j).zfill(4)}/{camera}/grasp')
            grasp_filter_root = os.path.join(data_root, f'amodal_grasp/scenes/scene_{str(j).zfill(4)}/{camera}/grasp_filtered')
            grasp_filter_root = os.path.join(data_root, f'amodal_grasp/scenes/scene_{str(j).zfill(4)}/{camera}/fric_mask')
            grasp_filter_root = os.path.join(data_root, f'amodal_grasp/scenes/scene_{str(j).zfill(4)}/{camera}/pc_normalized')
            # grasp_filter_root = os.path.join(data_root, f'amodal_grasp/scenes/scene_{str(j).zfill(4)}/{camera}/nocs_normalized')


            shutil.rmtree(grasp_root, ignore_errors=True)
            shutil.rmtree(grasp_filter_root, ignore_errors=True)
