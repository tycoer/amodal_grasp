import numpy as np

grasp_path = '/disk1/data/amodal_grasp/packed_raw/grasp_on_obj.npy'

if __name__ == '__main__':
    grasp = np.load(grasp_path, allow_pickle=True)
    grasp_scene = [{list(i.keys()):{}} for i in grasp]
