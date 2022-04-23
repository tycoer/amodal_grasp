import numpy as np
path_raw = './data_test/scenes/63263420b543490fba5e2b4e8fe826ea.npz'
path = './data_test/scenes_processed/63263420b543490fba5e2b4e8fe826ea.npz'
path_anno = './data_test/mesh_pose_list/63263420b543490fba5e2b4e8fe826ea.npy'
scene = dict(np.load(path))
scene_raw = dict(np.load(path_raw))
anno = np.load(path_anno, allow_pickle=True)