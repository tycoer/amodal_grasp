import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    data_root = '/disk1/data/giga/data_packed_train_raw'
    scene_cam_root = os.path.join(data_root, 'scenes_cam')

    for i in tqdm(os.listdir(scene_cam_root)):

        pc_path = os.path.join(scene_cam_root, i)
        print(pc_path)

        data = np.load(pc_path, allow_pickle=True)
        pc = data['pc']
        extrinsic = data['extrinsic']
        pc = pc.reshape(-1, 3)
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        pc_o3d : o3d.geometry.PointCloud
        pc_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_o3d.normalize_normals()
        pc_normals = np.float32(pc_o3d.normals)
        np.savez_compressed(pc_path,
                            pc_normals=pc_normals,
                            pc=pc,
                            extrinsic=extrinsic)


