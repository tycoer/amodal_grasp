import os
import numpy as np
import trimesh
obj_model_root = '/hddisk2/data/hanyang/amodel_dataset/obj_models_new/train'
urdf_path = os.path.join(obj_model_root, 'urdf_path.txt')
save_root = '/hddisk2/data/hanyang/amodel_dataset/data_test'



gt_mesh = []
gt_voxel = []
if __name__ == '__main__':
    with open(urdf_path, 'r') as f:
        obj_names = f.readlines()
    mesh_info_dict ={}
    for i in obj_names:
        obj_name = i[:-12]
        obj_path = os.path.join(obj_model_root, obj_name + '/model.obj')
        mesh = trimesh.load(obj_path, force='mesh')
        mesh: trimesh.Trimesh
        gt_mesh.append((mesh.vertices, mesh.faces))

        voxel_path = os.path.join(obj_model_root, obj_name + '/model.binvox')
        voxel = trimesh.load(voxel_path)
        gt_voxel.append(voxel.points)


        mesh_info_dict[obj_name] = dict(gt_mesh = gt_mesh,
                                        gt_voxel = gt_voxel,
                                        )
        gt_voxel = []
        gt_mesh = []
    np.save(os.path.join(save_root, 'mesh.npy'), mesh_info_dict)
