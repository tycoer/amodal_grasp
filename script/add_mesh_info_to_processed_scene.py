import os
import numpy as np
import trimesh
import tqdm
processed_scene_root = '/hddisk2/data/hanyang/amodel_dataset/data_test/scenes_processed'
anno_root = '/hddisk2/data/hanyang/amodel_dataset/data_test/mesh_pose_list'
obj_model_root = '/hddisk2/data/hanyang/amodel_dataset/obj_models_new/train/'

if __name__ == '__main__':
    for i in tqdm.tqdm(os.listdir(anno_root)):
        scene_id = i[:-4]
        anno_path = os.path.join(anno_root, scene_id + '.npy')
        processed_scene_path = os.path.join(processed_scene_root, scene_id + '.npz')
        anno = np.load(anno_path, allow_pickle=True)
        processed_scene = dict(np.load(processed_scene_path))

        obj_name_list = []
        mesh_info = []
        voxel_points = []
        uids = []
        for j in anno:
            obj_name = j['obj_name']
            obj_path = os.path.join(obj_model_root, obj_name,  'model.obj')
            mesh = trimesh.load(obj_path, force='mesh')
            mesh: trimesh.Trimesh
            mesh_info.append((mesh.vertices, mesh.faces))

            voxel_path = os.path.join(obj_model_root, obj_name, 'model.binvox')
            voxel = trimesh.load(voxel_path)
            voxel_points.append(voxel.points)
            obj_name_list.append(obj_name)
            uids.append(j['uid'])

        processed_scene.update(dict(gt_voxel_points = voxel_points,
                                    gt_meshes = mesh_info,
                                    obj_names = obj_name_list,
                                    uids = uids))


