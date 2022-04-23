import pickle
import xml.etree.ElementTree as ET
import os
import shutil
import argparse
import tqdm
import open3d as o3d
import trimesh
import numpy as np
import json

def load_plk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def dump_plk(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)



def modify_urdf(urdf_filename, visual_filename, collision_filename, object_name=None):
    handle = ET.parse(urdf_filename)
    root = handle.getroot()
    if object_name is not None:
        root.attrib = {'name': object_name}
    root[0][2][0][0].attrib = {'filename': visual_filename}
    root[0][3][0][0].attrib = {'filename': collision_filename}
    return handle

def get_nocs_para(points):

    # points 必须是完整物体（不能是partical的！！！）的点云或mesh的顶点
    # 这个 points 可以是 transform过的
    # points.shape : (n, 3)
    tight_w = max(points[:, 0]) - min(points[:, 0])
    tight_l = max(points[:, 1]) - min(points[:, 1])
    tight_h = max(points[:, 2]) - min(points[:, 2])

    # corner_pts[i+1] = np.amin(part_gts, axis=1)
    norm_factor = np.sqrt(1) / np.sqrt(tight_w ** 2 + tight_l ** 2 + tight_h ** 2)
    norm_factor = norm_factor.tolist() # scale
    corner_pt_left = np.amin(points, axis=0, keepdims=False).tolist()
    corner_pt_right = np.amax(points, axis=0, keepdims=False).tolist()
    norm_corner = np.array([corner_pt_left, corner_pt_right])
    norm_corner = [corner_pt_left, corner_pt_right]
    return norm_factor, norm_corner


# shapenet core 1
category = {'02876657': 'bottle',
            '02880940': 'bowl',
            '02946921': 'can',
            '02954340': 'cap',
            '02992529': 'cell_phone',
            '03797390': 'mug'}

# shapenet core 2


def main(obj_model_root, save_dir, urdf_template):
    # obj_model_root = 'G:/project/obj_models/train'
    # save_dir = 'G:/project/obj_models_new'
    # template = 'template.urdf'
    os.makedirs(save_dir, exist_ok=True)
    urdf_list = []
    nocs_para = {}
    assert os.path.exists(obj_model_root)
    for root, dirs, files in tqdm.tqdm(os.walk(obj_model_root)):
        if len(files) > 3:
            root = root.replace('\\', '/')
            category_id = root.split('/')[-2]
            if category_id in category and len(files) > 3:
                root_target = root.replace(obj_model_root, save_dir).replace(category_id, category[category_id])
                cache = root_target.split('/')
                os.makedirs(root_target, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(root, file), os.path.join(root_target, file))
                    if file.endswith('.obj'):
                        #concatenating texture: may result in visual artifacts
                        mesh = trimesh.load(os.path.join(root_target, file), force='mesh')
                        # mesh = o3d.io.read_triangle_mesh(os.path.join(root_target, file))

                        norm_factor, norm_corner = get_nocs_para(np.array(mesh.vertices))
                        nocs_para[f'{cache[-2]}/{cache[-1]}'] = dict(norm_factor=norm_factor,
                                                                     norm_corner=norm_corner)
                # save urdf
                visual_filename = collision_filename = 'model.obj'
                handle = modify_urdf(urdf_filename=urdf_template,
                                     visual_filename=visual_filename,
                                     collision_filename=collision_filename,
                                     object_name=category[category_id])
                urdf_target_path = os.path.join(root_target, 'model.urdf')
                handle.write(urdf_target_path)
                urdf_list.append(f'{cache[-2]}/{cache[-1]}/model.urdf\n')
    with open(f'{save_dir}/urdf_path.txt', 'w') as f:
        f.writelines(urdf_list)

    with open(f'{save_dir}/nocs_para.json', 'w') as f:
        json.dump(nocs_para, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从shapenet生成新的数据集')
    parser.add_argument('--obj_model_root', type=str, default='G:/project/obj_models/train')
    parser.add_argument('--save_dir', type=str, default='G:/project/obj_models_new/train')
    parser.add_argument('--urdf_template', type=str, default='data/template.urdf')

    args = parser.parse_args()
    main(
         obj_model_root=args.obj_model_root,
         save_dir=args.save_dir,
         urdf_template=args.urdf_template
         )