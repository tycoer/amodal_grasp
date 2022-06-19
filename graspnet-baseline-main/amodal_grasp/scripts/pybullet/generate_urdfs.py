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


obj_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
           12, 13, 14, 15, 34, 37, 43,
           46, 58, 61, 65,
           ]


if __name__ == '__main__':
    data_root = '/disk2/data/graspnet'
    save_dir = os.path.join(data_root, f'amodal_grasp/urdfs')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(88):
        if i not in obj_ids:
            continue
        obj_name = str(i).zfill(3)
        obj_visual_path = os.path.join(data_root,
                                       f'amodal_grasp/models_visual/{obj_name}/textured.obj'
                                       # f'amodal_grasp/models/{obj_name}.obj'

                                       )
        obj_collision_path = os.path.join(data_root, f'amodal_grasp/models_collision/{obj_name}.obj')
        handle = modify_urdf('../../data/template.urdf',
                             visual_filename=obj_visual_path,
                             collision_filename=obj_visual_path,
                             object_name=obj_name)

        urdf_target_path = os.path.join(save_dir, f'{obj_name}.urdf')
        handle.write(urdf_target_path)