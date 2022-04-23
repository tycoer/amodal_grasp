import pickle
import xml.etree.ElementTree as ET
import os
import shutil
import argparse

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


category = {'02876657': 'bottle',
            '02880940': 'bowl',
            '02946921': 'can',
            '02954340': 'cap',
            '02992529': 'cell_phone',
            '03797390': 'mug'}


def main(obj_model_root, save_dir, urdf_template):
    # obj_model_root = 'G:/project/obj_models/train'
    # save_dir = 'G:/project/obj_models_new'
    # template = 'template.urdf'
    os.makedirs(save_dir, exist_ok=True)
    urdf_save_dir = os.path.join(save_dir, 'urdfs')
    os.makedirs(urdf_save_dir, exist_ok=True)
    urdf_list = []
    assert os.path.exists(obj_model_root)
    for root, dirs, files in os.walk(obj_model_root):
        print(root, files)
        if len(files) > 3:
            root = root.replace('\\', '/')
            category_id = root.split('/')[-2]
            if category_id in category and len(files) > 3:
                root_target_model = root.replace(obj_model_root, save_dir + '/model')
                os.makedirs(root_target_model, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(root, file), os.path.join(root_target_model, file))
                cache = root_target_model.split('/')
                visual_filename = collision_filename = f'../{cache[-3]}/{cache[-2]}/{cache[-1]}/model.obj'
                handle = modify_urdf(urdf_filename=urdf_template,
                                     visual_filename=visual_filename,
                                     collision_filename=collision_filename,
                                     object_name=category[category_id])
                urdf_target_path = f'{urdf_save_dir}'
                handle.write(urdf_target_path)
                urdf_list.append()

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