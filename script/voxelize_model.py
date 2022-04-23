import os
import trimesh
import argparse


def main(obj_model_root: str,
         binvox_path: str,
         save_points = True,
         voxel_size = 40):
    for i, j, k in os.walk(obj_model_root):
        for name in k:
            if name.endswith('.obj'):
                obj_path = os.path.join(i, name)
                os.system(f'{binvox_path} -e -d {voxel_size} {obj_path}')
                model_binvox_path = os.path.join(i, 'model.binvox')
                voxel = trimesh.load(model_binvox_path)
                voxel_points = voxel.points
                if save_points:
                    ply = trimesh.PointCloud(voxel_points)
                    ply.export(os.path.join(i, 'voxel.ply'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从shapenet生成voxel')
    parser.add_argument('--obj_model_root', type=str, default='/hddisk2/data/hanyang/amodel_dataset/obj_models_new/train')
    parser.add_argument('--binvox_path', type=str, default='./binvox/binvox')
    parser.add_argument('--save_points', type=bool, default=True)
    parser.add_argument('--voxel_size', type=int, default=40)

    args = parser.parse_args()
    main(
         obj_model_root=args.obj_model_root,
         binvox_path=args.binvox_path,
         save_points=args.save_points,
         voxel_size= args.voxel_size
         )