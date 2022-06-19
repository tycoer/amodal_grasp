from os.path import join
import trimesh
import os
if __name__ == '__main__':
    data_root = '/disk2/data/graspnet/amodal_grasp/models'
    for i in os.listdir(data_root):
        path = join(data_root, i)
        mesh = trimesh.load(path)
        mesh: trimesh.Trimesh
        mesh.export(join(data_root, f'{i[:3]}.obj'),
                    include_color=True,
                    include_normals=False,
                    include_texture=True,
                    write_texture=False)