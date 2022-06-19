import trimesh
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from trimesh import creation
import json
import cv2
import tqdm




scene_processed_root = '/disk1/data/amodal_grasp/packed_raw/scenes_processed'
obj_model_root = '/disk1/data/amodal_grasp/obj_models_new/train'
setup_path = '/disk1/data/amodal_grasp/packed_raw/setup.json'
plane_path = '/home/hanyang/amodal_grisp/data/urdfs/setup/plane.obj'



with open(setup_path, 'r') as f:
    setup = json.load(f)
K = setup['intrinsic']['K']
fx = K[0]
fy = K[4]
cx = K[2]
cy = K[5]

visualize = True


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def xyz2uv(xyz, fx, fy, cx, cy):
    xyz=np.array(xyz).flatten()
    x,y=xyz[0]/xyz[2],xyz[1]/xyz[2]
    u,v=x*fx+cx,y*fy+cy
    uv=np.int0([u,v])
    return uv



if __name__ == '__main__':
    results = []
    results_scene = {}
    print('Processing... ')
    for filename in tqdm.tqdm(os.listdir(scene_processed_root)):
        scene_abs_path = os.path.join(scene_processed_root, filename)
        scene_info = dict(np.load(scene_abs_path))
        scene = trimesh.Scene()
        # plane = trimesh.load(plane_path, force='mesh')
        # plane.apply_transform(scene_info['extrinsic'])
        # scene.add_geometry(plane)
        for i, obj_name in enumerate(scene_info['obj_names']):
            obj_abs_path = os.path.join(obj_model_root, obj_name, 'model.obj')
            mesh = trimesh.load(obj_abs_path, force='mesh')
            scale = scene_info['gt_obj_scales'][i]
            pose = scene_info['gt_obj_poses'][i]

            mesh: trimesh.Trimesh
            mesh.apply_scale(scale)
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)
        scene_mesh = as_mesh(scene)
        scene_mesh: trimesh.Trimesh

        ray_origins = scene_info['gt_gripper_T_cam']
        ray_directions = np.array([Rotation.from_quat(quat).as_euler('xyz', degrees=False) for quat in scene_info['gt_gripper_quat']])
        locations, index_ray, index_tri = scene_mesh.ray.intersects_location(ray_directions=ray_directions,
                                                                        ray_origins=ray_origins,
                                                                        multiple_hits=False)

        gripper_T_cam_on_obj = locations
        gripper_quat = scene_info['gt_gripper_quat'][index_ray]
        gripper_T_uv_on_obj = np.array([xyz2uv(loc, fx=fx, fy=fy, cx=cx, cy=cy) for loc in locations])

        scene_info.update(dict(gt_gripper_T_uv_on_obj=gripper_T_uv_on_obj,
                               gt_gripper_T_cam_on_obj=gripper_T_cam_on_obj,
                               index_ray=index_ray))
        np.savez_compressed(scene_abs_path, **scene_info)


    #     result = [{str(scene_info['scene_id']): {'gripper_T_cam_on_obj':gripper_T_cam_on_obj[i],
    #                                              'gripper_quat':gripper_quat[i],
    #                                              'gripper_T_uv_on_obj':gripper_T_uv_on_obj[i],
    #                                              'gripper_width': scene_info['gt_gripper_width'][i],
    #                                              'gripper_qual': scene_info['gt_gripper_qual'][i],
    #                                              'index': index_ray[i],
    #                                          }} for i in range(len(index_ray))]
    #     results.extend(result)
    # np.save('/disk1/data/amodal_grasp/packed_raw/grasp_on_obj.npy', results, allow_pickle=True)

    # if visualize:
    #     for i in range(len(locations)):
    #         axis_transform = np.eye(4)
    #         axis_transform[:3, :3] = Rotation.from_quat(gripper_quat[i]).as_matrix()
    #         axis_transform[:3, 3] =  gripper_T_cam_on_obj[i]
    #         axis = creation.axis(transform=axis_transform, origin_size=0.004)
    #         scene.add_geometry(axis)
    #     scene.export('/home/hanyang/scene.obj')
    #
    #     img = scene_info['img']
    #     for uv in gripper_T_uv_on_obj:
    #         cv2.drawMarker(img, uv, (255, 255, 0))
    #     plt.imshow(img)
    #     plt.show()
    #


