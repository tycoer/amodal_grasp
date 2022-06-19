from amodal_grasp.utils.simulation import ClutterRemovalSim
from amodal_grasp.utils.transform import Transform, Rotation
from amodal_grasp.utils.perception import camera_on_sphere
from amodal_grasp.utils.implicit import get_mesh_pose_list_from_world
from amodal_grasp.utils.io import write_setup
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm

class ClutterRemovalSim(ClutterRemovalSim):
    def discover_objects(self):
        self.object_urdfs = glob.glob('/disk2/data/graspnet/amodal_grasp/urdfs/*.urdf')

def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.zeros((n, 7), np.float16)
    depths = np.zeros((n, height, width), np.float16)
    rgbs = np.zeros((n, height, width, 3), np.uint8)
    segs = np.zeros((n, height, width), np.uint8)
    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb, depth, seg = sim.camera.render(extrinsic)

        extrinsics[i] = extrinsic.to_list()
        depths[i] = depth
        rgbs[i] = rgb
        segs[i] = seg

    return rgbs, depths, segs, extrinsics


def main(
        scene_id=1,
        num_img_per_scene = 20,
        save_dir='/disk2/data/graspnet/amodal_grasp/pybullet',
        scene='pile',

    ):

    rgb_save_dir = os.path.join(save_dir, 'scenes/rgb')
    depth_save_dir = os.path.join(save_dir, 'scenes/depth')
    seg_save_dir = os.path.join(save_dir, 'scenes/seg')
    extr_save_dir = os.path.join(save_dir, 'scenes/extrinsic')
    ann_save_dir = os.path.join(save_dir, 'mesh_pose_list')
    os.makedirs(rgb_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)
    os.makedirs(extr_save_dir, exist_ok=True)
    os.makedirs(ann_save_dir, exist_ok=True)


    sim = ClutterRemovalSim(scene, object_set='block', gui=False)
    sim.global_scaling = 1.0
    write_setup(
        Path(save_dir),
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth,
    )

    num_obj = np.random.randint(4, 9)
    sim.reset(num_obj)
    sim.save_state()
    mesh_pose_list = get_mesh_pose_list_from_world(sim.world, 'blocks')

    rgbs, depths, segs, extrinsics = render_images(sim, n=num_img_per_scene)
    rgbs = rgbs[:, :, :, ::-1]
    segs[segs == 255] = 0
    depths = (depths * 1000).astype('uint16')

    for i in range(num_img_per_scene):
        cv2.imwrite(os.path.join(rgb_save_dir, f'scene_{str(scene_id).zfill(4)}_{str(i).zfill(4)}.jpg'), rgbs[i])
        cv2.imwrite(os.path.join(depth_save_dir, f'scene_{str(scene_id).zfill(4)}_{str(i).zfill(4)}.png'), depths[i])
        cv2.imwrite(os.path.join(seg_save_dir, f'scene_{str(scene_id).zfill(4)}_{str(i).zfill(4)}.png'), segs[i])
        np.savetxt(os.path.join(extr_save_dir, f'scene_{str(scene_id).zfill(4)}_{str(i).zfill(4)}.txt'), extrinsics[i])

    with open(os.path.join(ann_save_dir, f'scene_{str(scene_id).zfill(4)}.json'), 'w') as f:
        json.dump(mesh_pose_list, f, indent=2)

if __name__ == '__main__':
    save_dir = '/disk2/data/graspnet/amodal_grasp/pybullet'
    for i in tqdm(range(500)):
        main(scene_id=i, num_img_per_scene=20)