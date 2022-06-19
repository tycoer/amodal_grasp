from utils.simulation import ClutterRemovalSim, Gripper
from utils.perception import *
from utils.transform import Rotation, Transform
from pathlib import Path
import os
from utils.grasp import Label
from pathlib import Path
import time

import numpy as np
import pybullet

from utils.grasp import Label
from utils import btsim, workspace_lines
from utils.transform import Rotation, Transform
from utils.misc import apply_noise
from utils.perception import *


class ClutterRemovalSimWithCategory(ClutterRemovalSim):
    def __init__(self,
                 obj_model_dir,
                 scene,
                 object_set,
                 gui=True,
                 seed=None,
                 add_noise=False,
                 sideview=False,
                 save_dir=None,
                 save_freq=8,
                 category='bowl'):
        self.category = category
        self.obj_model_dir = Path(obj_model_dir)
        assert scene in ["pile", "packed"]

        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(320, 240, 540.0, 540.0, 160, 120) # tycoer
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        self.global_scaling = 0.1

    def discover_objects(self):
        # root = '/hdd0/data/giga/obj_models_new/train'
        # self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
        urdf_path = os.path.join(self.obj_model_dir, 'urdf_path.txt')
        with open(urdf_path, 'r') as f:
            urdf_path = f.readlines()
        self.object_urdfs = [os.path.join(self.obj_model_dir, f[:-1]) for f in urdf_path]


    # def discover_objects(self):
    #     # root = '/hdd0/data/giga/obj_models_new/train'
    #     # self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
    #     urdf_path = os.path.join(self.obj_model_dir, 'urdf_path.txt')
    #     with open(urdf_path, 'r') as f:
    #         urdf_path = f.readlines()
    #     self.object_urdfs_without_category = [os.path.join(self.obj_model_dir, f[:-1]) for f in urdf_path if self.category not in str(f)] # tycoer
    #     # tycoer
    #     self.object_urdfs_with_category = [os.path.join(self.obj_model_dir, f[:-1]) for f in urdf_path if self.category in str(f)]


    # def generate_pile_scene(self, object_count, table_height):
    #     # place box
    #     urdf = self.urdf_root / "setup" / "box.urdf"
    #     pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
    #     box = self.world.load_urdf(urdf, pose, scale=1.3)
    #
    #     # drop objects\
    #     # tycoer
    #     if object_count < 2:
    #         object_count = 2
    #     urdf_with_category_num = np.random.randint(1, object_count)
    #     urdf_without_category_num = object_count - urdf_with_category_num
    #
    #     urdfs = self.rng.choice(self.object_urdfs_with_category, size=urdf_with_category_num).tolist() + \
    #             self.rng.choice(self.object_urdfs_without_category, size=urdf_without_category_num).tolist()
    #
    #     ###############
    #     # urdfs = self.rng.choice(self.object_urdfs, size=object_count)
    #     for urdf in urdfs:
    #         rotation = Rotation.random(random_state=self.rng)
    #         xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
    #         pose = Transform(rotation, np.r_[xy, table_height + 0.2])
    #         scale = self.rng.uniform(0.8, 1.0)
    #         self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
    #         self.wait_for_objects_to_rest(timeout=1.0)
    #     # remove box
    #     self.world.remove_body(box)
    #     self.remove_and_wait()

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width, None
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width, None
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    contacts = self.world.get_contacts(self.gripper.body)
                    result = Label.SUCCESS, self.gripper.read(), contacts[0].bodyB.uid
                    if remove:
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width, None

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result
