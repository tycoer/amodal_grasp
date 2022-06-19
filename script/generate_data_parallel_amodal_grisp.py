import argparse
from pathlib import Path
import sys
# sys.path.append('..')
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp
import os
from utils.grasp import Grasp, Label
from utils.io import *
from utils.perception import *
from utils.simulation_amodal_grisp import ClutterRemovalSimWithCategory # tycoer
from utils.transform import Rotation, Transform
from utils.implicit import get_mesh_pose_list_from_world
from utils.grasp import Grasp, Label
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)



OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6

# tycoer  修改自 utils/io.write_sensor_data
def write_sensor_data(save_dir, rgb_imgs, depth_imgs, mask_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = save_dir / "scenes" / (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path,
                        rgb_imgs=rgb_imgs,
                        depth_imgs=depth_imgs,
                        mask_imgs=mask_imgs,
                        extrinsics=extrinsics,)
    return scene_id


def get_mesh_pose_list_from_world(world, object_set, exclude_plane=True):
    mesh_pose_list = []
    # collect object mesh paths and poses
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, _, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('./data/urdfs', object_set, name + '.urdf')

        ################ tycoer ################
        _, verts = world.p.getMeshData(uid)
        aabb = world.p.getAABB(uid) # aabb: (bbox_x_min, bbox_y_min, bbox_z_min), (bbox_x_max, bbox_y_max, bbox_z_max)
        category = mesh_path.split('/')[-2]
        cache = mesh_path.split('/')
        mesh_info = dict(mesh_path=mesh_path,
                         obj_name=f'{cache[-3]}/{cache[-2]}',
                         scale=scale,
                         pose=pose,
                         uid=uid,
                         category=category,
                         aabb=aabb)
        mesh_pose_list.append(mesh_info)
    return mesh_pose_list


def write_grasp(save_dir, scene_id, grasp, label):
    # TODO concurrent writes could be an issue
    csv_path = save_dir / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label", "grispped_object_uid"],
            # ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    grispped_object_uid = grasp.grispped_object_uid
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label,
               grispped_object_uid
               )

def write_point_cloud(save_dir, scene_id, point_cloud, name="point_clouds"):
    path = save_dir / name / (scene_id)
    np.save(path, point_cloud)


def main(args, rank):
    GRASPS_PER_SCENE = args.grasps_per_scene
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ClutterRemovalSimWithCategory(args.obj_model_dir, args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    if rank == 0:
        (args.save_dir / "scenes").mkdir(parents=True)
        write_setup(
            args.save_dir,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )
        if args.save_scene:
            (args.save_dir / "mesh_pose_list").mkdir(parents=True)

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()
        # render synthetic depth images
        n = MAX_VIEWPOINT_COUNT
        # depth_imgs, extrinsics = render_images(sim, n)
        # depth_imgs_side, extrinsics_side = render_side_images(sim, 1, args.random)
        # tycoer
        '''
        render_images 和 render_side_images 区别
        render_images 采用n个视角相机，生成n个rgb_imgs, depth_imgs, mask_imges, extrinsics 最终depth_imgs将用于tsdf的生成（不做为gt）
        render_side_images 只采用1视角相机， 生成1个rgb_imgs, depth_imgs, mask_imges, extrinsics， 最终作为gt
        '''

        rgb_imgs, depth_imgs, mask_imges, extrinsics = render_images(sim, n)
        rgb_imgs_side, depth_imgs_side, mask_imges_side, extrinsics_side = render_side_images(sim, 1, args.random)



        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud() # 此时的tsdf 是带有 plane

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box) # 去除 plane
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        # scene_id = write_sensor_data(args.save_dir, depth_imgs_side, extrinsics_side)
        # tycoer
        scene_id = write_sensor_data(args.save_dir, rgb_imgs_side, depth_imgs_side, mask_imges_side, extrinsics_side)

        if args.save_scene:
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
            write_point_cloud(args.save_dir, scene_id, mesh_pose_list, name="mesh_pose_list")

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)

            # store the sample
            write_grasp(args.save_dir, scene_id, grasp, label)
            pbar.update()

    pbar.close()
    print('Process %d finished!' % rank)


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), dtype='float32')

    # tycoer
    rgb_imgs = np.empty((n, height, width, 3), np.uint8)
    mask_imgs = np.empty((n, height, width), np.int32)


    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb, depth_img, mask = sim.camera.render(extrinsic) # tycoer

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

        # tycoer
        rgb_imgs[i] = rgb
        mask_imgs[i] = mask
    # return depth_imgs, extrinsics
    return rgb_imgs, depth_imgs, mask_imgs, extrinsics


def render_side_images(sim, n=1, random=False):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    # tycoer
    rgb_imgs = np.empty((n, height, width, 3), np.uint8)
    mask_imgs = np.empty((n, height, width), np.int32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb, depth_img, mask = sim.camera.render(extrinsic) # tycoer

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

        # tycoer
        rgb_imgs[i] = rgb
        mask_imgs[i] = mask
    # return depth_imgs, extrinsics
    return rgb_imgs, depth_imgs, mask_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    outcomes, widths, grispped_object_uids = [], [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        # remove 抓取成功后 是否在场景中移除被抓成功的物体
        outcome, width, grispped_object_uid = sim.execute_grasp(candidate, remove=False)
        # outcome, width = sim.execute_grasp(candidate, remove=False)


        outcomes.append(outcome)
        widths.append(width)
        grispped_object_uids.append(grispped_object_uid) # tycoer
    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]
        grispped_object_uid = grispped_object_uids[idx_of_widest_peak] # tycoer
    # tycoer
    grasp = Grasp(pose=Transform(ori, pos),
                  width=width,
                  )
    grasp.grispped_object_uid = grispped_object_uid
    return grasp, int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_model_dir", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    args.save_scene = True
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_proc)(delayed(main)(args, i) for i in range(args.num_proc))
