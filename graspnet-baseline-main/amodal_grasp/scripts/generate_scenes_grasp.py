import numpy as np
from graspnetAPI import GraspNet, RectGraspGroup, GraspGroup
import matplotlib.pyplot as plt
import os
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from tqdm import tqdm
import json


def filter_points_by_max_score(data):
    scores = data[:, 0]
    point_ids = data[:, -1]
    # index_all = data[:, -1]
    sort_idx_big_to_small = np.argsort(scores)[::-1]
    point_ids_sorted = point_ids[sort_idx_big_to_small]
    _, index = np.unique(point_ids_sorted, return_index=True)
    data_filtered = data[sort_idx_big_to_small][index]
    # data_filtered[:, -1] = index_all[index]
    return data_filtered

GRASP_HEIGHT = 0.02
class MyGraspNet(GraspNet):
    def __init__(self, root, camera='kinect', split='train'):
        super(MyGraspNet, self).__init__(root=root,
                                         camera=camera,
                                         split=split)

        self.grasp_labels = self.loadGraspLabels()

    def loadGrasp(self, sceneId, annId=0, format='6d', camera='kinect', grasp_labels=None, collision_labels=None,
                  fric_coef_thresh=0.4):
        '''
        **Input:**

        - sceneId: int of scene id.

        - annId: int of annotation id.

        - format: string of grasp format, '6d' or 'rect'.

        - camera: string of camera type, 'kinect' or 'realsense'.

        - grasp_labels: dict of grasp labels. Call self.loadGraspLabels if not given.

        - collision_labels: dict of collision labels. Call self.loadCollisionLabels if not given.

        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp.

        **ATTENTION**

        the LOWER the friction coefficient is, the better the grasp is.

        **Output:**

        - If format == '6d', return a GraspGroup instance.

        - If format == 'rect', return a RectGraspGroup instance.
        '''
        import numpy as np
        assert format == '6d' or format == 'rect', 'format must be "6d" or "rect"'
        if format == '6d':
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % (sceneId,), camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            scene_reader = xmlReader(
                os.path.join(self.root, 'scenes', 'scene_%04d' % (sceneId,), camera, 'annotations', '%04d.xml' % (annId,)))
            pose_vectors = scene_reader.getposevectorlist()

            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
            if grasp_labels is None:
                # print('warning: grasp_labels are not given, calling self.loadGraspLabels to retrieve them')
                grasp_labels = {k: self.grasp_labels[k] for k in obj_list} #tycoer
            if collision_labels is None:
                # print('warning: collision_labels are not given, calling self.loadCollisionLabels to retrieve them')
                collision_labels = self.collision_labels['scene_'+str(sceneId).zfill(4)]

            num_views, num_angles, num_depths = 300, 12, 4
            template_views = generate_views(num_views)
            template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
            template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

            collision_dump = collision_labels['scene_' + str(sceneId).zfill(4)]

            # grasp = dict()
            grasp_group = GraspGroup()
            grasp_group.grasp_group_array = np.zeros((0, 18), dtype='float16')
            for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
                sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
                # collision = collision_dump[i]
                collision = collision_dump[i]
                point_inds = np.arange(sampled_points.shape[0])
                num_points = len(point_inds)
                target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
                target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])

                # tycoer
                point_inds = point_inds[:, np.newaxis, np.newaxis, np.newaxis]
                point_inds = np.tile(point_inds, [1, num_views, num_angles, num_depths])

                views = np.tile(template_views, [num_points, 1, 1, 1, 1])
                angles = offsets[:, :, :, :, 0]
                depths = offsets[:, :, :, :, 1]
                widths = offsets[:, :, :, :, 2]
                mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                target_points = target_points[mask1]
                target_points = transform_points(target_points, trans)
                target_points = transform_points(target_points, np.linalg.inv(camera_pose))
                views = views[mask1]
                angles = angles[mask1]
                depths = depths[mask1]
                widths = widths[mask1]
                fric_coefs = fric_coefs[mask1]
                point_inds = point_inds[mask1]
                Rs = batch_viewpoint_params_to_matrix(-views, angles)
                Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)
                Rs = np.matmul(np.linalg.inv(camera_pose)[np.newaxis, :3, :3], Rs)

                num_grasp = widths.shape[0]
                scores = (1.1 - fric_coefs).reshape(-1, 1)
                widths = widths.reshape(-1, 1)
                heights = GRASP_HEIGHT * np.ones((num_grasp, 1))
                depths = depths.reshape(-1, 1)
                rotations = Rs.reshape((-1, 9))
                object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)
                points_ids = point_inds.reshape(-1, 1)

                obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids, points_ids]).astype(
                    np.float16)
                # obj_grasp_array = filter_points_by_max_score(obj_grasp_array)
                grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))
            return grasp_group
        else:
            # 'rect'
            rect_grasps = RectGraspGroup(os.path.join(self.root,'scenes','scene_%04d' % sceneId,camera,'rect','%04d.npy' % annId))
            return rect_grasps


def main(scene_id):
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    collision_labels = g.loadCollisionLabels(scene_id)
    for ann_id in tqdm(range(256), desc=f'{scene_name} processing ...'):
        grasp_group = g.loadGrasp(sceneId=scene_id,
                                  annId=ann_id,
                                  collision_labels=collision_labels)

        save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/grasp')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{save_dir}/{str(ann_id).zfill(4)}.npz')
        np.savez_compressed(
            save_path,
            grasp=np.float16(grasp_group.grasp_group_array)
        )


if __name__ == '__main__':
    import multiprocess as mp
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--camera", type=str, choices=['kinect', 'realsense'], default="kinect")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--num-proc", type=int, default=20)
    args = parser.parse_args()

    data_root = args.data_root
    camera = args.camera
    split = args.split

    g = MyGraspNet(root=data_root, camera=camera, split=split)
    scene_ids = g.sceneIds

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        pool.map(main, scene_ids)
    else:
        for scene_id in scene_ids:
            main(scene_id)
