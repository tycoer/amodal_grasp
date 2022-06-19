import os
import open3d as o3d
import numpy as np
from graspnetAPI.utils.utils import get_model_grasps, generate_views, batch_viewpoint_params_to_matrix
import h5py
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pandas as pd


def filter_points_by_max_score(data):
    scores = data[:, -3]
    point_ids = data[:, -1]
    sort_idx_big_to_small = np.argsort(scores)[::-1]
    point_ids_sorted = point_ids[sort_idx_big_to_small]
    _, index = np.unique(point_ids_sorted, return_index=True)
    data_filtered = data[sort_idx_big_to_small][index]
    data_filtered = np.hstack((data_filtered, index[:, None]))
    return data_filtered

def process_single_grasp(dataset_root, obj_idx=66, fric_coef_thresh=0.3):
    num_views, num_angles, num_depths = 300, 12, 4
    tolerance_path = '%s/tolerance/%03d_tolerance.npy' % (dataset_root, obj_idx)
    tolerance = np.load(tolerance_path, allow_pickle=True)
    sampled_points, offsets, fric_coefs, collision = get_model_grasps('%s/grasp_label/%03d_labels.npz' % (dataset_root, obj_idx))
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])
    num_points = sampled_points.shape[0]
    point_inds = np.arange(num_points)
    point_inds = point_inds[:, np.newaxis, np.newaxis, np.newaxis]
    point_inds = np.tile(point_inds, [1, num_views, num_angles, num_depths])

    angles = offsets[:, :, :, :, 0]
    depths = offsets[:, :, :, :, 1]
    widths = offsets[:, :, :, :, 2]
    views = np.tile(template_views, [num_points, 1, 1, 1, 1])
    target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
    target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
    mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0))
    target_points = target_points[mask1]
    views = views[mask1]
    angles = angles[mask1]
    depths = depths[mask1]
    widths = widths[mask1]
    fric_coefs = fric_coefs[mask1]
    tolerance = tolerance[mask1]
    point_inds = point_inds[mask1]
    Rs = batch_viewpoint_params_to_matrix(-views, angles)
    quat = Rotation.from_matrix(Rs).as_quat()
    scores = (1.1 - fric_coefs)

    index = mask1.flatten().nonzero()[0]
    data = np.hstack((quat, target_points, widths[:, None], depths[:, None], scores[:, None],
                      tolerance[:, None],
                      point_inds[:, None], index[:, None]))
    # data = np.float32(filter_points_by_max_score(data))
    return np.float32(data), np.float16(sampled_points)



def process_all_grasp(dataset_root, fric_coef_thresh=0.2):
    save_dir = os.path.join(dataset_root, 'amodal_grasp/grasp_label')
    os.makedirs(save_dir, exist_ok=True)
    max_obj_num = 88
    for obj_idx in tqdm(range((max_obj_num))):
        data, points = process_single_grasp(dataset_root, obj_idx, fric_coef_thresh)
        # grasps = pd.DataFrame(columns=['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z', 'width', 'depth', 'score', 'tolerance', 'point_id', 'index'],
        #                       data=data)
        # grasps_path = os.path.join(save_dir, f'{(str(obj_idx).zfill(3))}_labels.csv')
        # grasps.to_csv(grasps_path, mode='w', index=False)

        np.savez_compressed(os.path.join(save_dir, f'{(str(obj_idx).zfill(3))}.npz'),
                            grasps=data,
                            points=points)



if __name__ == '__main__':
    dataset_root = '/disk2/data/graspnet'
    # process_single_grasp(dataset_root, obj_idx=0)
    process_all_grasp(dataset_root)