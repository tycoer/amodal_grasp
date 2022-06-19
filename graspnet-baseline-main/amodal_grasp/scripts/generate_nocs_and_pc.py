from graspnetAPI import GraspNet
import json
import numpy as np
import os
import cv2
from skimage.measure import regionprops
from tqdm import tqdm
from graspnetAPI.utils.utils import xmlReader
# import open3d as o3d
import matplotlib.pyplot as plt
import argparse

def load_obj_poses(obj_poses_path):
    xml = xmlReader(obj_poses_path)
    obj_poses = xml.get_pose_list()
    obj_poses_dict = {}
    for pose in obj_poses:
        obj_poses_dict[str(pose.id).zfill(3)] = pose.mat_4x4
    return obj_poses_dict

class DepthToPc:
    def __init__(self,
                 depth_size=(720, 1280),
                 depth_scale=1000):
        self.depth_size = depth_size
        self.h, self.w = depth_size
        self.h_grid, self.w_grid = np.mgrid[0: self.h, 0: self.w]
        self.depth_scale = depth_scale
    def __call__(self, depth, fx, fy, cx, cy):
        z = depth / self.depth_scale
        x = (self.w_grid - cx) * z / fx
        y = (self.h_grid - cy) * z / fy
        xyz = np.dstack((x, y, z))
        return xyz

def generate_nocs_map(pc, seg, nocs_para, obj_poses_dict):
    h, w, d = pc.shape
    pc_flatten = pc.reshape(-1, 3)
    nocs_map_flatten = pc_flatten.copy()
    seg_flatten = seg.reshape(-1)
    for i in np.unique(seg):
        if i == 0: continue
        obj_id = i - 1
        H = obj_poses_dict[str(obj_id).zfill(3)]
        H_inv = np.linalg.inv(H)
        R = H_inv[:3, :3]
        T = H_inv[:3, 3]

        pc_obj_index = (seg_flatten == i).nonzero()
        pc_obj = pc_flatten[pc_obj_index]
        pc_obj_origin = (R @ pc_obj.reshape(-1, 3).T).T + T # pc_obj transform 到原点(0, 0, 0)

        nocs_para_obj = nocs_para[str(obj_id).zfill(3)]
        norm_factor, norm_corner = nocs_para_obj['norm_factor'], np.array(nocs_para_obj['norm_corner'])
        nocs_map_obj = (pc_obj_origin - norm_corner[0]) * norm_factor + np.array((0.5, 0.5, 0.5)).reshape(1, 3) - 0.5 * (
                norm_corner[1] - norm_corner[0]) * norm_factor
        nocs_map_flatten[pc_obj_index] = nocs_map_obj
    nocs_map = nocs_map_flatten.reshape(h, w, d)
    return nocs_map


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m

def get_pc_normalize_para(pc):
    centroid = np.mean(pc, axis=0)
    pc_normalized = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_normalized ** 2, axis=1)))
    return centroid, m

def remove_zero_point_in_pc(pc):
    nozero_mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[nozero_mask], nozero_mask


def normalize_point_cloud_in_2d(pc):
    pc_flatten = pc.reshape(-1, 3).copy()
    pc_nozero, nonzero_mask = remove_zero_point_in_pc(pc_flatten)
    nozero_index = nonzero_mask.nonzero()[0]
    pc_normalized, centroid, m = normalize_point_cloud(pc_nozero)
    pc_flatten[nozero_index] = pc_normalized
    pc_normalized = pc_flatten.reshape(pc.shape)
    return pc_normalized, centroid, m


def main(data_root = '/disk2/data/graspnet',
         camera = 'kinect',
         split = 'train',
         # save_dir = '/disk1/data/amodal_grasp'
         ):

    nocs_para_path = os.path.join(data_root, 'amodal_grasp/nocs_para.json')
    with open(nocs_para_path, 'r') as f:
        nocs_para = json.load(f)

    depth2pc = DepthToPc()
    g = GraspNet(data_root, camera=camera, split=split)
    scene_name_flag = ''
    for i in tqdm(range(len(g))):
        rgb_path, depth_path, seg_path, meta_path, rect_path, scene_name, ann_id = g.loadData(i)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        boxes_xyxy = {str(i.label - 1).zfill(3): (i.bbox[1], i.bbox[0], i.bbox[3], i.bbox[2]) for i in regionprops(seg)}

        K_path = os.path.join(data_root, f'scenes/{scene_name}/{camera}/camK.npy')
        K = np.load(K_path, allow_pickle=True)
        obj_poses_path = os.path.join(data_root, f'scenes/{scene_name}/{camera}/annotations/{str(ann_id).zfill(4)}.xml')
        obj_poses_dict = load_obj_poses(obj_poses_path)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = depth2pc(depth, fx, fy, cx, cy)
        pc_flatten = remove_zero_point_in_pc(pc.reshape(-1, 3))[0]
        centroid, m = get_pc_normalize_para(pc_flatten)
        pc[seg == 0] = 0
        nocs_map = generate_nocs_map(pc, seg, nocs_para, obj_poses_dict)
        nocs_map_rgb = np.uint8(nocs_map * 255)
        invaild_nocs_mask = (nocs_map < 0) | (nocs_map > 1)
        invaild_nocs_mask = (invaild_nocs_mask[:, :, 0] | invaild_nocs_mask[:, :, 1] | invaild_nocs_mask[:, :, 2])
        nocs_map_rgb[invaild_nocs_mask] == 0 # 异常点置0

        #################################### save #######################################
        if scene_name_flag != scene_name:
            scene_name_flag = scene_name
            pc_save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/pc')
            nocs_save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/nocs')
            box_save_dir = os.path.join(data_root, f'amodal_grasp/scenes/{scene_name}/{camera}/2d_box_xyxy')

            os.makedirs(pc_save_dir, exist_ok=True)
            os.makedirs(nocs_save_dir, exist_ok=True)
            os.makedirs(box_save_dir, exist_ok=True)


        filename = str(ann_id).zfill(4)
        pc_save_path = os.path.join(pc_save_dir, f'{filename}.npz')
        np.savez_compressed(pc_save_path,
                            pc=np.float16(pc),
                            centroid=centroid,
                            m=m)

        nocs_save_path = os.path.join(nocs_save_dir, f'{filename}.png')
        cv2.imwrite(nocs_save_path, nocs_map_rgb)

        boxes_xyxy_save_path = os.path.join(box_save_dir, f'{filename}.json')
        with open(boxes_xyxy_save_path, 'w') as f:
            json.dump(boxes_xyxy, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--camera", type=str, choices=['kinect', 'realsense'], default="kinect")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    # parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    main(args.data_root,
         args.camera,
         args.split,)

    # nocs_para_path = os.path.join(data_root, 'amodal_grasp/nocs_para.json')
    # with open(nocs_para_path, 'r') as f:
    #     nocs_para = json.load(f)
    #
    # depth2pc = DepthToPc()
    # g = GraspNet(data_root, camera=camera, split=split)
    # scene_name_flag = ''
    #
    #
    # import multiprocessing as mp
    # if args.num_proc > 1:
    #     pool = mp.Pool(processes=args.num_proc)
    #     pool.map()
    # else:
    #     main(args, 0)
    #
    #
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=args.num_proc)(delayed(main)(args.data_root,
    #                                              args.camera,
    #                                              args.split,
    #                                              args.num_proc
    #                                              ) for _ in range(args.num_proc))



