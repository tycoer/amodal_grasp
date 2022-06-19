from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from os.path import join
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import trimesh
from skimage.measure import regionprops
from copy import deepcopy
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


def get_extr(extr):
    H = np.eye(4)
    H[:3, :3] = Rotation.from_quat(extr[:4]).as_matrix()
    H[:3, 3] = extr[4:]
    return H

if __name__ == '__main__':
    data_root = '/disk2/data/graspnet/amodal_grasp/pybullet'

    nocs_root = join(data_root, 'scenes/nocs')
    anns_new_root = join(data_root, 'scenes/ann')
    os.makedirs(nocs_root, exist_ok=True)
    os.makedirs(anns_new_root, exist_ok=True)

    scene_root = join(data_root, 'scenes')
    anns_root = join(data_root, 'mesh_pose_list')
    rgbs_root = join(data_root, 'scenes/rgb')
    segs_root = join(data_root, 'scenes/seg')
    depths_root = join(data_root, 'scenes/depth')
    extrs_root = join(data_root, 'scenes/extrinsic')

    rgbs_path = sorted(glob(rgbs_root + '/*.jpg'))
    segs_path = sorted(glob(segs_root + '/*.png'))
    depths_path = sorted(glob(depths_root + '/*.png'))
    extrs_path = sorted(glob(extrs_root + '/*.txt'))

    ann_dict = {}
    for i in os.listdir(anns_root):
        with open(os.path.join(anns_root, i), 'r') as f:
            ann_dict[i[:-5]] = json.load(f)
    nocs_para_path = join(data_root, 'nocs_para.json')
    with open(nocs_para_path, 'r') as f:
        nocs_para_dict = json.load(f)

    setup_path = join(data_root, 'setup.json')
    with open(setup_path, 'r') as f:
        setup = json.load(f)
    intr = setup['intrinsic']
    H, W, K = intr['width'], intr['height'], intr['K']
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    depth2pc = DepthToPc(depth_size=(H, W),
                         depth_scale=1000)
    num_imgs = len(rgbs_path)


    for i in tqdm(range(num_imgs)):
        scene_name = rgbs_path[i].split('/')[-1][:10]
        rgb = cv2.imread(rgbs_path[i])[:, :, ::-1]
        depth = cv2.imread(depths_path[i], cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(segs_path[i], cv2.IMREAD_UNCHANGED)
        pc = depth2pc(depth, fx, fy, cx, cy)
        pc_flatten = pc.reshape(-1, 3)
        nocs_flatten = np.zeros_like(pc, dtype='float32').reshape(-1, 3)
        extr = np.loadtxt(extrs_path[i])
        extr = get_extr(extr)
        ann = ann_dict[scene_name]
        label = np.unique(seg)[1:]
        obj_ann_new = deepcopy(ann)
        for j in label:
            obj_ann = ann[str(j)]
            filename, scale, pose_world = obj_ann['mesh_path'], obj_ann['scale'], np.array(obj_ann['pose'])
            obj_name = filename.split('/')[-2]
            pose_cam = extr @ pose_world
            pc_obj_index = np.where(seg.flatten() == j)[0]
            pc_obj = pc_flatten[pc_obj_index]
            pose_cam_inv = np.linalg.inv(pose_cam)
            pc_obj_origin = ((pose_cam_inv[:3, :3] @ pc_obj.T).T + pose_cam_inv[:3, 3]) / scale # 必须先transform 再scale, 否则是错的

            nocs_para = nocs_para_dict[obj_name]
            norm_factor, norm_corner = nocs_para['norm_factor'], np.array(nocs_para['norm_corner'])
            nocs_obj = (pc_obj_origin - norm_corner[0]) * norm_factor + \
                           np.array((0.5, 0.5, 0.5)).reshape(1, 3) - 0.5 * \
                           (norm_corner[1] - norm_corner[0]) * norm_factor

            nocs_flatten[pc_obj_index] = nocs_obj

            obj_ann_new[str(j)]['pose_cam'] = pose_cam.tolist()
            obj_ann_new[str(j)]['pose_world'] = obj_ann_new[str(j)].pop('pose')
        nocs = (nocs_flatten.reshape(H, W, 3) * 255).astype('uint8')
        for k in regionprops(seg):
            obj_ann_new[str(k.label)]['box_xyxy'] = (k.bbox[1], k.bbox[0], k.bbox[3], k.bbox[2])
        cv2.imwrite(join(nocs_root, rgbs_path[i].split('/')[-1]), nocs)
        with open(join(anns_new_root, rgbs_path[i].split('/')[-1][:-4]) + '.json', 'w') as f:
            json.dump(obj_ann_new, f, indent=1)