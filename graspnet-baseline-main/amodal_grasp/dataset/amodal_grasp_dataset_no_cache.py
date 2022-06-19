import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from graspnetAPI import GraspNet
import cv2
from skimage.measure import regionprops
from torchvision.transforms.functional import normalize

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m

def remove_zero_point_in_pc(pc):
    nozero_mask = ~((pc[:, 0] == 0) & (pc[:, 1] == 0) & (pc[:, 2] == 0))
    return pc[nozero_mask], nozero_mask


def normalize_point_cloud_in_2d(pc):
    pc_flatten = pc.reshape(-1, 3)
    pc_nozero, nonzero_mask = remove_zero_point_in_pc(pc_flatten)
    nozero_index = nonzero_mask.nonzero()[0]
    pc_normalized, centroid, m = normalize_point_cloud(pc_nozero)
    pc_flatten[nozero_index] = pc_normalized
    pc = pc_flatten.reshape(pc.shape)
    return pc, centroid, m

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



class GraspNetDatasetForAmodalGrasp:
    def __init__(self,
                 data_root,
                 camera='kinect',
                 split='train',
                 with_mesh=True,
                 with_pc=True,
                 with_nocs=True,
                 with_grasp=True):
        self.with_grasp = with_grasp
        self.with_nocs = with_nocs
        self.split = split
        self.data_root = data_root
        self.camera = camera
        self.g = GraspNet(self.data_root, self.camera, self.split)
        self.with_pc = with_pc
        if self.with_pc:
            self.depth2pc = DepthToPc()
        self.with_mesh = with_mesh
        if self.with_mesh:
            self.meshes_path = os.path.join(data_root,'models.npz')
            print(f'Loading {self.meshes_path} ...')
            self.meshes = np.load(self.meshes_path, allow_pickle=True)

    def __getitem__(self, item):
        rgb_path, depth_path, seg_path, meta_path, rect_path, scene_name, ann_id = self.g.loadData(item)
        labels_path = os.path.join(self.data_root, f'scenes/{scene_name}/object_id_list.txt')
        labels = np.int32(np.loadtxt(labels_path))
        rgb = cv2.imread(rgb_path)[:, :, ::-1]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        masks = np.float32([seg == i for i in (labels + 1) if i != 0])  # seg start with 0, 0 is the background, so label + 1 is the objects the map value in seg
        boxes_xyxy = np.float32([(i.bbox[1], i.bbox[0], i.bbox[3], i.bbox[2]) for i in regionprops(seg)])

        if self.with_pc:
            K_path = os.path.join(self.data_root, f'scenes/{scene_name}/{self.camera}/camK.npy')
            K = np.load(K_path, allow_pickle=True)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pc = self.depth2pc(depth, fx, fy, cx, cy)
            # normalize
            pc, centroid, m = normalize_point_cloud_in_2d(pc)
            img = np.dstack((rgb, pc))

        results = dict(img=img,
                       gt_masks=masks,
                       gt_bboxes=boxes_xyxy,
                       gt_labels=labels)

        if self.with_mesh:
            meshes = [self.meshes[str(i).zfill(3)] for i in labels]

        results['gt_meshes'] = meshes
        return results

    def __len__(self):
        return len(self.g)



if __name__ == '__main__':
    import time
    dataset = GraspNetDatasetForAmodalGrasp(data_root='/disk5/data/graspnet')
    start = time.time()
    data = dataset[0]
    print(f'data time {time.time() - start}')