import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from graspnetAPI import GraspNet
import cv2
from skimage.measure import regionprops
from torchvision.transforms.functional import normalize, to_tensor
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import box_convert
import json
import glob
from tqdm import tqdm
import time

import torch
import time
from torchvision.transforms import transforms
import torch.nn.functional as F



def resize_box(box_xyxy, ori_wh, resized_wh):
    scale_factor_xyxy = ((np.array(ori_wh) / np.array(resized_wh))[None, :]).repeat(2, 1)
    box_xyxy_resized = box_xyxy / scale_factor_xyxy
    return box_xyxy_resized


def clip_box(box, wh):
    w, h = wh
    box[:, 0::2] = np.clip(box[:, 0::2], 0, w - 1)
    box[:, 1::2] = np.clip(box[:, 1::2], 0, h - 1)
    return box

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
                 with_mesh=False,
                 with_pc=True,
                 with_nocs=True,
                 with_grasp=True,
                 img_norm_std=[0.229,0.224,0.225],
                 img_norm_mean=[0.485,0.456,0.406],
                 img_shape = (288, 512),
                 quality_th = 0.04
                 # per_instance_max_grasp=16,
                 # global_max_grasp=32,
                 ):

        self.split = split
        self.data_root = data_root
        self.camera = camera
        self.quality_th = quality_th
        self.with_grasp = with_grasp
        self.with_nocs = with_nocs
        self.with_pc = with_pc
        self.with_mesh = with_mesh
        self.img_shape = img_shape
        self.img_norm_std = img_norm_std
        self.img_norm_mean = img_norm_mean


        self.depth2pc = DepthToPc()

        # self.pipeline = transforms.Compose(
        #     [transforms.Resize(size=img_shape,
        #                        interpolation=0 # NERAEST
        #                        ),
        #      # transforms.Pad()
        #      ]
        # )

        self.pipeline = ToTensor()

        self.g = GraspNet(self.data_root, self.camera, self.split)
        # if self.with_mesh:
        #     self.meshes_path = os.path.join(data_root,'models.npz')
        #     print(f'Loading {self.meshes_path} ...')
        #     self.meshes = np.load(self.meshes_path, allow_pickle=True)

        if self.with_grasp:
            self.grasps_obj_dir = os.path.join(self.data_root, 'amodal_grasp/grasp_label')
            self.grasps_obj = {}
            for i in self.g.objIds:
                obj_name = str(int(i)).zfill(3)
                self.grasps_obj[obj_name] = np.load(os.path.join(self.grasps_obj_dir, f'{obj_name}.npz'))['grasps']
        self.CLASSES = [str(i) for i in self.g.objIds]
        self.label_map = {obj_id: i for i, obj_id in enumerate(self.g.objIds)}
        self.flag = np.zeros(len(self), dtype='int32')



    def __getitem__(self, item):
        # start = time.time()
        h, w = self.img_shape
        rgb_path, depth_path, seg_path, meta_path, rect_path, scene_name, ann_id = self.g.loadData(item)

        # rgb
        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask and label
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        labels = np.unique(seg)[1:] - 1
        mask = (seg != 0).astype('int32')  # seg start with 0, 0 is the background, so label + 1 is the objects the map value in seg

        # box
        filename = str(int(ann_id)).zfill(4)
        box_path = os.path.join(self.data_root, f'amodal_grasp/scenes/{scene_name}/{self.camera}/2d_box_xyxy/{filename}.json')
        with open(box_path, 'r') as f:
            box = json.load(f)
        box_xyxy = np.float32([box[str(i).zfill(3)] for i in labels])
        # print('ori_time', time.time() - start)
        # pc
        # pc_start =  time.time()
        if self.with_pc:
            # 耗时越0.1s
            # start = time.time()
            # pc_path = os.path.join(self.data_root, f'amodal_grasp/scenes/{scene_name}/{self.camera}/pc/{filename}.npz')
            # pc = np.load(pc_path)['pc']
            # print(time.time() - start)

            start = time.time()
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            K_path = os.path.join(self.data_root, f'scenes/{scene_name}/{self.camera}/camK.npy')
            K = np.load(K_path)
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][0]
            pc = self.depth2pc(depth, fx, fy, cx, cy)
            print(time.time() - start)
            img = np.dstack((img, pc))


        img = np.dstack((img, seg[:, :, None], mask[:, :, None]))
        # print('pc_time', time.time() - pc_start)

        # nocs_start =  time.time()
        if self.with_nocs:
            nocs_path = os.path.join(self.data_root, f'amodal_grasp/scenes/{scene_name}/{self.camera}/nocs/{filename}.png')
            nocs = cv2.imread(nocs_path, cv2.IMREAD_UNCHANGED)
            nocs = np.float32(nocs / 255) # decode
            img = np.dstack((img, nocs))
        # print('nocs_time', time.time() - nocs_start)

        grasp_start =  time.time()
        if self.with_grasp:
            # 耗时约 0.2s
            grasp_map_path = os.path.join(self.data_root, f'amodal_grasp/scenes/{scene_name}/{self.camera}/grasp_map/{filename}.npz')
            grasp_map = np.load(grasp_map_path)['grasp_map']
            print('grasp_time', time.time() - grasp_start)

            distance =  grasp_map[:, :, 0]
            # 根据distance 获取 quality_map
            quality_map = ((distance < self.quality_th) & (distance != 0) ).astype('float32')
            grasp_map[:, :, 0] = quality_map
            # grasp_map中quality_map为0的位置置零
            grasp_map[:, :, 1:] = quality_map[:, :, None] * grasp_map[:, :, 1:]
            img = np.dstack((img, grasp_map))
        print('grasp_time', time.time() - grasp_start)


        # post_start = time.time()
        # img_shape (720, 1280, 19)
        # 19 names: r, g, b, x, y, z ,seg, mask, nocs_x, nocs_y, nocs_z, qual, qx, qy, qz, qw, width, depth, tolerance
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        box_xyxy = np.int32(resize_box(box_xyxy,
                              ori_wh=(1280, 720),
                              resized_wh=(w, h)))
        box_xyxy = clip_box(box_xyxy, (w, h))
        #
        box_xyxy = torch.from_numpy(box_xyxy)

        '''
        a = img[:, :, :3].astype('uint8')
        plt.imshow(a)
        plt.show()
        for i in np.int32(box_xyxy):
            cv2.rectangle(a, i[:2], i[2:], (255, 0, 0))
        plt.imshow(a)
        plt.show()
        '''

        seg = img[:, :, 6]
        maps = img[:, :, 7:]
        gt_maps = np.zeros((len(labels),) + maps.shape)
        for i, label in enumerate(labels + 1):
            gt_maps[i][seg == label] = maps[seg == label]
        # gt_maps = torch.from_numpy(gt_maps.transpose(0, 3, 1, 2))
        rgb = to_tensor(img[:, :, :3].astype('uint8'))
        # rgb = normalize(rgb, std=self.img_norm_std, mean=self.img_norm_mean)
        # pc = torch.from_numpy(img[:, :, 3:6].transpose(2, 0, 1))
        # rgb = torch.cat((rgb, pc))

        labels = torch.Tensor([self.label_map[i] for i in labels])
        # img_pad = torch.zeros((17, 640, 640), dtype=torch.float32)
        # img_pad[:, :360, :640] = img
        '''

        '''

        results = dict(
                       img=rgb.float(),
                       depth=pc[:, :, -1],
                       gt_bboxes=box_xyxy.float(),
                       gt_labels=labels.long(),
                       gt_masks=gt_maps[:, :, :, 0],
                       gt_nocs=gt_maps[:, :, :, 1:4],
                       gt_grasps =gt_maps[:, :, :, 4:],
                       filename=f'{str(scene_name).zfill(4)}_{str(ann_id).zfill(4)}.png',
                       intrinsics=K,
                       )
        # print('post_time', time.time() - post_start)
        # print(item)
        return results

    def __len__(self):
        return len(self.g)


if __name__ == '__main__':
    import time
    dataset = GraspNetDatasetForAmodalGrasp(data_root='/disk2/data/graspnet')
    start = time.time()
    data = dataset[0]
    print(f'data time {time.time() - start}')