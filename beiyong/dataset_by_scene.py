import os
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.datasets import CustomDataset
from mmdet.core.mask import BitmapMasks
import cv2
import json
import torch
import torchvision
import matplotlib.pyplot as plt

def generate_heatmap_2d(uv, heatmap_shape, kernal_size=3, sigma_x=2):
    hm = np.zeros(heatmap_shape)
    hm[uv[1], uv[0]] = 1
    hm = cv2.GaussianBlur(hm, (kernal_size, kernal_size), sigma_x)
    hm /= hm.max()  # normalize hm to [0, 1]
    return hm # outshape


@PIPELINES.register_module()
class GenerateHM:
    def __init__(self,
                 max_heatmap_num=10,
                 heatmap_shape=(80, 80),
                 kernal_size=7,
                 sigma_x=2):
        self.max_heatmap_num = max_heatmap_num
        self.heatmap_shape = heatmap_shape
        self.sigma_x = sigma_x
        self.kernal_size = kernal_size

    def __call__(self, results):
        uvs = results['gt_gripper_T_uv_on_obj']
        valid_index = np.zeros(self.max_heatmap_num)
        valid_index[:len(uvs)] = 1
        valid_index = valid_index.astype('bool')
        ray_index = results['index_ray']
        quat = results['gt_gripper_quat']
        qual = results['gt_gripper_qual'].reshape(-1, 1)
        width = results['gt_gripper_width'].reshape(-1, 1)
        grasp_info = np.hstack((uvs, np.hstack((quat, qual, width))[ray_index]))

        if len(uvs) < self.max_heatmap_num:
            grasp_info_padding = np.zeros((self.max_heatmap_num, grasp_info.shape[1]))
            grasp_info_padding[:, 7] = 0.08 # width 负样本应该为 0.08 而不是0
            grasp_info_padding[:len(grasp_info), :] = grasp_info
            grasp_info = np.float32(grasp_info_padding)

        else:
            grasp_info = np.float32(grasp_info[:self.max_heatmap_num])

        uvs = grasp_info[:, :2]
        uv_for_hm = uvs / results['img'].shape[:2] * np.array(self.heatmap_shape)
        uv_for_hm = np.int32(uv_for_hm)
        heatmaps = np.zeros((self.max_heatmap_num,) + self.heatmap_shape, dtype='float32')
        for i in range(len(uv_for_hm)):
            uv = uv_for_hm[i]
            if uv[0] == 0 and uv[1] == 0:
                continue
            heatmaps[i] = generate_heatmap_2d(uv,
                                              heatmap_shape=self.heatmap_shape,
                                              kernal_size=self.kernal_size,
                                              sigma_x=self.sigma_x)

        results['gt_heatmaps'] = heatmaps.astype('float32')
        results['gt_gripper_T_uv_for_hm'] = uv_for_hm
        results['gt_gripper_T_uv'] = grasp_info[:, :2]
        results['gt_gripper_quat'] = grasp_info[:, 2:6].astype('float32')
        results['gt_gripper_qual'] = grasp_info[:, 6].astype(np.int8).reshape(-1, 1)
        results['gt_gripper_width'] = grasp_info[:, 7].astype('float32').reshape(-1, 1)
        results['gt_gripper_valid_index'] = valid_index
        return results

@PIPELINES.register_module()
class SimplePadding:
    def __init__(self,
                 out_shape=(640, 640)):
        self.out_shape = out_shape
    def __call__(self, results):
        img = results['img']
        h, w, d = img.shape
        padding_shape = self.out_shape + (d,)
        padding = np.zeros(padding_shape, dtype='float32')
        padding[:h, :w, :] = img

        results['img'] = padding
        results['pad_shape'] = self.out_shape
        results['scale_factor'] = [1, 1]
        return results

@PIPELINES.register_module()
class StackImgXYZ:
    def __call__(self, results):
        results['img'] = np.dstack((results['img'], results['xyz']))
        return results


@PIPELINES.register_module()
class WarpMask:
    def __call__(self, results):
        gt_masks = results['gt_masks']
        results['gt_masks'] = BitmapMasks(gt_masks.transpose(2, 0, 1), height=gt_masks.shape[0],
                                          width=gt_masks.shape[1])
        return results

@DATASETS.register_module()
class AmodalGraspDataset(CustomDataset):
    CLASSES = {'bottle': '0',
                 'bowl': '1',
                 'can': '2',
                 'cap': '3',
                 'cell_phone': '4',
                 'mug': '5'}

    def __init__(self,
                 data_root,
                 pipeline=None,
                 **kwargs):
        self.data_root = data_root
        self.scene_id_list = os.listdir(self.data_root)
        self.scene_abs_path = tuple(os.path.join(self.data_root, i) for i in self.scene_id_list)

        # self.nocs_para_path = os.path.join(self.data_root, 'nocs_para.json')
        # with open(self.nocs_para_path, 'r') as f:
        #     self.nocs_para = json.load(f)
        self.mesh_processed_path = os.path.join(self.data_root, 'mesh.npz')
        # self.gt_mesh_dict = dict(np.load(self.mesh_processed_path))


        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self), dtype='int64')

    def __len__(self):
        return len(self.scene_id_list)


    def __getitem__(self, item):
        scene_abs_path = self.scene_abs_path[item]
        results = dict(np.load(scene_abs_path, allow_pickle=True))

        # add mesh_info
        # obj_names = results['obj_names']



        # add other info
        results['scene_id'] = self.scene_id_list[item]
        results['img_shape'] = results['img'].shape
        results['gt_labels'] = np.int64(results['gt_labels'])


        if self.pipeline is not None:
            # results = self.pipeline(results)

            try:
                results = self.pipeline(results)
            except:
                # print(item)
                results = self[np.random.choice(len(self))]
        return results


    def get_mesh_and_voxel(self, obj_names):
        voxel = []
        mesh = []
        for i in obj_names:
            voxel.append(self.gt_mesh_dict[i]['gt_voxel'])
            mesh.append(self.gt_mesh_dict[i]['gt_mesh'])
        return voxel, mesh


@PIPELINES.register_module()
class GenerateGraspMap:
    def __init__(self,
                 map_size = 80):
        self.map_size = map_size

    def __call__(self, results):
        grasp_uv_on_obj = np.int32(results['gt_gripper_T_uv_on_obj'] / np.array(results['pad_shape']) * np.array(self.map_size))
        grasp_qual = results['gt_gripper_qual']
        grasp_width = results['gt_gripper_width']
        grasp_quat = results['gt_gripper_quat']
        index_ray = results['index_ray']

        grasp_map = np.zeros((6, self.map_size, self.map_size), dtype='float32')
        grasp_map_value = np.hstack((grasp_qual.reshape(-1, 1),
                                     grasp_width.reshape(-1, 1),
                                     grasp_quat))[index_ray]
        for i in range(grasp_uv_on_obj.__len__()):
            u, v = grasp_uv_on_obj[i]
            grasp_map[:, v, u] = grasp_map_value[i]

        results['grasp_map'] = grasp_map
        return results



if __name__ == '__main__':
    data_root = '/hddisk2/data/hanyang/amodel_dataset/data_test/scenes_processed'
    dataset = AmodalGraspDataset(data_root=data_root)
    data = dataset[0]