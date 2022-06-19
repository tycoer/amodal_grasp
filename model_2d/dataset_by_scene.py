import os
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.datasets import CustomDataset
from mmdet.core.mask import BitmapMasks
import cv2
import json
import torch


def generate_heatmap_2d(uv, heatmap_shape, sigma=3):
    hm = np.zeros(heatmap_shape)
    hm[uv[1], uv[0]] = 1
    hm = cv2.GaussianBlur(hm, (sigma, sigma), 1)
    hm /= hm.max()  # normalize hm to [0, 1]
    return hm # outshape


@PIPELINES.register_module()
class GenerateHM:
    def __init__(self,
                 max_heatmap_num=16,
                 heatmap_shape=(40, 40),
                 sigma=7):
        self.max_heatmap_num = max_heatmap_num
        self.heatmap_shape = heatmap_shape
        self.sigma = sigma

    def __call__(self, results):
        uvs = results['gt_gripper_T_uv']
        quat = results['gt_gripper_quat']
        qual = results['gt_gripper_qual'].reshape(-1, 1)
        width = results['gt_gripper_width'].reshape(-1, 1)
        grasp_info = np.hstack((uvs, quat, qual, width))

        if len(uvs) < self.max_heatmap_num:
            grasp_info_padding = np.zeros((self.max_heatmap_num, grasp_info.shape[1]))
            grasp_info_padding[:len(grasp_info), :] = grasp_info
            grasp_info = np.float32(grasp_info_padding)

        else:
            grasp_info = np.float32(grasp_info[:self.max_heatmap_num])

        uvs = grasp_info[:, :2]
        uv_for_hm = uvs / results['img_shape'][:2] * np.array(self.heatmap_shape)
        uv_for_hm = np.int32(uv_for_hm)
        heatmaps = np.zeros((self.max_heatmap_num,) + self.heatmap_shape, dtype='uint8')
        for i in range(len(uv_for_hm)):
            uv = uv_for_hm[i]
            heatmaps[i] = generate_heatmap_2d(uv, heatmap_shape=self.heatmap_shape, sigma=self.sigma)

        results['gt_heatmaps'] = heatmaps
        results['gt_gripper_T_uv_for_hm'] = uv_for_hm
        results['gt_gripper_T_uv'] = grasp_info[:, :2]
        results['gt_gripper_quat'] = grasp_info[:, 2:6]
        results['gt_gripper_qual'] = grasp_info[:, 6]
        results['gt_gripper_width'] = grasp_info[:, 7]
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
        padding = np.zeros(padding_shape, dtype='uint8')
        padding[:h, :w, :] = img

        results['img'] = padding
        results['pad_shape'] = padding_shape
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
        results = dict(np.load(scene_abs_path))

        # add mesh_info
        # obj_names = results['obj_names']



        # add other info
        results['scene_id'] = self.scene_id_list[item]
        results['img_shape'] = results['img'].shape
        results['gt_labels'] = np.int64(results['gt_labels'])


        if self.pipeline is not None:
            results = self.pipeline(results)
        return results


    def get_mesh_and_voxel(self, obj_names):
        voxel = []
        mesh = []
        for i in obj_names:
            voxel.append(self.gt_mesh_dict[i]['gt_voxel'])
            mesh.append(self.gt_mesh_dict[i]['gt_mesh'])
        return voxel, mesh





if __name__ == '__main__':
    data_root = '/hddisk2/data/hanyang/amodel_dataset/data_test/scenes_processed'
    dataset = AmodalGraspDataset(data_root=data_root)
    data = dataset[0]