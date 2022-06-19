import os
from os.path import join
import h5py
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import PIPELINES, DATASETS
import matplotlib.pyplot as plt
import cv2
@PIPELINES.register_module()
class GenerateGlobalHM:
    def __init__(self,
                 hm_size=(40, 40)):
        self.hm_size = hm_size
    def __call__(self, results):
        hm = np.zeros(self.hm_size, dtype='float32')
        uv = results['gt_uv_obj']
        hm_uv = np.int32(uv / np.array(results['ori_shape']) * np.array(self.hm_size))

        hm_vu = hm_uv[:, ::-1]
        hm[hm_vu[:, 0], hm_vu[:, 1]] = 1

        results['gt_hm'] = hm
        results['gt_hm_vu'] = hm_vu
        return results


# @PIPELINES.register_module()
# class GenerateGlobalHM:
#     def __init__(self,
#                  hm_size=(40, 40),
#                  max_hm_num=50):
#         self.hm_size = hm_size
#     def __call__(self, results):
#         hm = np.zeros(self.hm_size, dtype='float32')
#         uv = results['gt_uv_obj']
#         hm_uv = np.int32(uv / np.array(results['ori_shape']) * np.array(self.hm_size))
#
#         hm_vu = hm_uv[:, ::-1]
#         hm[hm_vu[:, 0], hm_vu[:, 1]] = 1
#
#         results['gt_hm'] = hm
#         results['gt_hm_vu'] = hm_vu
#         return results


@DATASETS.register_module()
class AmodalGraspDatasetByScene:
    def __init__(self,
                 data_root,
                 pipeline=None,
                 **kwargs):
        self.data_root = data_root
        self.scenes_cam_root = join(self.data_root, 'scenes_cam')
        self.ann_path = join(self.data_root, 'grasps_cam_by_scene.h5')

        self.h5 = h5py.File(self.ann_path, mode='r')
        self.ann = dict(self.h5)
        self.scene_id = list(self.ann.keys())

        self.CLASSES = None
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype='int32')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, item):
        scene_id = self.scene_id[item]
        ann = np.float32(self.ann[scene_id])
        data = np.load(join(self.scenes_cam_root, f'{scene_id}.npz'))

        # release ann
        gt_qual = ann[:, 8]
        project_success = ann[:, 16]
        project_success_uv = ann[:, 17]
        positive_valid = (project_success.astype('bool') & project_success_uv.astype('bool') & gt_qual.astype('bool'))
        ann = ann[positive_valid]

        gt_uv = ann[:, 9:11]
        gt_uv_obj = ann[:, 11:13]
        gt_xyz_obj = ann[:, 13:16]
        gt_quat = ann[:, :4]
        gt_xyz = ann[:, 4:7]
        gt_width = ann[:, 7]

        pc = np.float32(data['pc']).reshape(480, 480, 3)
        results = dict(img=pc,
                       gt_qual=gt_qual,
                       gt_width=gt_width,
                       gt_quat=gt_quat,
                       gt_uv=gt_uv,
                       gt_uv_obj=gt_uv_obj,
                       gt_xyz=gt_xyz,
                       gt_xyz_obj=gt_xyz_obj,
                       ori_shape=pc.shape[:2])

        if self.pipeline is not None:
            results = self.pipeline(results)
        return results



if __name__ == '__main__':
    dataset = AmodalGraspDatasetByScene(data_root='/disk1/data/giga/data_packed_train_raw')
    data = dataset[0]