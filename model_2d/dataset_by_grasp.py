import os
import numpy as np
from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.datasets.pipelines import Compose
import json
import trimesh
import pandas as pd


def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized, centroid, m


def normalize_uv(uv, w, h):
    return uv / np.array([w, h])


@DATASETS.register_module()
class AmodelGraspDatasetByGrasp:
    def __init__(self,
                 data_root,
                 pipeline=None,
                 **kwargs):
        self.grasp_path = os.path.join(data_root, "grasps_cam.csv")
        self.scene_cam_root = os.path.join(data_root, 'scenes_cam')
        self.setup_path = os.path.join(data_root, 'setup_rerender.json')

        with open(self.setup_path, 'r') as f:
            self.setup = json.load(f)

        self.K = self.setup['intrinsic']['K']
        self.w, self.h = self.setup['intrinsic']['width'], self.setup['intrinsic']['height']
        self.size = self.setup['size']
        self.fx, self.fy, self.cx, self.cy = self.K[0], self.K[4], self.K[2], self.K[5]

        self.grasp = pd.read_csv(self.grasp_path)
        uv_obj = self.grasp[['u_obj', 'v_obj']]
        self.valid = np.logical_not((uv_obj > 255).any(axis=1) | (uv_obj < 0).any(axis=1)) # 去除坐标小于0, 大于255的grasp
        self.grasp = self.grasp[self.valid]

        self.pipeline =pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(pipeline)

        # mmdet format
        self.flag = np.zeros(len(self), dtype='int64')
        self.CLASSES = None

    def __getitem__(self, item):
        # realease data
        grasp = self.grasp.iloc[item]
        scene_id = grasp['scene_id']
        grasp_values = np.float32(grasp.values[1:])
        grasp_quat = grasp_values[0:4]
        grasp_xyz = grasp_values[4:7]
        grasp_width = grasp_values[7]
        grasp_qual = grasp_values[8]
        grasp_uv = grasp_values[9:11]
        grasp_uv_on_obj = grasp_values[11:13]
        grasp_xyz_on_obj = grasp_values[13:16]
        process_success = grasp_values[16]

        # normalize uv
        grasp_uv_on_obj = grasp_uv_on_obj / np.float32([self.w, self.h])
        grasp_uv = grasp_uv / np.float32([self.w, self.h])
        grasp_vu = grasp_uv[::-1]
        grasp_vu_on_obj = grasp_uv_on_obj[::-1]


        pc_path = os.path.join(self.scene_cam_root, scene_id + '.npz')
        pc = np.load(pc_path, allow_pickle=True)['pc']
        pc = np.float32(pc).reshape(self.h, self.w, 3)
        # normalize pc and keypoints
        # pc, centroid, m = normalize_point_cloud(pc)
        # grasp_xyz_on_obj = ((grasp_xyz_on_obj - centroid) / m).astype('float32')
        # grasp_xyz = ((grasp_xyz - centroid) / m)

        # reshape pc to img
        results = dict(img=pc.astype('float32'),
                       scene_id=scene_id,
                       item=item,
                       gt_grasp_vu=grasp_vu.astype('float32'),
                       gt_grasp_vu_on_obj=grasp_vu_on_obj.astype('float32'),
                       gt_grasp_uv=grasp_uv.astype('float32'),
                       gt_grasp_uv_on_obj=grasp_uv_on_obj.astype('float32'),
                       # gt_grasp_qual=grasp_qual.astype('long'),
                       gt_grasp_qual=grasp_qual.astype('float32'),
                       gt_grasp_quat=grasp_quat.astype('float32'),
                       gt_grasp_xyz=grasp_xyz.astype('float32'),
                       gt_grasp_xyz_on_obj=grasp_xyz_on_obj.astype('float32'),
                       gt_grasp_width=grasp_width.astype('float32'),

                       process_success=process_success)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results

    def __len__(self):
        return len(self.grasp)

    def giga_offical_process(self, results):
        results['gt_grasp_xyz_on_obj'] = results['gt_grasp_xyz_on_obj'] / self.size - 0.5

    def filter_grasp_by_uv(self):
        uv_obj = self.grasp[['u_obj', 'v_obj']]
        valid = ~(uv_obj > self.w - 1 and uv_obj < 0).all(axis=1)
        self.grasp = self.grasp[valid]
        return self.grasp

if __name__ == '__main__':
    dataset = AmodelGraspDatasetByGrasp(data_root='/disk1/data/giga/data_packed_train_raw')
    res = dataset[0]