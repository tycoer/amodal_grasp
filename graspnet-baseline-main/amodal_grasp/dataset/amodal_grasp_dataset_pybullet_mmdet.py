from mmdet.core.mask import BitmapMasks
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import DATASETS
from .amodal_grasp_dataset_pybullet import AmodalGraspDataset
import time
from mmdet.models.detectors.base import BaseDetector
import numpy as np
@DATASETS.register_module()
class AmodalGraspDatasetMMDet(AmodalGraspDataset):
    def __init__(self,
                 data_root,
                 split='train',
                 with_mesh=True,
                 with_pc=True,
                 with_nocs=True,
                 with_grasp=True,
                 img_norm_std=[0.229, 0.224, 0.225],
                 img_norm_mean=[0.485, 0.456, 0.406],
                 norm_quat=True,
                 pipeline=None,
                 test_mode=False,
                 **kwargs

                 ):
        super(AmodalGraspDatasetMMDet, self).__init__(
            data_root=data_root,
            split=split,
            with_mesh=with_mesh,
            with_pc=with_pc,
            with_nocs=with_nocs,
            with_grasp=with_grasp,
            img_norm_mean=img_norm_mean,
            img_norm_std=img_norm_std,
            norm_quat=norm_quat
        )
        self.test_mode = test_mode
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)
        self.flag = np.zeros(len(self), dtype='int32')

    def __getitem__(self, item):
        results = super().__getitem__(item)
        results = self.mmdet_format(results)
        if self.pipeline is not None:
            results = self.pipeline(results)
        return results

    def mmdet_format(self, results):
        # start = time.time()
        d, h, w = results['img'].shape
        results['gt_masks'] = DC(BitmapMasks(results['gt_masks'], height=h, width=w), cpu_only=True)
        results['gt_nocs'] = DC(results['gt_nocs'], cpu_only=True)
        results['gt_bboxes'] = DC(results['gt_bboxes'])
        results['gt_labels'] =  DC(results['gt_labels'])
        results['gt_grasps'] = DC(results['gt_grasps'], cpu_only=True)
        results['gt_voxels'] = DC(results['gt_voxels'], cpu_only=True)
        results['gt_meshes'] = DC(results['gt_meshes'], cpu_only=True)
        # results['depth'] = DC(results['depth'], cpu_only=True)
        results['pad_shape'] = (h, w, d)
        results['img_shape'] = (h, w, d)
        results['ori_shape'] = (h, w, d)
        results['ori_filename'] = results['filename']
        results['Ks'] =  DC(results['Ks'])
        # results['scale_factor'] = np.float32((0.4, 0.4, 0.4, 0.4))
        # if self.test_mode:
        #     results['img'] = [results['img']]

        return results
