from mmdet.core.mask import BitmapMasks
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import DATASETS
from .amodal_grasp_dataset import GraspNetDatasetForAmodalGrasp
import time
from mmdet.models.detectors.base import BaseDetector
import numpy as np
@DATASETS.register_module()
class GraspNetDatasetForAmodalGraspMMDet(GraspNetDatasetForAmodalGrasp):
    def __init__(self,
                 data_root,
                 camera='kinect',
                 split='train',
                 with_mesh=True,
                 with_pc=True,
                 with_nocs=True,
                 with_grasp=True,
                 per_instance_max_grasp=16,
                 global_max_grasp=32,
                 pipeline=None,
                 test_mode=False,
                 **kwargs

                 ):
        super(GraspNetDatasetForAmodalGraspMMDet, self).__init__(
            data_root=data_root,
            camera=camera,
            split=split,
            with_mesh=with_mesh,
            with_pc=with_pc,
            with_nocs=with_nocs,

        )
        self.test_mode = test_mode
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)

    def __getitem__(self, item):
        # start = time.time()
        results = super(GraspNetDatasetForAmodalGraspMMDet, self).__getitem__(item)
        # print(time.time() - start)
        # start1 = time.time()
        results = self.mmdet_format(results)
        if self.pipeline is not None:
            results = self.pipeline(results)
        # print('total_time', time.time() - start)
        return results

    def mmdet_format(self, results):
        # start = time.time()
        d, h, w = results['img'].shape
        results['gt_masks'] = DC(BitmapMasks(results['gt_masks'], height=h, width=w), cpu_only=True)
        results['gt_nocs'] = DC(results['gt_nocs'], cpu_only=True)
        results['gt_bboxes'] = DC(results['gt_bboxes'])
        results['gt_labels'] =  DC(results['gt_labels'])
        results['gt_grasps'] = DC(results['gt_grasps'], cpu_only=True)
        results['gt_meshes'] = DC(results['gt_meshes'], cpu_only=True)
        results['gt_voxels'] = DC(results['gt_voxels'], cpu_only=True)
        # results['depth'] = DC(results['depth'], cpu_only=True)
        results['pad_shape'] = (h, w, d)
        results['img_shape'] = (h, w, d)
        results['ori_shape'] = (480, 480, d)
        results['ori_filename'] = results['filename']
        # results['scale_factor'] = np.float32((0.4, 0.4, 0.4, 0.4))
        # if self.test_mode:
        #     results['img'] = [results['img']]

        return results
