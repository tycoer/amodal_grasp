from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.datasets.pipelines import Compose
import os
import json
from .utils.shape import read_voxel, transform_verts
import numpy as np
import torch
from pytorch3d.io import load_obj
from mmdet.core.mask import BitmapMasks
import cv2
import pickle
from mmdet.datasets.pipelines import LoadAnnotations
from mmcv.parallel import DataContainer as DC
import matplotlib.pyplot as plt
from collections import OrderedDict
import contextlib
import io
import logging
import os
from pycocotools.coco import COCO


def xywh_to_xyxy(boxes):
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


# @PIPELINES.register_module()
class LoadAnnotationsPix3D(LoadAnnotations):
    def _load_masks(self, results):
        masks_abs_path = os.path.join(results['img_prefix'],
                                      results['ann_info']['masks'])
        masks = cv2.imread(masks_abs_path, cv2.IMREAD_UNCHANGED)
        masks = masks / 255
        masks = np.atleast_3d(masks).astype('float32')

        masks = BitmapMasks(masks, width=masks.shape[1], height=masks.shape[0])
        results['gt_masks'] = masks
        results['mask_fields'].append('gt_masks')
        return results

# @PIPELINES.register_module()
class MinMaxNormalize:
    def __init__(self):
        pass
    def __call__(self, results):
        results['img'] = cv2.normalize(results['img'].astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        return results




# @DATASETS.register_module()
class Pix3DDatasetCOCO:
    CLASSES = ['bed', 'bookcase', 'chair', 'desk', 'misc',
               'sofa', 'table', 'tool', 'wardrobe']
    def __init__(self,
                 data_root,
                 anno_path,
                 pipeline=None,
                 ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.model_root = os.path.join(self.data_root, 'model')

        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(self.anno_path)

        with open(self.anno_path, 'r') as f:
            self.anno = json.load(f)
            self.annotations = self.anno['annotations']
            self.images = self.anno['images']


        ################# preload meshes ############################
        self.mesh_cache_path = os.path.join(self.data_root, 'model.pickle')
        self.all_meshes = {}
        if os.path.exists(self.mesh_cache_path):
            print(f'Model cache {self.mesh_cache_path} is founded, loading the cache.')
            with open(self.mesh_cache_path, 'rb') as f:
                self.all_meshes = pickle.load(f)

        else:
            print('Preload meshes ...')
            data_root_len = len(self.data_root) if self.data_root.endswith('/') else len(self.data_root + '/')
            for i, j, k in os.walk(self.model_root):
                for filename in k:
                    if filename.endswith('obj'):
                        obj_abs_path = os.path.join(i, filename)
                        mesh = load_obj(obj_abs_path, load_textures = False)
                        verts, faces = mesh[0], mesh[1].verts_idx
                        self.all_meshes[obj_abs_path[data_root_len:]] = (verts, faces)

            with open(self.mesh_cache_path, 'wb') as f:
                pickle.dump(self.all_meshes, f)
            print(f'Save meshes cache to {self.mesh_cache_path}!')
        #######################################################################

        self.flag = np.zeros(len(self), dtype='int64')

        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(self.pipeline)

    def __getitem__(self, item):
        results = {}
        annotations = self.annotations[item]
        images = self.images[item]

        # mmdet pipeline
        # mmdet需要的格式
        results['img_prefix'] = self.data_root
        results['img_info'] = dict(filename=images['file_name'],
                                   height=images['height'],
                                   width=images['width'])

        # bbox xywh to xyxy
        bbox_xywh = np.atleast_2d(annotations['bbox'])
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        bbox_xyxy = bbox_xyxy.astype('float32')

        results['ann_info'] = dict(bboxes=bbox_xyxy,
                                   labels=np.atleast_1d(annotations['category_id'] - 1).astype('int64'), # pix3d中的 label 从 1 开始, 但 mmdet 要求从 0 开始
                                   masks=annotations['segmentation'])
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['flip'] = 0
        results['flip_direction'] = 'horizontal'

        results['R'] = annotations['rot_mat']
        results['t'] = annotations['trans_mat']
        results['model'] = annotations['model']
        results['image_id'] = annotations['image_id']
        results['iscrowd'] = annotations['iscrowd']
        results['K'] = annotations['K']
        results['gt_voxels'] = read_voxel(os.path.join(self.data_root, annotations['voxel']))
        results['gt_meshes'] = self.all_meshes[annotations['model']]
        results['image_id'] = annotations['image_id']
        results['item'] = item
        if self.pipeline is not None:
            try:
                results = self.pipeline(results) # pix3d 中有无效数据 典型如 img/table/1749.jpg, 与mask/table/1749.png 对不上
            except:
                random_item = np.random.choice(len(self))
                results = self.__getitem__(random_item)
        return results

    def __len__(self):
        return len(self.annotations)



# @PIPELINES.register_module()
class Pix3DPipeline:
    def __init__(self,
                 with_dz=True):
        self.with_dz = with_dz

    def __call__(self, results):
        K = results['K']
        h, w = results['img'].shape[:2]
        # update K
        # camera intr
        K = torch.as_tensor([K[0], w / 2, h / 2], dtype=torch.float32)
        # camera extr
        R = torch.as_tensor(results['R'], dtype=torch.float32)
        t = torch.as_tensor(results['t'], dtype=torch.float32)
        if self.with_dz:
            dz =  self._process_dz(mesh=results['gt_meshes'],
                                   focal_length=K[0],
                                   R=R,
                                   t=t)
        # process meshes and voxels
        verts, faces = results['gt_meshes']
        voxels = results['gt_voxels']
        # transform vertices to camera coordinate system
        verts_camera = transform_verts(verts, R, t)
        voxels_camera = transform_verts(voxels, R, t)


        verts_camera[:, 0] = -verts_camera[:, 0]
        verts_camera[:, 1] = -verts_camera[:, 1]
        voxels_camera[:, 0] = -voxels_camera[:, 0]
        voxels_camera[:, 1] = -voxels_camera[:, 1]


        # augmentation
        if 'scale_factor' in results:
            w_scale_factor, h_scale_factor = results['scale_factor'][:2]
            verts_camera = self.resize_coords(verts_camera, w_scale_factor, h_scale_factor)
            voxels_camera = self.resize_coords(voxels_camera, w_scale_factor, h_scale_factor)
            # NOTE normalize the dz by the height scaling of the image.
            # This is necessary s.t. z-regression targets log(dz/roi_h)
            # are invariant to the scaling of the roi_h
            dz = dz * h_scale_factor
        if 'flip' in results and 'flip_direction' in results:
            if results['flip']:
                if results['flip_direction'] == 'horizontal':
                    verts_camera[:, 0] = -verts_camera[:, 0]
                    voxels_camera[:, 0] = -voxels_camera[:, 0]
                if results['flip_direction'] == 'vertical':
                    verts_camera[:, 1] = -verts_camera[:, 1]
                    voxels_camera[:, 1] = -voxels_camera[:, 1]

        results['gt_meshes'] = DC([verts_camera, faces])
        results['gt_voxels'] =  DC([voxels_camera])
        results['Ks'] = DC(torch.atleast_2d(K))
        results['gt_zs'] = DC(torch.atleast_2d(dz))
        return results


    def _process_dz(self, mesh, focal_length=1.0, R=None, t=None):
        # clone mesh
        verts, faces = mesh
        # transform vertices to camera coordinate system
        verts = transform_verts(verts, R, t)

        dz = verts[:, 2].max() - verts[:, 2].min()
        z_center = (verts[:, 2].max() + verts[:, 2].min()) / 2.0
        dz = dz / z_center
        dz = dz * focal_length
        return dz

    def resize_coords(self, coords, w_scale_factor, h_scale_factor):
        coords[:, 0] = coords[:, 0] * w_scale_factor  # resize x
        coords[:, 1] = coords[:, 1] * h_scale_factor  # resize y
        return coords

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results