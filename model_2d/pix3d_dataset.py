from mmdet.datasets.builder import DATASETS, PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.coco import CocoDataset
import os
from .utils.shape import read_voxel, transform_verts
import numpy as np
import torch
from pytorch3d.io import load_obj
from mmdet.core.mask import BitmapMasks
import cv2
import pickle
import matplotlib.pyplot as plt
from mmdet.datasets.api_wrappers import COCO
from .utils.metrics import compare_meshes
import tqdm

def xywh_to_xyxy(boxes):
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


@PIPELINES.register_module()
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

@PIPELINES.register_module()
class MinMaxNormalize:
    def __init__(self):
        pass
    def __call__(self, results):
        results['img'] = cv2.normalize(results['img'].astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        return results




@DATASETS.register_module()
class Pix3DDataset(CocoDataset):
    CLASSES = ['bed', 'bookcase', 'chair', 'desk', 'misc',
               'sofa', 'table', 'tool', 'wardrobe']

    PALETTE = [[255, 255, 25], # bed
               [230, 25, 75],  # bookcase
               [250, 190, 190],# chair
               [60, 180, 75],  # desk
               [230, 190, 255],# misc
               [0, 130, 200],  # sofa
               [245, 130, 48], # table
               [70, 240, 240], # tool
               [210, 245, 60]  # wardrobe
               ]


    def __init__(self,
                 data_root,
                 anno_path,
                 pipeline=None,
                 test_mode=False
                 ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.model_root = os.path.join(self.data_root, 'model')
        self.test_mode = test_mode


        self.coco = COCO(self.anno_path)
        self.annotations = self.coco.dataset['annotations']
        self.images = self.coco.dataset['images']

        if self.test_mode:
            print('Coverting segmentation from mask to polygon for COCO evaluation required format.')
            self.mask2poly()
            self.cat_ids = sorted(self.coco.getCatIds())
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()


        self.invalid = ["img/table/1749.jpg", "img/table/0045.png"]
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
                        with open(obj_abs_path, 'rb') as f:
                            mesh = load_obj(f, load_textures = False)
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
        # if images['file_name'] in self.invalid:
        #     random_item = np.random.choice(len(self))
        #     results = self.__getitem__(random_item)

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
                item_random = np.random.choice(len(self))
                results = self[item_random]
        return results

    def __len__(self):
        return len(self.annotations)


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):


        metrics = {}
        # evaluate coco: include mask and bbox metrics
        coco_metric = super().evaluate(results=results,
                         metric=metric,
                         logger=logger,
                         jsonfile_prefix=jsonfile_prefix,
                         classwise=classwise,
                         proposal_nums=proposal_nums,
                         iou_thrs=iou_thrs)

        # evaluate shape: include mesh, voxel and z metrics

        for res in results:
            pred_meshes = results['pred_meshes']
            gt_mesh = self[i]
            shape_metric = compare_meshes(pred_meshes, gt_mesh, reduce=False)



        metrics.update(coco_metric)
        return metrics

    def mask2poly(self):
        for ann in tqdm.tqdm(self.annotations):
            mask = cv2.imread(os.path.join(self.data_root, ann['segmentation']), 0)
            poly, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            poly = np.concatenate(poly).flatten()
            ann['segmentation'] = [poly.tolist()]



@PIPELINES.register_module()
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

        # 可视化verts_camera, voxels_camera后相对于图片中物体是 既水平翻转又竖直反转的, 因此将 x, y 轴反转
        # 但貌似 batch_crop_meshes_within_box 中源码做了一次反转, 因次此处暂不做翻转
        # verts_camera[:, 0] = -verts_camera[:, 0]
        # verts_camera[:, 1] = -verts_camera[:, 1]
        # voxels_camera[:, 0] = -voxels_camera[:, 0]
        # voxels_camera[:, 1] = -voxels_camera[:, 1]


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