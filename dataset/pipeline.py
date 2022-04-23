from mmdet.datasets.builder import PIPELINES
import scipy
import os.path as osp
from mmdet.datasets.pipelines.formating import to_tensor
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.core.mask import BitmapMasks
@PIPELINES.register_module()
class ResizeNOCS(object):
    def __init__(self, min_dim, max_dim, padding=True, no_depth=True):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.padding = padding
        self.no_depth = no_depth

    def _resize_img(self, image):
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        self.window = (0, 0, h, w)
        self.scale = 1

        # Scale?
        if self.min_dim:
            # Scale up but not down
            self.scale = max(1, self.min_dim / min(h, w))
        # Does it exceed max dim?
        if self.max_dim:
            image_max = max(h, w)
            if round(image_max * self.scale) > self.max_dim:
                self.scale = self.max_dim / image_max
        # Resize image and mask
        if self.scale != 1:
            image = scipy.misc.imresize(
                image, (round(h * self.scale), round(w * self.scale)))
        # Need padding?
        if self.padding:
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (self.max_dim - h) // 2
            bottom_pad = self.max_dim - h - top_pad
            left_pad = (self.max_dim - w) // 2
            right_pad = self.max_dim - w - left_pad
            if len(image.shape) == 3:
                self.padding_size = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            else:
                self.padding_size = [(top_pad, bottom_pad), (left_pad, right_pad)]
            image = np.pad(image, self.padding_size, mode='constant', constant_values=0)
            self.window = (top_pad, left_pad, h + top_pad, w + left_pad)
        else:
            self.padding_size = [(0, 0), (0, 0), (0, 0)]

        return image

    def _resize_mask(self, mask):
        h, w = mask.shape[:2]
        # for instance mask
        if len(mask.shape) == 3:
            mask = scipy.ndimage.zoom(mask, zoom=[self.scale, self.scale, 1], order=0)
            new_padding = self.padding_size
        # for coordinate map
        elif len(mask.shape) == 4:
            mask = scipy.ndimage.zoom(mask, zoom=[self.scale, self.scale, 1, 1], order=0)
            new_padding = self.padding_size + [(0, 0)]
        else:
            assert False
        mask = np.pad(mask, new_padding, mode='constant', constant_values=0)

        return mask

    def __call__(self, results):
        results['img'] = self._resize_img(results['img'])
        if not self.no_depth:
            results['depth'] = self._resize_img(results['depth'])
        if 'gt_masks' in results:
            results['gt_masks'] = self._resize_mask(results['gt_masks'])
        if 'gt_coords' in results:
            results['gt_coords'] = self._resize_mask(results['gt_coords'])
        results['img_shape'] = results['img'].shape
        # if self.padding:
        results['pad_shape'] = results['img'].shape

        results['window'] = self.window
        results['scale_factor'] = np.array((self.scale, self.scale))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__

        return repr_str



@PIPELINES.register_module()
class ExtractBBoxFromMask(object):

    def __init__(self):
        pass

    def __call__(self, results):
        mask = results['gt_masks']
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])

        results['gt_bboxes'] = boxes.astype(np.float32)
        results['bbox_fields'].append('gt_bboxes')

        return results

    def __repr__(self):
        return self.__class__.__name__



@PIPELINES.register_module()
class LoadAnnotationsNOCS(object):
    def __init__(self,
                 with_mask=True,
                 with_coord=False):
        self.with_mask = with_mask
        self.with_coord = with_coord

    @staticmethod
    def _load_masks(results):
        mask_path = osp.join(results['img_prefix'],
                             results['mask_path'])
        gt_masks = mmcv.imread(mask_path)[:, :, 2]
        results['gt_masks'] = gt_masks

        return results

    @staticmethod
    def _load_coords(results):
        coord_path = osp.join(results['img_prefix'],
                              results['coord_path'])
        gt_coords = mmcv.imread(coord_path)[:, :, :3]
        gt_coords = gt_coords[:, :, (2, 1, 0)] # shape (480, 640, 3)
        results['gt_coords'] = gt_coords

        return results

    @staticmethod
    def process_data(results):
        cdata = results['gt_masks']
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata==255] = -1
        assert(np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = results['gt_coords']
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        meta_path = osp.join(results['img_prefix'],
                             results['meta_path'])
        obj_model_dir = results['obj_model_dir']
        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = osp.join(obj_model_dir, words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = osp.join(obj_model_dir, words[2] + '.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = osp.join(obj_model_dir, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]
        i = 0

        # delete ids of background objects and non-existing objects
        inst_dict = results['inst']
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]

        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        results['gt_masks'] = masks[:, :, :i]
        results['mask_fields'].append('gt_masks')
        coords = coords[:, :, :i, :]
        results['gt_coords'] = np.clip(coords, 0, 1)
        results['coord_fields'].append('gt_coords')

        results['gt_labels'] = class_ids[:i]
        results['scales'] = scales[:i]

        return results

    def __call__(self, results):
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_coord:
            results = self._load_coords(results)

        self.process_data(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_mask={}, with_coord={}, with_meta={})'
                     ).format(self.with_mask, self.with_coord, self.with_meta)
        return repr_str


@PIPELINES.register_module()
class LoadColorAndDepthFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        color_path = osp.join(results['img_prefix'],
                              results['color_path'])
        depth_path = osp.join(results['img_prefix'],
                              results['depth_path'])

        color_img = mmcv.imread(color_path)
        depth_img = mmcv.imread(depth_path, -1)
        if len(depth_img.shape) == 3:
            depth_img = np.uint8(depth_img[:, :, 1]*256) + np.uint8(depth_img[:, :, 2])
        depth_img = depth_img.astype(np.uint8)

        if self.to_float32:
            color_img = color_img.astype(np.float32)
            depth_img = depth_img.astype(np.float32)

        assert color_img.shape[:2] == depth_img.shape[:2]
        results['img_path'] = color_path
        results['depth_path'] = color_path
        results['img'] = color_img
        results['depth'] = depth_img
        results['img_shape'] = color_img.shape
        results['ori_shape'] = color_img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

@PIPELINES.register_module()
class DefaultFormatBundleNOCS(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = to_tensor(img)
        if 'depth' in results:
            # depth = np.ascontiguousarray(results['depth'].transpose(2, 0, 1))
            results['depth'] = DC(to_tensor(results['depth']), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_labels', 'scales']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            gt_masks = results['gt_masks']
            results['gt_masks'] = BitmapMasks(gt_masks.transpose(2, 0, 1), height=gt_masks.shape[0], width=gt_masks.shape[1])
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_coords' in results:
            results['gt_coords'] = DC(results['gt_coords'].transpose(2, 0, 1, 3), cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__