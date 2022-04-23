import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import build_loss, HEADS
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from .coord import coord_target


@HEADS.register_module()
class NOCSHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 num_bins=-1,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_coord=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(NOCSHead, self).__init__()

        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_coord = build_loss(loss_coord)

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)

        self.convs_x = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_x.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.convs_x.append(nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio))

        self.convs_y = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_y.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.convs_y.append(nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio))

        self.convs_z = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_z.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        self.convs_z.append(nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio))

        if self.num_bins > 0:
            out_channels = self.num_bins
        else:
            out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)

        self.conv_logits_x = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits_y = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.conv_logits_z = nn.Conv2d(logits_in_channel, out_channels, 1)

        self.debug_imgs = None

    def init_weights(self):
        for m in [self.conv_logits_x, self.conv_logits_y, self.conv_logits_z]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x, **kwargs):
        for conv in self.convs_x:
            coord_x = conv(x)
        for conv in self.convs_y:
            coord_y = conv(x)
        for conv in self.convs_z:
            coord_z = conv(x)

        num_rois, _, roi_height, roi_width = x.size()
        roi_height, roi_width = roi_height * self.upsample_ratio, roi_width * self.upsample_ratio

        coord_x = self.conv_logits_x(coord_x)
        if self.num_bins > 0:
            coord_x = coord_x.permute(0, 2, 3, 1).view(num_rois, roi_height, roi_width, -1, self.num_bins)
            coord_x = F.softmax(coord_x, -1)
        else:
            coord_x = F.sigmoid(coord_x)

        coord_y = self.conv_logits_x(coord_y)
        if self.num_bins > 0:
            coord_y = coord_y.permute(0, 2, 3, 1).view(num_rois, roi_height, roi_width, -1, self.num_bins)
            coord_y = F.softmax(coord_y, -1)
        else:
            coord_y = F.sigmoid(coord_y)

        coord_z = self.conv_logits_x(coord_z)
        if self.num_bins > 0:
            coord_z = coord_z.permute(0, 2, 3, 1).view(num_rois, roi_height, roi_width, -1, self.num_bins)
            coord_z = F.softmax(coord_z, -1)
        else:
            coord_z = F.sigmoid(coord_z)

        return coord_x, coord_y, coord_z

    def get_target(self, sampling_results, gt_coords, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        coord_targets = coord_target(pos_proposals, pos_assigned_gt_inds,
                                     gt_coords, rcnn_train_cfg)
        return coord_targets

    @force_fp32(apply_to=('nocs_pred', ))
    def loss(self, nocs_pred, nocs_targets, labels, mask_targets):
        loss = dict()
        if self.class_agnostic:
            loss_coord = self.loss_coord(nocs_pred, nocs_targets,
                                       torch.zeros_like(labels), mask_targets)
        else:
            loss_coord = self.loss_coord(nocs_pred, nocs_targets, labels, mask_targets, use_bins=self.num_bins > 0)
        loss_coord_x, loss_coord_y, loss_coord_z = loss_coord[0], loss_coord[1], loss_coord[2]
        loss['loss_coord_x'] = loss_coord_x
        loss['loss_coord_y'] = loss_coord_y
        loss['loss_coord_z'] = loss_coord_z
        return loss

    def get_nocs(self, nocs_pred, det_bboxes, det_labels, rcnn_test_cfg,
                 ori_shape, scale_factor, window, rescale):
        if isinstance(nocs_pred, tuple):
            nocs_coord_x = nocs_pred[0].cpu().numpy()
            nocs_coord_y = nocs_pred[1].cpu().numpy()
            nocs_coord_z = nocs_pred[2].cpu().numpy()
        else:
            raise TypeError('nocs_pred is expected to be a tuple')
        # num_bins = nocs_pred[0].size(-1)
        cls_nocs = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if self.num_bins > 0:
            nocs_coord_x = np.argmax(nocs_coord_x, -1).astype(np.float32) / self.num_bins
            nocs_coord_y = np.argmax(nocs_coord_y, -1).astype(np.float32) / self.num_bins
            nocs_coord_z = np.argmax(nocs_coord_z, -1).astype(np.float32) / self.num_bins

        nocs_coord = np.stack([nocs_coord_x, nocs_coord_y, nocs_coord_z], axis=-1)

        if rescale:
            img_h, img_w = ori_shape[:2]
            shift = window[:2]
            shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
            shifts = np.array([0, 0, 0, 0])

        for i in range(bboxes.shape[0]):
            # bbox = ((bboxes[i, :] - shifts) / scale_factor).astype(np.int32)
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                nocs_coord_ = nocs_coord[i, :, :, label, :]
            else:
                nocs_coord_ = nocs_coord[i, :, :, 0, :]
            im_nocs = np.zeros((img_h, img_w, 3), dtype=np.float32)

            bbox_nocs = mmcv.imresize(nocs_coord_, (w, h))
            im_nocs[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w, :] = bbox_nocs
            cls_nocs[label - 1].append(im_nocs)

        return cls_nocs
